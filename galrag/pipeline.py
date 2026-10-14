"""Pipeline orchestration: harvest -> fetch -> parse -> chunk -> embed.

Every stage is idempotent and driven by the ``status`` column in SQLite, so
it can be interrupted and resumed.  Network stages are serial (rate-limited
per host); CPU stages are batched.
"""

from __future__ import annotations

import logging
import signal
import time
from collections.abc import Iterable

from galrag.chunk import chunk_abstract_only, chunk_document
from galrag.db import Database
from galrag.fetch import FetchError, fetch_fulltext
from galrag.models import Chunk, Document, PaperMeta, Status
from galrag.parse import load_document, parse_fetched, resolve_bibliography, save_document

log = logging.getLogger(__name__)


class _Stop:
    """Cooperative stop on SIGINT/SIGTERM so the current paper finishes cleanly."""

    def __init__(self) -> None:
        self.flag = False
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._set)
            except ValueError:  # not main thread
                pass

    def _set(self, *_):
        log.warning("stop requested; finishing current item")
        self.flag = True


def fetch_and_parse_one(db: Database, meta: PaperMeta, *, force: bool = False) -> str:
    """Fetch full text for one paper, parse it, resolve its bibliography, save the
    Document and advance status to PARSED.  Returns the source used."""
    try:
        res = fetch_fulltext(meta, force=force)
    except FetchError as e:
        db.set_status(meta.paper_key, Status.FAILED_FETCH, error=str(e)[:500])
        raise
    db.set_status(meta.paper_key, Status.FETCHED, source=res.source)
    try:
        doc = parse_fetched(res, meta)
        n_res = resolve_bibliography(doc, db, meta)
        save_document(doc)
    except Exception as e:
        db.set_status(meta.paper_key, Status.FAILED_PARSE, source=res.source, error=repr(e)[:500])
        raise
    db.set_status(meta.paper_key, Status.PARSED, source=res.source)
    log.info("%s <- %s (%d chars, %d sections, %d/%d bib resolved)", meta.paper_key, res.source,
             len(doc.full_text), len(doc.sections), n_res, len(doc.bibliography))
    return res.source


def run_fetch_parse(db: Database, *, tiers: Iterable[str] = ("A", "B"), limit: int | None = None,
                    paper_keys: Iterable[str] | None = None, retry_failed: bool = False) -> dict[str, int]:
    """Process papers in priority order.  Returns counts per outcome."""
    stop = _Stop()
    counts: dict[str, int] = {"ok": 0, "fetch_failed": 0, "parse_failed": 0}
    if paper_keys:
        keys = [k for k in paper_keys if db.get_row(k) is not None]
    else:
        statuses = [Status.METADATA] + ([Status.FAILED_FETCH, Status.FAILED_PARSE] if retry_failed else [])
        keys = [k for k, _ in db.iter_keys_by_status(statuses, tiers=tiers, limit=limit)]
    rows = keys
    log.info("fetch/parse queue: %d papers (tiers=%s)", len(rows), list(tiers))
    t0 = time.time()
    for i, key in enumerate(rows, 1):
        if stop.flag:
            break
        row = db.get_row(key)
        if row is None:
            continue
        meta = PaperMeta.model_validate_json(row["meta_json"])
        if row["status"] in (Status.PARSED, Status.CHUNKED, Status.EMBEDDED) and not paper_keys:
            continue
        try:
            fetch_and_parse_one(db, meta)
            counts["ok"] += 1
        except FetchError as e:
            counts["fetch_failed"] += 1
            log.warning("fetch failed %s: %s", meta.paper_key, str(e)[:200])
        except Exception:
            counts["parse_failed"] += 1
            log.exception("parse failed %s", meta.paper_key)
        if i % 50 == 0:
            rate = i / (time.time() - t0)
            log.info("progress %d/%d (%.2f papers/s) %s", i, len(rows), rate, counts)
    return counts


def iter_parsed_without_embedding(db: Database, limit: int | None = None,
                                  paper_keys: Iterable[str] | None = None):
    """Yield (meta, Document) for papers that are parsed but not yet in the vector store."""
    if paper_keys:
        rows = [r for r in (db.get_row(k) for k in paper_keys) if r is not None]
    else:
        rows = db.iter_by_status(Status.PARSED, limit=limit)
    for row in rows:
        doc = load_document(row["paper_key"])
        if doc is None:
            db.set_status(row["paper_key"], Status.FAILED_PARSE, error="document file missing")
            continue
        yield PaperMeta.model_validate_json(row["meta_json"]), doc


# ---------------------------------------------------------------------------
# chunk + embed stage
# ---------------------------------------------------------------------------


def _paper_info(meta: PaperMeta) -> dict[str, str]:
    n = len(meta.authors or [])
    fa = meta.first_author or (meta.authors[0] if meta.authors else "")
    fa = fa.split(",")[0].strip()
    if n <= 1:
        short = fa
    elif n == 2:
        second = meta.authors[1].split(",")[0].strip()
        short = f"{fa} & {second}"
    else:
        short = f"{fa} et al."
    return {"title": meta.title or "", "authors_short": f"{short} ({meta.year})" if meta.year else short}


def embed_documents(db: Database, store, embedder, items: list[tuple[PaperMeta, Document]]) -> int:
    """Chunk, embed and upsert a batch of parsed documents; set status per paper.
    ``embedder`` needs ``embed_passages(texts) -> ndarray``.  Returns #chunks written."""
    chunks: list[Chunk] = []
    per_paper: dict[str, int] = {}
    info: dict[str, dict[str, str]] = {}
    for meta, doc in items:
        try:
            cs = chunk_document(doc, meta)
        except Exception as e:
            db.set_status(meta.paper_key, Status.FAILED_EMBED, error=f"chunk: {e!r}"[:500])
            log.exception("chunking failed %s", meta.paper_key)
            continue
        if not cs:
            db.set_status(meta.paper_key, Status.FAILED_EMBED, error="no chunks produced")
            continue
        chunks.extend(cs)
        per_paper[meta.paper_key] = len(cs)
        info[meta.paper_key] = _paper_info(meta)
    if not chunks:
        return 0
    try:
        vecs = embedder.embed_passages([c.text for c in chunks])
        store.upsert_chunks(chunks, vecs, paper_info=info)
    except Exception as e:
        for key in per_paper:
            db.set_status(key, Status.FAILED_EMBED, error=f"embed: {e!r}"[:500])
        raise
    for key, n in per_paper.items():
        db.set_status(key, Status.EMBEDDED, n_chunks=n)
    return len(chunks)


def run_embed(db: Database, store, embedder, *, limit: int | None = None,
              paper_keys: Iterable[str] | None = None, chunks_per_batch: int = 2000) -> dict[str, int]:
    """Embed all PARSED papers (priority order) in batches of ~``chunks_per_batch`` chunks."""
    stop = _Stop()
    counts = {"papers": 0, "chunks": 0, "failed": 0}
    batch: list[tuple[PaperMeta, Document]] = []
    est = 0
    t0 = time.time()

    def flush():
        nonlocal batch, est
        if not batch:
            return
        before = db.count_by_status().get(Status.FAILED_EMBED, 0)
        n = embed_documents(db, store, embedder, batch)
        counts["chunks"] += n
        counts["papers"] += len(batch)
        counts["failed"] += db.count_by_status().get(Status.FAILED_EMBED, 0) - before
        rate = counts["chunks"] / max(1e-6, time.time() - t0)
        log.info("embedded %d papers / %d chunks (%.1f chunks/s)", counts["papers"], counts["chunks"], rate)
        batch, est = [], 0

    for meta, doc in iter_parsed_without_embedding(db, limit=limit, paper_keys=paper_keys):
        if stop.flag:
            break
        batch.append((meta, doc))
        est += max(1, len(doc.full_text) // 1400)  # ~350 tokens per chunk
        if est >= chunks_per_batch:
            flush()
    flush()
    return counts


def run_embed_abstracts(db: Database, store, embedder, *, limit: int | None = None,
                        batch_size: int = 2000) -> dict[str, int]:
    """Give every paper a title+abstract vector in the paper-level table, and papers
    without full text an abstract-only chunk so the whole corpus is searchable."""
    stop = _Stop()
    counts = {"papers": 0, "abstract_chunks": 0}
    rows: list = []
    t0 = time.time()

    def flush():
        nonlocal rows
        if not rows:
            return
        metas = [PaperMeta.model_validate_json(r["meta_json"]) for r in rows]
        abs_chunks: list[tuple[int, Chunk]] = []
        for i, (r, m) in enumerate(zip(rows, metas)):
            cur = db.get_row(m.paper_key)  # re-read: the full-text embed job may have advanced it
            if cur is not None and cur["status"] != Status.EMBEDDED:
                cs = chunk_abstract_only(m)
                if cs:
                    abs_chunks.append((i, cs[0]))
        texts = [f"{m.title}\n\n{m.abstract or ''}".strip() or m.title or m.paper_key for m in metas]
        vecs = embedder.embed_passages(texts)
        store.upsert_papers(metas, vecs)
        abs_keys: list[str] = []
        if abs_chunks:
            idx = [i for i, _ in abs_chunks]
            cs = [c for _, c in abs_chunks]
            store.upsert_chunks(cs, vecs[idx], paper_info={metas[i].paper_key: _paper_info(metas[i]) for i in idx})
            abs_keys = [c.paper_key for c in cs]
        db.mark_paper_vec([m.paper_key for m in metas], abs_chunk_keys=abs_keys)
        counts["papers"] += len(metas)
        counts["abstract_chunks"] += len(abs_keys)
        rate = counts["papers"] / max(1e-6, time.time() - t0)
        log.info("paper vectors %d (+%d abstract chunks) (%.1f papers/s)", counts["papers"],
                 counts["abstract_chunks"], rate)
        rows = []

    for row in list(db.iter_without_paper_vec(limit=limit)):
        if stop.flag:
            break
        rows.append(row)
        if len(rows) >= batch_size:
            flush()
    flush()
    return counts
