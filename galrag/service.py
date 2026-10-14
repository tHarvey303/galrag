"""Operations shared by the CLI and the MCP server.

Nothing in here writes to stdout (the MCP stdio transport owns it); progress goes
through ``logging``.  Heavy dependencies (torch / sentence-transformers) are only
imported when an embedder is actually used.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from typing import Any

from galrag.config import get_settings
from galrag.db import Database, meta_from_ads, normalise_ident, now_iso
from galrag.models import PaperMeta, Status

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- logging
def setup_logging(level: int | str = logging.INFO, *, console: str = "rich",
                  console_level: int | str | None = None) -> None:
    """Configure the ``galrag`` logger: a console handler on **stderr** and a rotating
    file handler at ``<data_dir>/logs/galrag.log``.

    ``console`` is "rich" (pretty, CLI), "plain" (MCP server) or "none".
    """
    s = get_settings()
    root = logging.getLogger("galrag")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(process)d] %(message)s")
    try:
        fh = logging.handlers.RotatingFileHandler(s.logs_dir / "galrag.log", maxBytes=20_000_000,
                                                  backupCount=5, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:  # read-only data dir: console only
        pass
    clevel = console_level if console_level is not None else level
    if console == "rich":
        from rich.console import Console
        from rich.logging import RichHandler

        ch: logging.Handler = RichHandler(console=Console(stderr=True), show_path=False,
                                          rich_tracebacks=False, log_time_format="%H:%M:%S")
    elif console == "plain":
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(fmt)
    else:
        return
    ch.setLevel(clevel)
    root.addHandler(ch)


# ---------------------------------------------------------------- formatting
def surname(name: str | None) -> str:
    return (name or "").split(",")[0].strip()


def authors_short(first_author: str | None, n_authors: int | None = None) -> str:
    s = surname(first_author)
    if not s:
        return ""
    if n_authors is not None and n_authors <= 1:
        return s
    return f"{s} et al."


def citation_string(first_author: str | None, year: int | None, bibcode: str | None,
                    paper_key: str, n_authors: int | None = None) -> str:
    """"Bunker et al. (2023), 2023A&A...677A..88B"."""
    who = authors_short(first_author, n_authors) or "Anon."
    yr = f" ({year})" if year else ""
    return f"{who}{yr}, {bibcode or paper_key}"


def ads_url(bibcode: str | None) -> str | None:
    return f"https://ui.adsabs.harvard.edu/abs/{bibcode}/abstract" if bibcode else None


def arxiv_url(arxiv_id: str | None) -> str | None:
    return f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None


# ---------------------------------------------------------------- embedding helpers
def embed_paper_vectors(db: Database, store, embedder, keys: Sequence[str]) -> dict[str, int]:
    """Title+abstract vector in the paper table for ``keys``; papers that are not
    full-text embedded also get an abstract-only chunk (same logic as
    :func:`galrag.pipeline.run_embed_abstracts`, but for an explicit key list)."""
    from galrag.chunk import chunk_abstract_only
    from galrag.pipeline import _paper_info

    metas: list[PaperMeta] = []
    for k in dict.fromkeys(keys):
        m = db.get_meta(k)
        if m is not None:
            metas.append(m)
    if not metas:
        return {"papers": 0, "abstract_chunks": 0}
    texts = [f"{m.title}\n\n{m.abstract or ''}".strip() or m.title or m.paper_key for m in metas]
    vecs = embedder.embed_passages(texts)
    store.upsert_papers(metas, vecs)
    abs_idx, abs_chunks = [], []
    for i, m in enumerate(metas):
        row = db.get_row(m.paper_key)
        if row is not None and row["status"] != Status.EMBEDDED:
            cs = chunk_abstract_only(m)
            if cs:
                abs_idx.append(i)
                abs_chunks.append(cs[0])
    if abs_chunks:
        store.upsert_chunks(abs_chunks, vecs[abs_idx],
                            paper_info={metas[i].paper_key: _paper_info(metas[i]) for i in abs_idx})
    db.mark_paper_vec([m.paper_key for m in metas], abs_chunk_keys=[c.paper_key for c in abs_chunks])
    return {"papers": len(metas), "abstract_chunks": len(abs_chunks)}


def embed_fulltext(db: Database, store, embedder, keys: Iterable[str]) -> dict[str, int]:
    """Chunk+embed the given keys that are PARSED (others are ignored)."""
    from galrag.pipeline import run_embed

    parsed = [k for k in keys if (r := db.get_row(k)) is not None and r["status"] == Status.PARSED]
    if not parsed:
        return {"papers": 0, "chunks": 0, "failed": 0}
    return run_embed(db, store, embedder, paper_keys=parsed)


# ---------------------------------------------------------------- ingest
def ads_query_for(ident: str) -> str:
    """ADS query that finds exactly one identifier (bibcode, arXiv id or DOI)."""
    n = normalise_ident(ident)
    kind, val = n.split(":", 1)
    if kind == "doi":
        return f'doi:"{ident.strip().removeprefix("doi:").removeprefix("DOI:")}"'
    if kind == "arxiv":
        return f'identifier:"arXiv:{val}"'
    return f'identifier:"{ident.strip()}"'


def harvest_exact(db: Database, query: str, *, max_records: int = 5, client=None) -> list[str]:
    """Run an ADS query *without* the corpus year/doctype filters and upsert the
    results.  Returns the paper keys."""
    from galrag.ads import ADSClient

    own = client is None
    client = client or ADSClient()
    try:
        run = db.start_run("ingest", query)
        metas = []
        for d in client.search(query, max_records=max_records, rows=min(max_records, 2000),
                               sort="citation_count desc"):
            try:
                metas.append(meta_from_ads(d))
            except Exception as e:  # noqa: BLE001
                log.warning("skipping malformed ADS record %s: %s", d.get("bibcode"), e)
        n_new, n_upd = db.upsert_papers(metas) if metas else (0, 0)
        db.finish_run(run, n_new=n_new, n_updated=n_upd)
        return [m.paper_key for m in metas]
    finally:
        if own:
            client.close()


def ingest_ident(db: Database, store, embedder, ident: str, *, embed: bool = True,
                 force_fetch: bool = False, max_records: int = 20, client=None) -> list[dict[str, Any]]:
    """Ingest one identifier (or ``query:<ADS query>``): metadata -> full text -> embed.

    Returns one result dict per paper with the outcome of each stage."""
    from galrag.fetch import FetchError
    from galrag.parse import load_document
    from galrag.pipeline import fetch_and_parse_one

    ident = ident.strip()
    results: list[dict[str, Any]] = []
    if ident.lower().startswith("query:"):
        keys = harvest_exact(db, ident[6:].strip(), max_records=max_records, client=client)
        if not keys:
            return [{"ident": ident, "error": "ADS query matched nothing"}]
        meta_stage = "harvested"
    else:
        key = db.resolve(ident)
        meta_stage = "already in corpus"
        if key is None:
            try:
                keys = harvest_exact(db, ads_query_for(ident), max_records=1, client=client)
            except Exception as e:  # noqa: BLE001
                return [{"ident": ident, "error": f"ADS lookup failed: {e}"}]
            if not keys:
                return [{"ident": ident, "error": "not found in ADS"}]
            meta_stage = "harvested"
        else:
            keys = [key]

    for key in keys:
        res: dict[str, Any] = {"ident": ident, "paper_key": key, "metadata": meta_stage}
        row = db.get_row(key)
        status = row["status"] if row is not None else None
        res["status_before"] = status
        meta = db.get_meta(key)
        if meta is None:
            res["error"] = "metadata missing after harvest"
            results.append(res)
            continue
        res["bibcode"], res["title"] = meta.bibcode, meta.title
        have_doc = status in (Status.PARSED, Status.CHUNKED, Status.EMBEDDED) and \
            load_document(key) is not None
        if have_doc and not force_fetch:
            res["fulltext"] = f"already parsed ({row['source']})"
        else:
            try:
                res["fulltext"] = fetch_and_parse_one(db, meta, force=force_fetch)
            except FetchError as e:
                res["fulltext"] = f"unavailable: {str(e)[:200]}"
            except Exception as e:  # noqa: BLE001
                res["fulltext"] = f"parse failed: {e!r}"[:300]
        if embed:
            row = db.get_row(key)
            if row["status"] == Status.EMBEDDED and have_doc and not force_fetch:
                res["embed"] = f"already embedded ({row['n_chunks']} chunks)"
            else:
                if row["status"] == Status.EMBEDDED:  # re-parsed: re-embed
                    db.set_status(key, Status.PARSED)
                try:
                    c = embed_fulltext(db, store, embedder, [key])
                    pv = embed_paper_vectors(db, store, embedder, [key])
                    row = db.get_row(key)
                    if row["status"] == Status.EMBEDDED:
                        res["embed"] = f"full text: {row['n_chunks']} chunks"
                    else:
                        res["embed"] = f"abstract only ({pv['abstract_chunks']} chunk)"
                    res["embed_counts"] = {**c, **{f"paper_{k}": v for k, v in pv.items()}}
                except Exception as e:
                    log.exception("embedding failed for %s", key)
                    res["embed"] = f"failed: {e!r}"[:300]
        row = db.get_row(key)
        res["status"] = row["status"]
        res["source"] = row["source"]
        results.append(res)
    return results


def keys_harvested_since(db: Database, ts: str, *, tiers: Sequence[str] | None = ("A", "B"),
                         statuses: Sequence[str] | None = (Status.METADATA,),
                         limit: int | None = None) -> list[str]:
    q = "SELECT paper_key FROM papers WHERE harvested_at >= ?"
    vals: list[Any] = [ts]
    if tiers:
        q += f" AND tier IN ({','.join('?' * len(tiers))})"
        vals += list(tiers)
    if statuses:
        q += f" AND status IN ({','.join('?' * len(statuses))})"
        vals += [str(s) for s in statuses]
    q += " ORDER BY priority DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    return [r[0] for r in db.conn.execute(q, vals)]


def update_corpus(db: Database, store, embedder_factory, *, process: bool = True,
                  max_papers: int | None = None, client=None) -> dict[str, Any]:
    """Incremental ADS update, then fetch/parse/embed the new tier-A/B papers and give
    every new paper a title+abstract vector.  ``embedder_factory()`` is only called if
    something needs embedding."""
    from galrag.ads import ADSClient
    from galrag.pipeline import run_fetch_parse

    t0 = now_iso()
    own = client is None
    client = client or ADSClient()
    try:
        n_new, n_upd = client.update(db)
    finally:
        if own:
            client.close()
    out: dict[str, Any] = {"ads_new": n_new, "ads_updated": n_upd, "since": t0}
    if not process or n_new == 0:
        return out
    todo = keys_harvested_since(db, t0, limit=max_papers)
    out["fulltext_queue"] = len(todo)
    out["fetch"] = run_fetch_parse(db, paper_keys=todo) if todo else {}
    new_all = keys_harvested_since(db, t0, tiers=None, statuses=None)
    if new_all:
        emb = embedder_factory()
        out["embed"] = embed_fulltext(db, store, emb, todo)
        out["paper_vectors"] = {"papers": 0, "abstract_chunks": 0}
        for i in range(0, len(new_all), 1000):
            r = embed_paper_vectors(db, store, emb, new_all[i:i + 1000])
            for k, v in r.items():
                out["paper_vectors"][k] += v
        try:
            store.refresh_indexes()
        except Exception as e:  # noqa: BLE001
            log.warning("index refresh failed: %s", e)
    return out


# ---------------------------------------------------------------- stats
def _du(path) -> int | None:
    """Disk usage in bytes (``du -sb``; None on failure / timeout)."""
    if not os.path.exists(path):
        return 0
    du = shutil.which("du")
    if du:
        try:
            out = subprocess.run([du, "-sb", str(path)], capture_output=True, text=True, timeout=60,
                                 check=False)
            return int(out.stdout.split()[0])
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            return None
    return None


def human_bytes(n: int | None) -> str:
    if n is None:
        return "n/a"
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f} {unit}" if unit != "B" else f"{int(f)} B"
        f /= 1024
    return str(n)


def corpus_stats(db: Database, store=None, *, disk: bool = True) -> dict[str, Any]:
    """Counts per status/tier, vector-store size, last runs, failure reasons, disk usage.
    Never loads an embedding model."""
    s = get_settings()
    t0 = time.time()
    out: dict[str, Any] = {
        "data_dir": str(s.data_dir),
        "papers_total": int(db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]),
        "by_status": db.count_by_status(),
        "by_tier": db.count_by_tier(),
        "with_paper_vector": int(db.conn.execute(
            "SELECT COUNT(*) FROM papers WHERE paper_vec=1").fetchone()[0]),
    }
    full = db.conn.execute(
        "SELECT tier, COUNT(*) FROM papers WHERE status=? GROUP BY tier", (str(Status.EMBEDDED),)).fetchall()
    out["fulltext_embedded_by_tier"] = {r[0]: r[1] for r in full}
    try:
        st = store
        if st is None:
            from galrag.store import VectorStore

            st = VectorStore()
        out["vector_chunks"] = st.count()
        out["vector_papers"] = int(st.papers.count_rows()) if st.papers is not None else 0
    except Exception as e:  # noqa: BLE001
        out["vector_chunks"] = f"error: {e}"
    try:
        out["docs_on_disk"] = sum(1 for e in os.scandir(s.docs_dir) if e.name.endswith((".json", ".json.gz")))
    except OSError:
        out["docs_on_disk"] = 0
    runs = {}
    for kind in ("harvest", "update", "ingest"):
        r = db.last_run(kind)
        if r is not None:
            runs[kind] = {"finished_at": r["finished_at"], "n_new": r["n_new"], "n_updated": r["n_updated"],
                          "note": r["note"]}
    out["last_runs"] = runs
    out["max_entdate"] = db.max_entdate()
    from collections import Counter

    fails: Counter = Counter()
    for st_, err in db.conn.execute("SELECT status, error FROM papers WHERE status LIKE 'failed:%'"):
        fails[(st_, _clean_err(err))] += 1
    out["top_failures"] = [{"status": k[0], "error": k[1], "n": n} for k, n in fails.most_common(12)]
    if disk:
        out["disk"] = {name: human_bytes(_du(s.data_dir / name))
                       for name in ("raw", "docs", "lancedb", "models", "logs")}
        try:
            out["disk"]["galrag.sqlite"] = human_bytes(os.path.getsize(s.sqlite_path))
        except OSError:
            pass
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


def _clean_err(e: str | None) -> str:
    """Normalise an error string so identical causes group together."""
    if not e:
        return ""
    e = re.sub(r"^\S{19}:\s*", "", e)            # leading "<bibcode>: "
    e = re.sub(r"https?://\S+", "<url>", e)
    e = re.sub(r"\b\d{4}\.\d{4,5}(v\d+)?\b|\b[a-z-]+/\d{7}\b", "<arxiv>", e)
    e = re.sub(r"\d+(\.\d+)?", "N", e)
    return e[:120]
