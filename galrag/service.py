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
    ident = ident.strip()
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

    return process_paper_keys(db, store, embedder, keys, embed=embed, force_fetch=force_fetch,
                              base={"ident": ident, "metadata": meta_stage})


def process_paper_keys(db: Database, store, embedder, keys: Iterable[str], *, embed: bool = True,
                       force_fetch: bool = False, base: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Fetch/parse and (optionally) embed papers that are already in SQLite.  Returns one
    result dict per paper (``base`` is copied into each) with the outcome of each stage."""
    from galrag.fetch import FetchError
    from galrag.parse import load_document
    from galrag.pipeline import fetch_and_parse_one

    results: list[dict[str, Any]] = []
    for key in keys:
        res: dict[str, Any] = {**(base or {}), "paper_key": key}
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
            load_document(key, db) is not None
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
    # papers taken from the arXiv feed that ADS has now indexed: re-keyed to the bibcode,
    # their old store rows go, and they are re-embedded below under the new key
    rekeyed = [new for _, new in purge_rekeyed(db, store)]
    out["rekeyed"] = len(rekeyed)
    if not process or (n_new == 0 and not rekeyed):
        return out
    todo = keys_harvested_since(db, t0, limit=max_papers)
    out["fulltext_queue"] = len(todo)
    out["fetch"] = run_fetch_parse(db, paper_keys=todo) if todo else {}
    new_all = keys_harvested_since(db, t0, tiers=None, statuses=None) + rekeyed
    if new_all:
        emb = embedder_factory()
        out["embed"] = embed_fulltext(db, store, emb, todo + rekeyed)
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


# ---------------------------------------------------------------- re-keyed papers
def purge_rekeyed(db: Database, store) -> list[tuple[str, str]]:
    """Delete vector-store rows of papers that were re-keyed from ``arXiv:<id>`` to a
    bibcode (see :meth:`galrag.db.Database.upsert_papers`).  Returns the (old, new) key
    pairs purged; the new keys still need embedding (status ``parsed``, no paper vector)."""
    done: list[tuple[str, str]] = []
    for old, new in db.pending_rekeys():
        try:
            store.delete_paper(old)
            done.append((old, new))
        except Exception:
            log.exception("could not purge store rows of re-keyed paper %s (-> %s)", old, new)
    if done:
        db.mark_rekeys_cleaned([o for o, _ in done])
    return done


# ---------------------------------------------------------------- arXiv RSS feed
def ingest_arxiv_feed(db: Database, store, embedder_factory, *, category: str | None = None,
                      feed_path: str | None = None, dry_run: bool = False, embed: bool = True,
                      max_papers: int | None = None, min_score: float | None = None,
                      include_replacements: bool = False, use_ads: bool = True,
                      client=None, since: str | None = None, until: str | None = None) -> dict[str, Any]:
    """Screen one day's arXiv listing (or, with ``since``, every submission in a date range
    from the arXiv API) and ingest the relevant papers.

    Every entry is screened with the topic keyword rule
    (:func:`galrag.arxiv_feed.topic_keywords`, the local mirror of the ADS harvest query)
    and, when ``min_score`` > 0 or ``dry_run``, scored by the cross-encoder reranker
    (``store.reranker``) against ``settings.feed_topic`` (0-1; the bi-encoder cosine to the
    corpus does not separate topics, the cross-encoder does).  An entry is accepted when
    it matches a keyword or its topic score reaches ``min_score``.
    Accepted papers get ADS metadata when ADS already has them, otherwise metadata from
    the feed (keyed ``arXiv:<id>``, re-keyed by a later ADS update), then full text and
    embeddings.  Replacements of papers already in the corpus are skipped unless
    ``include_replacements`` (then they are re-fetched and re-embedded).  With
    ``dry_run`` nothing is written except the ``arxiv_feed`` log."""
    from galrag.arxiv_feed import fetch_feed, fetch_listing, meta_from_feed, topic_keywords

    s = get_settings()
    if min_score is None:
        min_score = s.feed_min_score
    if since:
        info, entries = fetch_listing(category, since=since, until=until)
    else:
        info, entries = fetch_feed(category, path=feed_path)
    run = db.start_run("rss", info.get("source"))
    out: dict[str, Any] = {"feed": info, "n_entries": len(entries), "min_score": min_score,
                           "dry_run": dry_run, "by_type": {}, "accepted": [], "skipped": [],
                           "rejected": 0, "ingested": [], "purged_rekeyed": 0}
    for e in entries:
        out["by_type"][e.announce_type] = out["by_type"].get(e.announce_type, 0) + 1

    # -- which entries are candidates
    cand: list[dict[str, Any]] = []
    for e in entries:
        key = db.resolve(f"arXiv:{e.arxiv_id}")
        row = db.get_row(key) if key else None
        have_text = row is not None and row["status"] in (Status.PARSED, Status.CHUNKED, Status.EMBEDDED)
        item = {"arxiv_id": e.arxiv_id, "title": e.title, "announce_type": e.announce_type,
                "categories": e.categories, "paper_key": key, "status_before": row["status"] if row else None}
        if e.is_replacement and not include_replacements:
            item["note"] = "replacement"
            out["skipped"].append(item)
        elif have_text and not (e.is_replacement and include_replacements):
            item["note"] = "already in corpus"
            out["skipped"].append(item)
        else:
            cand.append({**item, "entry": e, "keywords": topic_keywords(e.title, e.abstract)})

    # -- topic score from the cross-encoder (one batch; ~1 s per paper on a busy CPU)
    if cand and (min_score > 0 or dry_run):
        try:
            scores = store.reranker.rerank(s.feed_topic, [c["entry"].text for c in cand])
            for c, sc in zip(cand, scores):
                c["topic_score"] = round(float(sc), 4)
        except Exception as exc:
            log.exception("feed topic scoring failed; using the keyword rule only")
            out["score_error"] = str(exc)[:200]
    for c in cand:
        sc = c.get("topic_score")
        c["accept"] = bool(c["keywords"]) or (min_score > 0 and sc is not None and sc >= min_score)
    cand.sort(key=lambda c: (-int(c["accept"]), -(c.get("topic_score") or 0.0)))
    accepted = [c for c in cand if c["accept"]]
    if max_papers is not None and len(accepted) > max_papers:
        for c in accepted[max_papers:]:
            c["accept"] = False
            c["note"] = "over max_papers"
        accepted = accepted[:max_papers]
    out["rejected"] = sum(1 for c in cand if not c["accept"])
    out["accepted"] = [_feed_public(c) for c in accepted]
    if dry_run:
        out["candidates"] = [_feed_public(c) for c in cand]

    # -- metadata + full text + embeddings
    results = []
    if accepted and not dry_run:
        from galrag.ads import ADSClient

        own = client is None
        if use_ads and client is None and s.ads_token:
            try:
                client = ADSClient()
            except Exception as exc:  # noqa: BLE001
                log.warning("ADS unavailable for the feed run (%s); using feed metadata", exc)
                client = None
        try:
            for c in accepted:
                e = c["entry"]
                key = c["paper_key"]
                stage = "already in corpus"
                if key is None and client is not None:
                    try:
                        found = harvest_exact(db, ads_query_for(f"arXiv:{e.arxiv_id}"), max_records=1,
                                              client=client)
                        if found:
                            key, stage = found[0], "ADS"
                    except Exception as exc:  # noqa: BLE001
                        log.warning("ADS lookup failed for %s: %s", e.arxiv_id, exc)
                if key is None:
                    m = meta_from_feed(e)
                    db.upsert_papers([m])
                    key, stage = m.paper_key, "feed"
                c["paper_key"], c["metadata"] = key, stage
            emb = embedder_factory() if embed else None
            for c in accepted:
                e = c["entry"]
                force = bool(e.is_replacement and include_replacements)
                r = process_paper_keys(db, store, emb, [c["paper_key"]], embed=embed, force_fetch=force,
                                       base={"arxiv_id": e.arxiv_id, "metadata": c["metadata"]})
                results.extend(r)
            if embed:
                out["purged_rekeyed"] = len(purge_rekeyed(db, store))
                try:
                    store.refresh_indexes()
                except Exception as exc:  # noqa: BLE001
                    log.warning("index refresh failed: %s", exc)
        finally:
            if own and client is not None:
                client.close()
    out["ingested"] = results

    # -- bookkeeping
    ts_note = "dry-run" if dry_run else None
    log_rows = []
    for c in cand:
        log_rows.append({"arxiv_id": c["arxiv_id"], "announce_type": c["announce_type"], "title": c["title"],
                         "categories": ",".join(c["categories"]), "keywords": ",".join(c["keywords"]),
                         "topic_score": c.get("topic_score"), "paper_key": c.get("paper_key"),
                         "decision": ts_note or ("accepted" if c["accept"] else "rejected"),
                         "note": c.get("note")})
    for c in out["skipped"]:
        log_rows.append({"arxiv_id": c["arxiv_id"], "announce_type": c["announce_type"], "title": c["title"],
                         "categories": ",".join(c["categories"]), "paper_key": c.get("paper_key"),
                         "decision": "skipped", "note": c.get("note")})
    db.log_feed_items(log_rows)
    db.finish_run(run, n_new=len(results), n_updated=len(accepted),
                  note=f"{info.get('updated')}: {len(entries)} items, {len(accepted)} accepted"
                       + (" (dry run)" if dry_run else ""))
    return out


def _feed_public(c: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in c.items() if k != "entry"}


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
        "work_dir": str(s.work_dir) if s.work_dir else None,
        "sqlite_path": str(s.sqlite_path),
        "lancedb_dir": str(s.lancedb_dir),
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
    out["documents"] = db.count_documents()
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
        out["disk"] = {"raw": human_bytes(_du(s.raw_dir)), "lancedb": human_bytes(_du(s.lancedb_dir)),
                       **{name: human_bytes(_du(s.data_dir / name)) for name in ("models", "logs")}}
        for label, path in (("galrag.sqlite", s.sqlite_path), ("galrag.sqlite (archive)", s.archive_sqlite_path)):
            try:
                out["disk"][label] = human_bytes(os.path.getsize(path))
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
