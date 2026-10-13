"""Pipeline orchestration: harvest -> fetch -> parse -> chunk -> embed.

Every stage is idempotent and driven by the ``status`` column in SQLite, so
it can be interrupted and resumed.  Network stages are serial (rate-limited
per host); CPU stages are batched.
"""

from __future__ import annotations

import logging
import signal
import time
from typing import Iterable

from galrag.config import get_settings
from galrag.db import Database
from galrag.fetch import FetchError, fetch_fulltext
from galrag.models import PaperMeta, Status
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
    except Exception as e:  # noqa: BLE001
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
        rows = [db.get_row(k) for k in paper_keys]
        rows = [r for r in rows if r is not None]
    else:
        statuses = [Status.METADATA] + ([Status.FAILED_FETCH, Status.FAILED_PARSE] if retry_failed else [])
        rows = []
        for st in statuses:
            rows.extend(db.iter_by_status(st, tiers=tiers, limit=limit))
        rows.sort(key=lambda r: -r["priority"])
        if limit:
            rows = rows[:limit]
    log.info("fetch/parse queue: %d papers (tiers=%s)", len(rows), list(tiers))
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        if stop.flag:
            break
        meta = PaperMeta.model_validate_json(row["meta_json"])
        if row["status"] in (Status.PARSED, Status.CHUNKED, Status.EMBEDDED) and not paper_keys:
            continue
        try:
            fetch_and_parse_one(db, meta)
            counts["ok"] += 1
        except FetchError as e:
            counts["fetch_failed"] += 1
            log.warning("fetch failed %s: %s", meta.paper_key, str(e)[:200])
        except Exception as e:  # noqa: BLE001
            counts["parse_failed"] += 1
            log.exception("parse failed %s: %r", meta.paper_key, e)
        if i % 50 == 0:
            rate = i / (time.time() - t0)
            log.info("progress %d/%d (%.2f papers/s) %s", i, len(rows), rate, counts)
    return counts


def iter_parsed_without_embedding(db: Database, limit: int | None = None):
    """Yield (meta, Document) for papers that are parsed but not yet in the vector store."""
    for row in db.iter_by_status(Status.PARSED, limit=limit):
        doc = load_document(row["paper_key"])
        if doc is None:
            db.set_status(row["paper_key"], Status.FAILED_PARSE, error="document file missing")
            continue
        yield PaperMeta.model_validate_json(row["meta_json"]), doc
