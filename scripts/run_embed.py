"""Long-running, resumable chunk+embed loop.

Each pass: (1) give every paper a title+abstract vector (and papers without
full text an abstract-only chunk), (2) chunk+embed all PARSED papers, (3) fold
new rows into the LanceDB indexes.  With ``--loop`` it repeats every
``--interval`` seconds so it can run alongside the fetch loop.

Usage: python scripts/run_embed.py [--workers N] [--threads N] [--loop] [--interval S]
                                   [--sync-interval S] [--push lancedb]
Logs to $GALRAG_DATA_DIR/logs/embed.log.  Safe to stop with SIGTERM/SIGINT and restart.

On the GPU machine (GALRAG_WORK_DIR and GALRAG_LANCEDB_DIR on local disk) the local
LanceDB is rsynced into $GALRAG_DATA_DIR/lancedb every --sync-interval seconds and at
exit.  ``--push`` defaults to ``lancedb`` only: the SQLite there is a snapshot of the
archive and must not overwrite it.  Afterwards run ``galrag reconcile`` on the fetch node.
"""
import argparse
import logging
import time
from logging.handlers import RotatingFileHandler

from galrag.config import get_settings
from galrag.db import Database
from galrag.embed import PoolEmbedder
from galrag.pipeline import run_embed, run_embed_abstracts
from galrag.store import VectorStore
from galrag.sync import sync_push_every


def _push_targets(s: str) -> dict[str, bool]:
    parts = {p.strip() for p in s.split(",") if p.strip()}
    bad = parts - {"sqlite", "lancedb", "none"}
    if bad:
        raise SystemExit(f"--push: unknown target(s) {sorted(bad)}")
    return {"sqlite": "sqlite" in parts, "lancedb": "lancedb" in parts}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--no-abstracts", action="store_true")
    ap.add_argument("--sync-interval", type=float, default=1800, help="Seconds between pushes to data_dir.")
    ap.add_argument("--push", default="lancedb", help="What to push: lancedb | sqlite,lancedb | none.")
    a = ap.parse_args()
    s = get_settings()
    h = RotatingFileHandler(s.logs_dir / "embed.log", maxBytes=50_000_000, backupCount=3)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=[h])
    for name in ("httpx", "sentence_transformers", "lance", "lancedb"):
        logging.getLogger(name).setLevel(logging.WARNING)
    log = logging.getLogger("run_embed")

    db = Database()
    store = VectorStore()
    log.info("sqlite=%s lancedb=%s archive=%s", s.sqlite_path, s.lancedb_dir, s.data_dir)
    push = sync_push_every(s, a.sync_interval, **_push_targets(a.push))
    try:
        _loop(a, db, store, push, log)
    finally:
        log.info("final push: %s", push(force=True))


def _loop(a, db, store, push, log) -> None:
    from galrag.pipeline import _Stop

    stop = _Stop()
    with PoolEmbedder(n_workers=a.workers, threads_per_worker=a.threads) as emb:
        while True:
            t0 = time.time()
            try:
                if not a.no_abstracts:
                    log.info("abstracts: %s", run_embed_abstracts(db, store, emb, limit=a.limit,
                                                                  on_progress=push))
                res = run_embed(db, store, emb, limit=a.limit, on_progress=push)
                log.info("fulltext: %s", res)
                if res["papers"]:
                    log.info("indexes: %s", store.create_indexes())
                    store.refresh_indexes()
                log.info("pass done in %.0fs; store has %d chunks", time.time() - t0, store.count())
            except Exception:
                log.exception("pass failed; retrying after %ds", a.interval)
            push()
            if not a.loop or stop.flag:
                break
            for _ in range(int(a.interval)):     # sleep, but wake up for SIGTERM/SIGINT
                if stop.flag:
                    break
                time.sleep(1)
            if stop.flag:
                break


if __name__ == "__main__":  # required: workers are spawned and re-import this module
    main()
