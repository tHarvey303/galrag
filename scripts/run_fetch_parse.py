"""Long-running, resumable full-text fetch+parse loop (tiers A then B by priority).

Usage: python scripts/run_fetch_parse.py [--tiers A,B] [--limit N] [--retry-failed]
                                         [--sync-interval S] [--push sqlite,lancedb]
Logs to $GALRAG_DATA_DIR/logs/fetch_parse.log.  Safe to stop with SIGTERM/SIGINT and restart.
With GALRAG_WORK_DIR set, the SQLite it writes lives on local disk and a consistent
snapshot is pushed to $GALRAG_DATA_DIR every --sync-interval seconds and at exit.
"""
import argparse
import logging
from logging.handlers import RotatingFileHandler

from galrag.config import get_settings
from galrag.db import Database
from galrag.pipeline import run_fetch_parse
from galrag.sync import sync_push_every


def _push_targets(s: str) -> dict[str, bool]:
    parts = {p.strip() for p in s.split(",") if p.strip()}
    bad = parts - {"sqlite", "lancedb", "none"}
    if bad:
        raise SystemExit(f"--push: unknown target(s) {sorted(bad)}")
    return {"sqlite": "sqlite" in parts, "lancedb": "lancedb" in parts}


ap = argparse.ArgumentParser()
ap.add_argument("--tiers", default="A,B")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--retry-failed", action="store_true")
ap.add_argument("--sync-interval", type=float, default=1800, help="Seconds between pushes to data_dir.")
ap.add_argument("--push", default="sqlite,lancedb", help="What to push: sqlite,lancedb | none.")
a = ap.parse_args()
s = get_settings()
h = RotatingFileHandler(s.logs_dir / "fetch_parse.log", maxBytes=50_000_000, backupCount=3)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=[h])
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("run_fetch_parse")
log.info("sqlite=%s raw=%s archive=%s", s.sqlite_path, s.raw_dir, s.archive_sqlite_path)
push = sync_push_every(s, a.sync_interval, **_push_targets(a.push))
db = Database()
try:
    res = run_fetch_parse(db, tiers=a.tiers.split(","), limit=a.limit, retry_failed=a.retry_failed,
                          on_progress=push)
    log.info("fetch/parse finished: %s", res)
    print(res)
finally:
    log.info("final push: %s", push(force=True))
