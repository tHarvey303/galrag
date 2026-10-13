"""Long-running, resumable full-text fetch+parse loop (tiers A then B by priority).

Usage: python scripts/run_fetch_parse.py [--tiers A,B] [--limit N] [--retry-failed]
Logs to $GALRAG_DATA_DIR/logs/fetch_parse.log.  Safe to stop with SIGTERM/SIGINT and restart.
"""
import argparse
import logging
from logging.handlers import RotatingFileHandler

from galrag.config import get_settings
from galrag.db import Database
from galrag.pipeline import run_fetch_parse

ap = argparse.ArgumentParser()
ap.add_argument("--tiers", default="A,B")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--retry-failed", action="store_true")
a = ap.parse_args()
s = get_settings()
h = RotatingFileHandler(s.logs_dir / "fetch_parse.log", maxBytes=50_000_000, backupCount=3)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=[h])
logging.getLogger("httpx").setLevel(logging.WARNING)
db = Database()
print(run_fetch_parse(db, tiers=a.tiers.split(","), limit=a.limit, retry_failed=a.retry_failed))
