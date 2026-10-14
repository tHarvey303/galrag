"""Long-running, resumable chunk+embed loop.

Each pass: (1) give every paper a title+abstract vector (and papers without
full text an abstract-only chunk), (2) chunk+embed all PARSED papers, (3) fold
new rows into the LanceDB indexes.  With ``--loop`` it repeats every
``--interval`` seconds so it can run alongside the fetch loop.

Usage: python scripts/run_embed.py [--workers N] [--threads N] [--loop] [--interval S]
Logs to $GALRAG_DATA_DIR/logs/embed.log.  Safe to stop with SIGTERM/SIGINT and restart.
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

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=None)
ap.add_argument("--threads", type=int, default=None)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--loop", action="store_true")
ap.add_argument("--interval", type=int, default=600)
ap.add_argument("--no-abstracts", action="store_true")
a = ap.parse_args()
s = get_settings()
h = RotatingFileHandler(s.logs_dir / "embed.log", maxBytes=50_000_000, backupCount=3)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=[h])
for name in ("httpx", "sentence_transformers", "lance", "lancedb"):
    logging.getLogger(name).setLevel(logging.WARNING)
log = logging.getLogger("run_embed")

db = Database()
store = VectorStore()
with PoolEmbedder(n_workers=a.workers, threads_per_worker=a.threads) as emb:
    while True:
        t0 = time.time()
        if not a.no_abstracts:
            log.info("abstracts: %s", run_embed_abstracts(db, store, emb, limit=a.limit))
        res = run_embed(db, store, emb, limit=a.limit)
        log.info("fulltext: %s", res)
        if res["papers"]:
            log.info("indexes: %s", store.create_indexes())
            store.refresh_indexes()
        log.info("pass done in %.0fs; store has %d chunks", time.time() - t0, store.count())
        if not a.loop:
            break
        time.sleep(a.interval)
