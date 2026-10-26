"""SQLite metadata store and ingestion-state tracker.

One ``Database`` per process; it is cheap to construct.  WAL mode is used so a
long-running fetcher and the MCP server can share the file.

Tables
------
papers        one row per paper (metadata + status + tier/priority)
identifiers   many-to-one lookup table: any id (bibcode, arXiv id, DOI, alternate bibcode) -> paper_key
refs          (citing_key, cited_bibcode) edges from the ADS ``reference`` field
bibtex        cached BibTeX text per bibcode
runs          harvest/update bookkeeping (last entdate seen, counts)
documents     parsed full-text Document per paper (gzipped JSON); replaces the old
              ``docs/<key>.json`` files (thousands of small files are bad on Nexus)
"""

from __future__ import annotations

import gzip
import logging
import re
import sqlite3
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

from galrag.config import get_settings
from galrag.models import Document, PaperMeta, Status, Tier

log = logging.getLogger(__name__)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=-262144;
PRAGMA temp_store=MEMORY;
CREATE TABLE IF NOT EXISTS papers (
    paper_key      TEXT PRIMARY KEY,
    bibcode        TEXT UNIQUE,
    arxiv_id       TEXT,
    doi            TEXT,
    title          TEXT NOT NULL DEFAULT '',
    first_author   TEXT NOT NULL DEFAULT '',
    year           INTEGER,
    pubdate        TEXT,
    entdate        TEXT,
    refereed       INTEGER NOT NULL DEFAULT 0,
    doctype        TEXT,
    bibstem        TEXT,
    citation_count INTEGER NOT NULL DEFAULT 0,
    tier           TEXT NOT NULL DEFAULT 'C',
    priority       REAL NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'metadata',
    source         TEXT,                 -- full-text source actually used
    n_chunks       INTEGER NOT NULL DEFAULT 0,
    error          TEXT,
    meta_json      TEXT NOT NULL,        -- PaperMeta.model_dump_json()
    harvested_at   TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status);
CREATE INDEX IF NOT EXISTS idx_papers_tier_priority ON papers(tier, priority DESC);
CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
-- covering index: queue queries read only index pages (NFS round trips are expensive)
CREATE INDEX IF NOT EXISTS idx_papers_queue ON papers(status, tier, priority DESC, paper_key);
CREATE TABLE IF NOT EXISTS identifiers (
    ident      TEXT PRIMARY KEY,         -- normalised (lower-case, no arXiv version)
    paper_key  TEXT NOT NULL REFERENCES papers(paper_key) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS refs (
    citing_key   TEXT NOT NULL,
    cited_bibcode TEXT NOT NULL,
    PRIMARY KEY (citing_key, cited_bibcode)
);
CREATE INDEX IF NOT EXISTS idx_refs_cited ON refs(cited_bibcode);
CREATE TABLE IF NOT EXISTS bibtex (
    bibcode    TEXT PRIMARY KEY,
    bibtex     TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,            -- harvest | update | fetch | embed ...
    started_at TEXT NOT NULL,
    finished_at TEXT,
    query      TEXT,
    n_new      INTEGER DEFAULT 0,
    n_updated  INTEGER DEFAULT 0,
    note       TEXT
);
CREATE TABLE IF NOT EXISTS documents (
    paper_key      TEXT PRIMARY KEY REFERENCES papers(paper_key) ON DELETE CASCADE,
    parser_version TEXT,
    source         TEXT,
    json_gz        BLOB NOT NULL,        -- gzip(Document.model_dump_json())
    saved_at       TEXT NOT NULL
);
-- a paper first stored under arXiv:<id> (from the RSS feed) and later re-keyed to its ADS
-- bibcode; the vector store still holds rows under old_key until store_cleaned=1
CREATE TABLE IF NOT EXISTS rekeyed (
    old_key       TEXT PRIMARY KEY,
    new_key       TEXT NOT NULL,
    at            TEXT NOT NULL,
    store_cleaned INTEGER NOT NULL DEFAULT 0
);
-- every arXiv RSS item seen by `galrag rss`, with the relevance decision taken
CREATE TABLE IF NOT EXISTS arxiv_feed (
    arxiv_id      TEXT PRIMARY KEY,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    announce_type TEXT,
    title         TEXT,
    categories    TEXT,
    keywords      TEXT,                  -- matched topic terms, comma-separated
    topic_score   REAL,                  -- reranker score of title+abstract vs settings.feed_topic (0-1)
    decision      TEXT NOT NULL,         -- accepted | rejected | skipped | dry-run
    paper_key     TEXT,
    note          TEXT
);
"""

_ARXIV_NEW = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?$")
_ARXIV_OLD = re.compile(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?$")


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalise_arxiv_id(s: str) -> str | None:
    """'arXiv:2405.03781v2' -> '2405.03781'; 'astro-ph/0601001v1' -> 'astro-ph/0601001'."""
    if not s:
        return None
    s = s.strip()
    s = re.sub(r"^(arxiv:|https?://arxiv\.org/(abs|pdf|html)/)", "", s, flags=re.IGNORECASE)
    s = s.removesuffix(".pdf")
    m = _ARXIV_NEW.search(s) or _ARXIV_OLD.search(s)
    return m.group(1) if m else None


def normalise_ident(s: str) -> str:
    """Canonical form for the identifiers table."""
    s = s.strip()
    if s.lower().startswith("doi:"):
        s = s[4:]
    if s.lower().startswith("10."):
        return "doi:" + s.lower()
    a = normalise_arxiv_id(s)
    if a and (s.lower().startswith("arxiv") or _ARXIV_NEW.fullmatch(s.replace("arXiv:", "")) or "/" in s):
        return "arxiv:" + a.lower()
    return "bib:" + s


def paper_key_for(bibcode: str | None, arxiv_id: str | None) -> str:
    if bibcode:
        return bibcode
    if arxiv_id:
        return f"arXiv:{arxiv_id}"
    raise ValueError("need a bibcode or an arXiv id")


class Database:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else get_settings().sqlite_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=600, isolation_level=None)  # NFS locks can stall
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.migrate()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN")
        try:
            yield self.conn
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------ papers
    def upsert_papers(self, metas: Iterable[PaperMeta]) -> tuple[int, int]:
        """Insert or refresh metadata.  Never regresses ``status`` past 'metadata', except
        that an ADS record arriving for a paper stored under ``arXiv:<id>`` (from the RSS
        feed) re-keys that row to the bibcode (see :meth:`_rekey`) and counts as updated.

        Returns (n_new, n_updated)."""
        n_new = n_upd = 0
        ts = now_iso()
        with self.tx() as c:
            for m in metas:
                row = c.execute("SELECT status FROM papers WHERE paper_key=?", (m.paper_key,)).fetchone()
                if row is None and m.bibcode and m.arxiv_id:
                    old_key = f"arXiv:{m.arxiv_id}"
                    if c.execute("SELECT 1 FROM papers WHERE paper_key=?", (old_key,)).fetchone():
                        self._rekey(c, old_key, m.paper_key, ts)
                        row = c.execute("SELECT status FROM papers WHERE paper_key=?",
                                        (m.paper_key,)).fetchone()
                if row is None:
                    c.execute(
                        """INSERT INTO papers(paper_key,bibcode,arxiv_id,doi,title,first_author,year,pubdate,
                           entdate,refereed,doctype,bibstem,citation_count,tier,priority,status,meta_json,
                           harvested_at,updated_at)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (m.paper_key, m.bibcode, m.arxiv_id, m.doi, m.title, m.first_author, m.year,
                         m.pubdate, m.entdate, int(m.refereed), m.doctype, m.bibstem, m.citation_count,
                         m.tier.value, m.priority, Status.METADATA.value, m.model_dump_json(), ts, ts),
                    )
                    n_new += 1
                else:
                    c.execute(
                        """UPDATE papers SET bibcode=?,arxiv_id=?,doi=?,title=?,first_author=?,year=?,pubdate=?,
                           entdate=?,refereed=?,doctype=?,bibstem=?,citation_count=?,tier=?,priority=?,
                           meta_json=?,updated_at=? WHERE paper_key=?""",
                        (m.bibcode, m.arxiv_id, m.doi, m.title, m.first_author, m.year, m.pubdate, m.entdate,
                         int(m.refereed), m.doctype, m.bibstem, m.citation_count, m.tier.value, m.priority,
                         m.model_dump_json(), ts, m.paper_key),
                    )
                    n_upd += 1
                idents = {m.paper_key}
                if m.bibcode:
                    idents.add(m.bibcode)
                if m.arxiv_id:
                    idents.add("arXiv:" + m.arxiv_id)
                if m.doi:
                    idents.add("doi:" + m.doi)
                for alt in m.raw.get("alternate_bibcode", []) or []:
                    idents.add(alt)
                for ident in m.raw.get("identifier", []) or []:
                    idents.add(ident)
                for ident in idents:
                    c.execute("INSERT OR REPLACE INTO identifiers(ident,paper_key) VALUES(?,?)",
                              (normalise_ident(ident), m.paper_key))
                if m.references:
                    c.execute("DELETE FROM refs WHERE citing_key=?", (m.paper_key,))
                    c.executemany("INSERT OR IGNORE INTO refs(citing_key,cited_bibcode) VALUES(?,?)",
                                  [(m.paper_key, r) for r in m.references])
        return n_new, n_upd

    def _rekey(self, c: sqlite3.Connection, old_key: str, new_key: str, ts: str) -> None:
        """Move a paper stored under ``arXiv:<id>`` (feed metadata) to its ADS bibcode key.

        The row, identifiers, references and parsed document follow the paper.  A paper
        that was already embedded goes back to ``parsed`` with ``paper_vec``/``abs_chunk``
        cleared: the vector store holds its rows under the old key, so the next embed pass
        re-embeds it under the new one, and :func:`galrag.service.purge_rekeyed` deletes
        the old rows (bookkeeping in the ``rekeyed`` table)."""
        c.execute("UPDATE papers SET paper_key=?, updated_at=? WHERE paper_key=?", (new_key, ts, old_key))
        c.execute("UPDATE papers SET status=?, n_chunks=0 WHERE paper_key=? AND status=?",
                  (Status.PARSED.value, new_key, Status.EMBEDDED.value))
        c.execute("UPDATE papers SET paper_vec=0, abs_chunk=0 WHERE paper_key=?", (new_key,))
        c.execute("UPDATE identifiers SET paper_key=? WHERE paper_key=?", (new_key, old_key))
        c.execute("UPDATE refs SET citing_key=? WHERE citing_key=?", (new_key, old_key))
        c.execute("UPDATE arxiv_feed SET paper_key=? WHERE paper_key=?", (new_key, old_key))
        row = c.execute("SELECT json_gz FROM documents WHERE paper_key=?", (old_key,)).fetchone()
        if row is not None:
            doc = Document.model_validate_json(gzip.decompress(row[0]))
            doc.paper_key = new_key
            c.execute("DELETE FROM documents WHERE paper_key=?", (old_key,))
            c.execute(self._DOC_UPSERT, self._doc_row(doc, ts))
        c.execute("INSERT OR REPLACE INTO rekeyed(old_key,new_key,at,store_cleaned) VALUES(?,?,?,0)",
                  (old_key, new_key, ts))
        log.info("re-keyed %s -> %s", old_key, new_key)

    def pending_rekeys(self) -> list[tuple[str, str]]:
        """(old_key, new_key) pairs whose old rows may still be in the vector store."""
        return [(r[0], r[1]) for r in self.conn.execute(
            "SELECT old_key, new_key FROM rekeyed WHERE store_cleaned=0")]

    def mark_rekeys_cleaned(self, old_keys: Iterable[str]) -> None:
        with self.tx() as c:
            c.executemany("UPDATE rekeyed SET store_cleaned=1 WHERE old_key=?", [(k,) for k in old_keys])

    # ------------------------------------------------------------ arXiv feed log
    def log_feed_items(self, rows: Iterable[dict]) -> None:
        """Upsert ``arxiv_feed`` rows (dicts with the table's columns; ``first_seen`` is
        kept from the first sighting)."""
        ts = now_iso()
        with self.tx() as c:
            for r in rows:
                c.execute(
                    """INSERT INTO arxiv_feed(arxiv_id,first_seen,last_seen,announce_type,title,categories,
                       keywords,topic_score,decision,paper_key,note) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(arxiv_id) DO UPDATE SET last_seen=excluded.last_seen,
                       announce_type=excluded.announce_type, title=excluded.title,
                       categories=excluded.categories, keywords=excluded.keywords,
                       topic_score=excluded.topic_score, decision=excluded.decision,
                       paper_key=COALESCE(excluded.paper_key, arxiv_feed.paper_key), note=excluded.note""",
                    (r["arxiv_id"], ts, ts, r.get("announce_type"), r.get("title"), r.get("categories"),
                     r.get("keywords"), r.get("topic_score"), r["decision"], r.get("paper_key"), r.get("note")))

    def feed_history(self, *, decision: str | None = None, limit: int = 100) -> list[sqlite3.Row]:
        q, vals = "SELECT * FROM arxiv_feed", []
        if decision:
            q += " WHERE decision=?"; vals.append(decision)
        q += f" ORDER BY last_seen DESC, topic_score DESC LIMIT {int(limit)}"
        return self.conn.execute(q, vals).fetchall()

    def resolve(self, ident: str) -> str | None:
        """Map any identifier to a paper_key (None if unknown)."""
        if not ident:
            return None
        row = self.conn.execute("SELECT paper_key FROM papers WHERE paper_key=?", (ident,)).fetchone()
        if row:
            return row[0]
        for cand in {normalise_ident(ident), "bib:" + ident, "arxiv:" + (normalise_arxiv_id(ident) or "").lower(),
                     "doi:" + ident.lower()}:
            row = self.conn.execute("SELECT paper_key FROM identifiers WHERE ident=?", (cand,)).fetchone()
            if row:
                return row[0]
        return None

    def get_meta(self, paper_key: str) -> PaperMeta | None:
        row = self.conn.execute("SELECT meta_json FROM papers WHERE paper_key=?", (paper_key,)).fetchone()
        return PaperMeta.model_validate_json(row[0]) if row else None

    def get_row(self, paper_key: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM papers WHERE paper_key=?", (paper_key,)).fetchone()

    # -- schema migrations (columns added after the first harvest) -------------
    MIGRATIONS: ClassVar[dict[str, str]] = {
        "paper_vec": "INTEGER NOT NULL DEFAULT 0",   # title+abstract vector in papers table
        "abs_chunk": "INTEGER NOT NULL DEFAULT 0",   # abstract-only chunk in chunks table
    }

    def migrate(self) -> None:
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(papers)")}
        for col, decl in self.MIGRATIONS.items():
            if col not in cols:
                self.conn.execute(f"ALTER TABLE papers ADD COLUMN {col} {decl}")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_paper_vec ON papers(paper_vec, priority DESC)")

    # --------------------------------------------------------------- documents
    _DOC_UPSERT = ("INSERT OR REPLACE INTO documents(paper_key,parser_version,source,json_gz,saved_at) "
                   "VALUES(?,?,?,?,?)")

    @staticmethod
    def _doc_row(doc: Document, ts: str) -> tuple:
        blob = gzip.compress(doc.model_dump_json().encode("utf-8"), 6)
        return (doc.paper_key, doc.parser_version, doc.source, blob, ts)

    def put_document(self, doc: Document) -> None:
        """Store (or replace) a parsed document: one statement, retried on lock stalls."""
        self._execute_retry(self._DOC_UPSERT, self._doc_row(doc, now_iso()))

    def get_document(self, paper_key: str) -> Document | None:
        row = self.conn.execute("SELECT json_gz FROM documents WHERE paper_key=?", (paper_key,)).fetchone()
        if row is None:
            return None
        return Document.model_validate_json(gzip.decompress(row[0]))

    def has_document(self, paper_key: str) -> bool:
        return self.conn.execute("SELECT 1 FROM documents WHERE paper_key=?",
                                 (paper_key,)).fetchone() is not None

    def iter_document_keys(self) -> Iterator[str]:
        yield from (r[0] for r in self.conn.execute("SELECT paper_key FROM documents").fetchall())

    def count_documents(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def delete_document(self, paper_key: str) -> None:
        self._execute_retry("DELETE FROM documents WHERE paper_key=?", (paper_key,))

    def import_document_files(self, docs_dir: Path, *, batch: int = 500) -> int:
        """Load legacy ``docs/*.json`` / ``*.json.gz`` files into the documents table
        (transactions of ``batch``; keys already present are skipped).  Returns #imported."""
        docs_dir = Path(docs_dir)
        if not docs_dir.is_dir():
            return 0
        have = set(self.iter_document_keys())
        known = {r[0] for r in self.conn.execute("SELECT paper_key FROM papers")}
        n = 0
        pending: list[tuple] = []

        def flush() -> None:
            nonlocal pending
            if pending:
                with self.tx() as c:
                    c.executemany(self._DOC_UPSERT, pending)
                pending = []

        ts = now_iso()
        for entry in sorted(docs_dir.iterdir()):
            name = entry.name
            if not name.endswith((".json", ".json.gz")):
                continue
            try:
                data = entry.read_bytes()
                if name.endswith(".gz"):
                    data = gzip.decompress(data)
                doc = Document.model_validate_json(data)
            except Exception as e:  # noqa: BLE001
                log.warning("skipping unreadable document file %s: %s", name, e)
                continue
            if doc.paper_key in have:
                continue
            if doc.paper_key not in known:
                log.warning("skipping %s: paper_key %s not in papers table", name, doc.paper_key)
                continue
            pending.append(self._doc_row(doc, ts))
            have.add(doc.paper_key)
            n += 1
            if len(pending) >= batch:
                flush()
                log.info("imported %d documents", n)
        flush()
        return n

    def reconcile_from_store(self, store) -> dict[str, int]:
        """Bring the best-effort status columns in line with the vector store: papers whose
        full text is in the chunk table become ``embedded`` (with ``n_chunks``); papers in
        the paper-vector table get ``paper_vec=1``; papers with an abstract-only chunk get
        ``abs_chunk=1``.  Used after another machine embedded into the shared LanceDB."""
        counts = store.chunk_counts_by_paper()          # {paper_key: (n_fulltext, n_abstract)}
        pvec = store.paper_vec_keys()
        known = {r[0]: (r[1], r[2], r[3], r[4]) for r in self.conn.execute(
            "SELECT paper_key, status, n_chunks, paper_vec, abs_chunk FROM papers")}
        to_embedded, to_pvec, to_abs = [], [], []
        for key, (st, nch, pv, ab) in known.items():
            nf, na = counts.get(key, (0, 0))
            if nf and (st != Status.EMBEDDED or nch != nf):
                to_embedded.append((str(Status.EMBEDDED), nf, now_iso(), key))
            if key in pvec and not pv:
                to_pvec.append((key,))
            if na and not ab:
                to_abs.append((key,))
        with self.tx() as c:
            c.executemany("UPDATE papers SET status=?, n_chunks=?, error=NULL, updated_at=? WHERE paper_key=?",
                          to_embedded)
            c.executemany("UPDATE papers SET paper_vec=1 WHERE paper_key=?", to_pvec)
            c.executemany("UPDATE papers SET abs_chunk=1 WHERE paper_key=?", to_abs)
        return {"embedded": len(to_embedded), "paper_vec": len(to_pvec), "abs_chunk": len(to_abs),
                "store_fulltext_papers": sum(1 for v in counts.values() if v[0]),
                "store_paper_vectors": len(pvec)}

    def iter_without_paper_vec(self, *, limit: int | None = None) -> Iterator[sqlite3.Row]:
        q = "SELECT * FROM papers WHERE paper_vec=0 ORDER BY priority DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        yield from self.conn.execute(q)

    def mark_paper_vec(self, keys: Iterable[str], *, abs_chunk_keys: Iterable[str] = ()) -> None:
        with self.tx():  # one transaction: autocommit mode would fsync per row (slow on NFS)
            self.conn.executemany("UPDATE papers SET paper_vec=1 WHERE paper_key=?", [(k,) for k in keys])
            self.conn.executemany("UPDATE papers SET abs_chunk=1 WHERE paper_key=?",
                                  [(k,) for k in abs_chunk_keys])

    def set_status(self, paper_key: str, status: Status | str, *, source: str | None = None,
                   error: str | None = None, n_chunks: int | None = None) -> None:
        sets, vals = ["status=?", "updated_at=?"], [str(status), now_iso()]
        if source is not None:
            sets.append("source=?"); vals.append(source)
        if error is not None or str(status).startswith("failed"):
            sets.append("error=?"); vals.append(error)
        elif not str(status).startswith("failed"):
            sets.append("error=NULL")
        if n_chunks is not None:
            sets.append("n_chunks=?"); vals.append(n_chunks)
        vals.append(paper_key)
        self._execute_retry(f"UPDATE papers SET {', '.join(sets)} WHERE paper_key=?", vals)

    def _execute_retry(self, sql: str, vals, *, attempts: int = 6) -> None:
        """Execute a write, retrying on 'database is locked' (NFS lock stalls)."""
        for i in range(attempts):
            try:
                self.conn.execute(sql, vals)
                return
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) or i == attempts - 1:
                    raise
                time.sleep(5 * (i + 1))

    def iter_by_status(self, status: Status | str, *, tiers: Iterable[str] | None = None,
                       limit: int | None = None, order: str = "priority DESC") -> Iterator[sqlite3.Row]:
        q = "SELECT * FROM papers WHERE status=?"
        vals: list = [str(status)]
        if tiers:
            tiers = list(tiers)
            q += f" AND tier IN ({','.join('?' * len(tiers))})"
            vals += tiers
        q += f" ORDER BY {order}"
        if limit:
            q += f" LIMIT {int(limit)}"
        yield from self.conn.execute(q, vals)

    def iter_keys_by_status(self, statuses: Iterable[Status | str], *, tiers: Iterable[str] | None = None,
                            limit: int | None = None) -> list[tuple[str, float]]:
        """(paper_key, priority) for the given statuses, highest priority first.

        Reads only small columns, so it avoids touching the large ``meta_json`` overflow
        pages (important on network filesystems)."""
        statuses = [str(x) for x in statuses]
        q = f"SELECT paper_key, priority FROM papers WHERE status IN ({','.join('?' * len(statuses))})"
        vals: list = statuses
        if tiers:
            tiers = list(tiers)
            q += f" AND tier IN ({','.join('?' * len(tiers))})"
            vals += tiers
        q += " ORDER BY priority DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        return [(r[0], r[1]) for r in self.conn.execute(q, vals)]

    def iter_all_keys(self) -> list[tuple[str, float]]:
        """(paper_key, priority) for every paper, highest priority first (index-only read)."""
        return [(r[0], r[1]) for r in self.conn.execute(
            "SELECT paper_key, priority FROM papers ORDER BY priority DESC")]

    def count_by_status(self) -> dict[str, int]:
        return {r[0]: r[1] for r in self.conn.execute("SELECT status, COUNT(*) FROM papers GROUP BY status")}

    def count_by_tier(self) -> dict[str, int]:
        return {r[0]: r[1] for r in self.conn.execute("SELECT tier, COUNT(*) FROM papers GROUP BY tier")}

    def search_meta(self, *, author: str | None = None, year: int | None = None, title_like: str | None = None,
                    limit: int = 20) -> list[sqlite3.Row]:
        q, vals = "SELECT * FROM papers WHERE 1=1", []
        if author:
            q += " AND first_author LIKE ?"; vals.append(f"{author}%")
        if year:
            q += " AND year=?"; vals.append(year)
        if title_like:
            q += " AND title LIKE ?"; vals.append(f"%{title_like}%")
        q += " ORDER BY citation_count DESC LIMIT ?"; vals.append(limit)
        return self.conn.execute(q, vals).fetchall()

    # -------------------------------------------------------------------- refs
    def references_of(self, paper_key: str) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT cited_bibcode FROM refs WHERE citing_key=?", (paper_key,))]

    def cites(self, citing_key: str, cited_bibcode: str) -> bool:
        return self.conn.execute("SELECT 1 FROM refs WHERE citing_key=? AND cited_bibcode=?",
                                 (citing_key, cited_bibcode)).fetchone() is not None

    def local_citers_of(self, bibcode: str) -> list[str]:
        """Papers *in the corpus* that cite ``bibcode``."""
        return [r[0] for r in self.conn.execute("SELECT citing_key FROM refs WHERE cited_bibcode=?", (bibcode,))]

    # ------------------------------------------------------------------ bibtex
    def get_bibtex(self, bibcode: str) -> str | None:
        row = self.conn.execute("SELECT bibtex FROM bibtex WHERE bibcode=?", (bibcode,)).fetchone()
        return row[0] if row else None

    def put_bibtex(self, entries: dict[str, str]) -> None:
        ts = now_iso()
        self.conn.executemany("INSERT OR REPLACE INTO bibtex(bibcode,bibtex,fetched_at) VALUES(?,?,?)",
                              [(k, v, ts) for k, v in entries.items()])

    # -------------------------------------------------------------------- runs
    def start_run(self, kind: str, query: str | None = None) -> int:
        cur = self.conn.execute("INSERT INTO runs(kind,started_at,query) VALUES(?,?,?)", (kind, now_iso(), query))
        return cur.lastrowid

    def finish_run(self, run_id: int, *, n_new: int = 0, n_updated: int = 0, note: str | None = None) -> None:
        self.conn.execute("UPDATE runs SET finished_at=?, n_new=?, n_updated=?, note=? WHERE id=?",
                          (now_iso(), n_new, n_updated, note, run_id))

    def last_run(self, kind: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM runs WHERE kind=? AND finished_at IS NOT NULL ORDER BY id DESC LIMIT 1", (kind,)
        ).fetchone()

    def max_entdate(self) -> str | None:
        return self.conn.execute("SELECT MAX(entdate) FROM papers").fetchone()[0]


def compute_tier_and_priority(m: PaperMeta) -> tuple[Tier, float]:
    """Tier A/B/C and fetch priority as described in docs/PLAN.md §3."""
    import math

    s = get_settings()
    year = m.year or 0
    is_review = (m.bibstem in s.review_bibstems) or (m.doctype == "review")
    if (m.refereed and year >= s.tier_a_year_min) or is_review or m.citation_count >= s.tier_a_min_citations:
        tier = Tier.A
    elif (m.refereed and year >= s.year_min) or (not m.refereed and year >= 2024 and m.arxiv_id):
        tier = Tier.B
    else:
        tier = Tier.C
    prio = (math.log1p(max(m.citation_count, 0)) + 0.4 * max(year - 2000, 0) / 26
            + 2.0 * is_review + (1.0 if tier == Tier.A else 0.0))
    return tier, prio


def meta_from_ads(doc: dict) -> PaperMeta:
    """Normalise a raw ADS search document into :class:`PaperMeta`."""
    arxiv_id = None
    for ident in doc.get("identifier", []) or []:
        if ident.lower().startswith("arxiv:"):
            arxiv_id = normalise_arxiv_id(ident)
            break
    if arxiv_id is None:
        for d in doc.get("doi", []) or []:
            if d.lower().startswith("10.48550/arxiv."):
                arxiv_id = normalise_arxiv_id(d.split("arxiv.", 1)[1])
                break
    if arxiv_id is None:
        for ident in doc.get("identifier", []) or []:
            m = re.fullmatch(r"\d{4}arXiv(\d{4})(\d{5})[A-Z]", ident)
            if m:
                arxiv_id = f"{m.group(1)}.{m.group(2)}"
                break
    dois = [d for d in doc.get("doi", []) or [] if not d.lower().startswith("10.48550/arxiv.")]
    bibcode = doc.get("bibcode")
    props = doc.get("property", []) or []
    bibstem = (doc.get("bibstem") or [None])[0]
    year = int(doc["year"]) if str(doc.get("year", "")).isdigit() else None
    m = PaperMeta(
        paper_key=paper_key_for(bibcode, arxiv_id),
        bibcode=bibcode,
        arxiv_id=arxiv_id,
        doi=dois[0] if dois else None,
        title=(doc.get("title") or [""])[0],
        authors=doc.get("author", []) or [],
        first_author=doc.get("first_author") or "",
        year=year,
        pubdate=doc.get("pubdate"),
        entdate=doc.get("entry_date") or doc.get("entdate"),
        abstract=doc.get("abstract") or "",
        keywords=doc.get("keyword", []) or [],
        arxiv_class=doc.get("arxiv_class", []) or [],
        bibstem=bibstem,
        pub=doc.get("pub"),
        volume=doc.get("volume"),
        page=(doc.get("page") or [None])[0],
        doctype=doc.get("doctype"),
        refereed="REFEREED" in props,
        esources=doc.get("esources", []) or [],
        citation_count=int(doc.get("citation_count") or 0),
        read_count=int(doc.get("read_count") or 0),
        references=doc.get("reference", []) or [],
        raw=doc,
    )
    m.tier, m.priority = compute_tier_and_priority(m)
    return m
