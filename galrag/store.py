"""LanceDB vector store: chunk table + paper-level table, hybrid search.

LanceDB / FTS notes (verified on lancedb 0.40.0)
------------------------------------------------
* We use LanceDB's **native** inverted-index FTS (``create_index(..., config=FTS(...))``),
  not the legacy ``use_tantivy=True`` path: the tantivy path is deprecated, stores its
  index in a side directory outside the dataset (bad for NFS / multi-process) and
  cannot take a ``where`` pre-filter.  ``tantivy`` stays an installed dependency but is
  unused.  FTS is configured with positions, no stop-word removal (so "[O III]" and
  "z ~ 7" keep their tokens) and the English stemmer.
* After ``table.add`` the FTS index does **not** need re-creation: rows not yet in the
  index (``num_unindexed_rows`` in ``list_indices()``) are still searched by a flat scan
  and returned.  ``table.optimize()`` folds them into the index, which keeps queries fast
  after big ingests; :meth:`VectorStore.refresh_indexes` does exactly that.  The same
  holds for the IVF-PQ index (unindexed rows are brute-forced and merged).
* A native FTS search on a table without the inverted index raises, so the index is
  created lazily (first upsert / first FTS search).
* The FTS query string is reduced to word tokens before being sent, so characters such
  as ``( ) [ ] " : - ~`` in astronomy queries can never be parsed as query syntax.
* ``vector`` search uses cosine distance; embeddings are L2-normalised so cosine
  similarity = 1 - distance.
* ``-1`` offsets on abstract chunks are stored as-is.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pyarrow as pa

from galrag.config import Settings, get_settings
from galrag.models import Chunk, PaperMeta, SearchHit

VECTOR_INDEX_MIN_ROWS = 50_000
RRF_K = 60
_CHUNK_COLS = [
    "chunk_id", "paper_key", "ordinal", "kind", "section_id", "section_title", "section_path",
    "para_ids", "text", "char_start", "char_end", "page_start", "page_end", "n_tokens", "year",
    "bibcode", "arxiv_id", "first_author", "refereed", "citation_count", "source", "title",
    "authors_short",
]


def _q(s: str) -> str:
    """SQL string literal."""
    return "'" + str(s).replace("'", "''") + "'"


def _in_list(vals: Iterable[str]) -> str:
    return "(" + ", ".join(_q(v) for v in vals) + ")"


def build_where(filters: dict | None) -> str | None:
    """Translate the filter dict into a LanceDB SQL ``where`` string."""
    if not filters:
        return None
    c: list[str] = []
    if filters.get("year_from") is not None:
        c.append(f"year >= {int(filters['year_from'])}")
    if filters.get("year_to") is not None:
        c.append(f"year <= {int(filters['year_to'])}")
    if filters.get("first_author"):
        pre = re.sub(r"[%_\\]", "", str(filters["first_author"]).strip().lower())
        if pre:
            c.append(f"lower(first_author) LIKE {_q(pre + '%')}")
    if filters.get("refereed_only"):
        c.append("refereed = true")
    if filters.get("paper_keys"):
        c.append(f"paper_key IN {_in_list(filters['paper_keys'])}")
    if filters.get("kinds"):
        c.append(f"kind IN {_in_list(filters['kinds'])}")
    if filters.get("min_citations") is not None:
        c.append(f"citation_count >= {int(filters['min_citations'])}")
    if filters.get("source_in"):
        c.append(f"source IN {_in_list(filters['source_in'])}")
    return " AND ".join(c) or None


def fts_query(q: str) -> str:
    """Reduce a free-text query to space-separated word tokens (OR semantics in BM25)."""
    toks = re.findall(r"\w+", q, flags=re.UNICODE)
    return " ".join(toks)


def _vec_array(vectors: np.ndarray) -> pa.Array:
    v = np.ascontiguousarray(vectors, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(v.ravel(), pa.float32()), v.shape[1])


class VectorStore:
    def __init__(self, settings: Settings | None = None, embedder=None, reranker=None,
                 uri: str | None = None):
        import lancedb

        self.settings = settings or get_settings()
        self.uri = str(uri or self.settings.lancedb_dir)
        self.db = lancedb.connect(self.uri)
        self._embedder = embedder
        self._reranker = reranker
        self._chunks = None
        self._papers = None

    # ------------------------------------------------------------ helpers
    @property
    def embedder(self):
        if self._embedder is None:
            from galrag.embed import Embedder

            self._embedder = Embedder(self.settings)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from galrag.embed import Reranker

            self._reranker = Reranker(self.settings)
        return self._reranker

    def _open(self, name: str):
        try:
            return self.db.open_table(name)
        except Exception:  # noqa: BLE001
            return None

    @property
    def chunks(self):
        if self._chunks is None:
            self._chunks = self._open(self.settings.chunks_table)
        return self._chunks

    @property
    def papers(self):
        if self._papers is None:
            self._papers = self._open(self.settings.papers_table)
        return self._papers

    @staticmethod
    def _chunk_schema(dim: int) -> pa.Schema:
        return pa.schema([
            ("chunk_id", pa.string()), ("paper_key", pa.string()), ("ordinal", pa.int32()),
            ("kind", pa.string()), ("section_id", pa.string()), ("section_title", pa.string()),
            ("section_path", pa.string()), ("para_ids", pa.list_(pa.string())),
            ("text", pa.string()), ("char_start", pa.int64()), ("char_end", pa.int64()),
            ("page_start", pa.int32()), ("page_end", pa.int32()), ("n_tokens", pa.int32()),
            ("year", pa.int32()), ("bibcode", pa.string()), ("arxiv_id", pa.string()),
            ("first_author", pa.string()), ("refereed", pa.bool_()),
            ("citation_count", pa.int32()), ("source", pa.string()), ("title", pa.string()),
            ("authors_short", pa.string()), ("vector", pa.list_(pa.float32(), dim)),
        ])

    @staticmethod
    def _paper_schema(dim: int) -> pa.Schema:
        return pa.schema([
            ("paper_key", pa.string()), ("title", pa.string()), ("year", pa.int32()),
            ("first_author", pa.string()), ("citation_count", pa.int32()),
            ("refereed", pa.bool_()), ("vector", pa.list_(pa.float32(), dim)),
        ])

    def _delete_keys(self, table, keys: Sequence[str]) -> None:
        keys = list(dict.fromkeys(keys))
        for i in range(0, len(keys), 500):
            table.delete(f"paper_key IN {_in_list(keys[i:i + 500])}")

    # ------------------------------------------------------------ chunks
    def count(self) -> int:
        t = self.chunks
        return int(t.count_rows()) if t is not None else 0

    def has_paper(self, paper_key: str) -> bool:
        t = self.chunks
        return t is not None and t.count_rows(f"paper_key = {_q(paper_key)}") > 0

    @staticmethod
    def _scan(table, columns: list[str], where: str | None = None) -> pa.Table:
        """Projected full scan (reads only ``columns``; no vectors)."""
        q = table.search().select(columns)
        if where:
            q = q.where(where)
        return q.limit(None).to_arrow()

    def fulltext_paper_keys(self) -> set[str]:
        """Papers with full-text chunks (any chunk with ``kind != 'abstract'``)."""
        t = self.chunks
        if t is None:
            return set()
        return set(self._scan(t, ["paper_key"], "kind != 'abstract'").column("paper_key").to_pylist())

    def paper_vec_keys(self) -> set[str]:
        """Papers that have a title+abstract vector in the paper-level table."""
        t = self.papers
        if t is None:
            return set()
        return set(self._scan(t, ["paper_key"]).column("paper_key").to_pylist())

    def chunk_counts_by_paper(self) -> dict[str, tuple[int, int]]:
        """{paper_key: (n_fulltext_chunks, n_abstract_only_chunks)}.  ``n_fulltext_chunks``
        counts every chunk of a full-text paper (as ``papers.n_chunks`` does)."""
        t = self.chunks
        if t is None:
            return {}
        tbl = self._scan(t, ["paper_key", "source"])
        out: dict[str, list[int]] = {}
        for key, src in zip(tbl.column("paper_key").to_pylist(), tbl.column("source").to_pylist()):
            e = out.setdefault(key, [0, 0])
            e[1 if src == "abstract_only" else 0] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}

    def delete_paper(self, paper_key: str) -> None:
        for t in (self.chunks,):
            if t is not None:
                self._delete_keys(t, [paper_key])
        if self.papers is not None:
            self._delete_keys(self.papers, [paper_key])

    def upsert_chunks(self, chunks: Sequence[Chunk], vectors: np.ndarray,
                      paper_info: dict[str, dict[str, str]] | None = None) -> int:
        """Replace all rows of the papers in ``chunks`` by ``chunks`` (idempotent).

        ``paper_info`` optionally maps paper_key -> {"title":..., "authors_short":...}
        (stored per row so ``SearchHit.title`` needs no join).
        """
        if not chunks:
            return 0
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(chunks):
            raise ValueError("vectors must have shape (len(chunks), dim)")
        dim = vectors.shape[1]
        info = paper_info or {}
        rows = {c: [] for c in _CHUNK_COLS}
        for ch in chunks:
            d = ch.model_dump()
            pi = info.get(ch.paper_key, {})
            d["title"] = pi.get("title", "")
            d["authors_short"] = pi.get("authors_short", f"{ch.first_author} et al." if ch.first_author else "")
            for k in _CHUNK_COLS:
                rows[k].append(d[k])
        schema = self._chunk_schema(dim)
        tbl = pa.table({**{k: pa.array(rows[k], type=schema.field(k).type) for k in _CHUNK_COLS},
                        "vector": _vec_array(vectors)}, schema=schema)
        t = self.chunks
        if t is None:
            t = self.db.create_table(self.settings.chunks_table, schema=schema)
            self._chunks = t
        else:
            self._delete_keys(t, [c.paper_key for c in chunks])
        t.add(tbl)
        self._ensure_fts(t)
        return len(chunks)

    def _ensure_fts(self, t) -> None:
        try:
            if any(i.index_type == "FTS" for i in t.list_indices()):
                return
        except Exception:  # noqa: BLE001, S110
            pass
        self._create_fts(t)

    @staticmethod
    def _create_fts(t) -> None:
        from lancedb.index import FTS

        t.create_index("text", config=FTS(with_position=True, remove_stop_words=False,
                                          stem=True, lower_case=True, ascii_folding=True),
                       replace=True)

    def create_indexes(self, force: bool = False) -> dict[str, Any]:
        """(Re)build FTS, scalar and (if n > 50k rows) IVF-PQ vector indexes."""
        out: dict[str, Any] = {}
        t = self.chunks
        if t is None:
            return out
        n = t.count_rows()
        self._create_fts(t) if force else self._ensure_fts(t)
        out["fts"] = True
        try:
            from lancedb.index import BTree

            t.create_index("paper_key", config=BTree(), replace=True)
            out["paper_key_btree"] = True
        except Exception as e:  # pragma: no cover  # noqa: BLE001
            out["paper_key_btree"] = f"failed: {e}"
        has_vec = any(i.index_type.startswith("IVF") for i in t.list_indices())
        if n > VECTOR_INDEX_MIN_ROWS and (force or not has_vec):
            from lancedb.index import IvfPq

            dim = self.dim
            t.create_index("vector", config=IvfPq(distance_type="cosine",
                                        num_partitions=max(1, int(math.sqrt(n))),
                                        num_sub_vectors=max(1, dim // 8)),
                           replace=True)
            out["ivf_pq"] = {"num_partitions": int(math.sqrt(n))}
        else:
            out["ivf_pq"] = "skipped (brute force)" if n <= VECTOR_INDEX_MIN_ROWS else "exists"
        return out

    def refresh_indexes(self) -> None:
        """Fold rows appended since the last index build into the indexes
        (``Table.optimize``).  Not required for correctness (unindexed rows are
        scanned), only for speed after large ingests."""
        for t in (self.chunks, self.papers):
            if t is not None:
                t.optimize()

    @property
    def dim(self) -> int:
        t = self.chunks
        if t is not None:
            return int(t.schema.field("vector").type.list_size)
        return self.embedder.dim

    # ------------------------------------------------------------ search
    def _row_to_chunk(self, r: dict) -> Chunk:
        d = {k: r.get(k) for k in Chunk.model_fields if k in r}
        d["para_ids"] = list(d.get("para_ids") or [])
        return Chunk(**d)

    def _vector_hits(self, qvec: np.ndarray, n: int, where: str | None) -> list[dict]:
        t = self.chunks
        q = t.search(qvec.tolist(), vector_column_name="vector").distance_type("cosine").limit(n)
        if where:
            q = q.where(where, prefilter=True)
        q = q.select(_CHUNK_COLS + ["_distance"])
        if any(i.index_type.startswith("IVF") for i in t.list_indices()):
            q = q.nprobes(20).refine_factor(5)
        return q.to_list()

    def _fts_hits(self, query: str, n: int, where: str | None) -> list[dict]:
        fq = fts_query(query)
        if not fq:
            return []
        t = self.chunks
        self._ensure_fts(t)
        q = t.search(fq, query_type="fts").limit(n)
        if where:
            q = q.where(where, prefilter=True)
        return q.select(_CHUNK_COLS + ["_score"]).to_list()

    def search(self, query: str, k: int = 10, mode: str = "hybrid",
               filters: dict | None = None, rerank: bool | None = None) -> list[SearchHit]:
        if mode not in ("hybrid", "vector", "fts"):
            raise ValueError("mode must be hybrid|vector|fts")
        if self.chunks is None or self.chunks.count_rows() == 0 or not query.strip():
            return []
        where = build_where(filters)
        do_rerank = self.settings.rerank_enabled if rerank is None else rerank
        n = max(k * 5, 50)
        if do_rerank:
            n = max(n, self.settings.rerank_top_n)

        vec_rows: list[dict] = []
        fts_rows: list[dict] = []
        if mode in ("hybrid", "vector"):
            qv = self.embedder.embed_queries([query])[0]
            vec_rows = self._vector_hits(qv, n, where)
        if mode in ("hybrid", "fts"):
            fts_rows = self._fts_hits(query, n, where)

        cand: dict[str, dict] = {}
        for rank, r in enumerate(vec_rows, 1):
            e = cand.setdefault(r["chunk_id"], {"row": r, "scores": {}})
            e["scores"]["vector"] = 1.0 - float(r.get("_distance", 0.0))
            e["scores"]["vector_rank"] = float(rank)
            e["scores"]["rrf"] = e["scores"].get("rrf", 0.0) + 1.0 / (RRF_K + rank)
        for rank, r in enumerate(fts_rows, 1):
            e = cand.setdefault(r["chunk_id"], {"row": r, "scores": {}})
            e["scores"]["fts"] = float(r.get("_score", 0.0))
            e["scores"]["fts_rank"] = float(rank)
            e["scores"]["rrf"] = e["scores"].get("rrf", 0.0) + 1.0 / (RRF_K + rank)

        primary = {"hybrid": "rrf", "vector": "vector", "fts": "fts"}[mode]
        ordered = sorted(cand.values(), key=lambda e: -e["scores"][primary])

        if do_rerank and ordered:
            top = ordered[: self.settings.rerank_top_n]
            rs = self.reranker.rerank(query, [e["row"]["text"] for e in top])
            for e, s in zip(top, rs):
                e["scores"]["rerank"] = s
            top.sort(key=lambda e: -e["scores"]["rerank"])
            ordered, primary = top, "rerank"

        hits = []
        for i, e in enumerate(ordered[:k], 1):
            r = e["row"]
            hits.append(SearchHit(chunk=self._row_to_chunk(r), score=e["scores"][primary], rank=i,
                                  title=r.get("title") or "", authors_short=r.get("authors_short") or "",
                                  scores=e["scores"]))
        return hits

    # ------------------------------------------------------------ papers
    def upsert_papers(self, items: Sequence[PaperMeta | dict], vectors: np.ndarray) -> int:
        """``items``: PaperMeta or dicts with paper_key/title/year/first_author/
        citation_count/refereed; ``vectors`` embed title+abstract (passage mode)."""
        if not items:
            return 0
        vectors = np.asarray(vectors, dtype=np.float32)
        ds = [i if isinstance(i, dict) else i.model_dump() for i in items]
        schema = self._paper_schema(vectors.shape[1])
        cols = {
            "paper_key": [d["paper_key"] for d in ds], "title": [d.get("title", "") or "" for d in ds],
            "year": [d.get("year") for d in ds], "first_author": [d.get("first_author", "") or "" for d in ds],
            "citation_count": [int(d.get("citation_count") or 0) for d in ds],
            "refereed": [bool(d.get("refereed")) for d in ds],
        }
        tbl = pa.table({**{k: pa.array(v, type=schema.field(k).type) for k, v in cols.items()},
                        "vector": _vec_array(vectors)}, schema=schema)
        t = self.papers
        if t is None:
            t = self.db.create_table(self.settings.papers_table, schema=schema)
            self._papers = t
        else:
            self._delete_keys(t, cols["paper_key"])
        t.add(tbl)
        return len(ds)

    def search_papers(self, query: str, k: int = 10, filters: dict | None = None) -> list[dict]:
        t = self.papers
        if t is None or t.count_rows() == 0 or not query.strip():
            return []
        qv = self.embedder.embed_queries([query])[0]
        q = t.search(qv.tolist(), vector_column_name="vector").distance_type("cosine").limit(k)
        where = build_where(filters)
        if where:
            q = q.where(where, prefilter=True)
        out = []
        for r in q.select(["paper_key", "title", "year", "first_author", "citation_count",
                           "refereed"]).to_list():
            r["score"] = 1.0 - float(r.pop("_distance", 0.0))
            out.append(r)
        return out
