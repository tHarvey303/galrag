"""MCP server (stdio) exposing the galrag corpus to LLM clients.

Run with ``galrag-mcp`` / ``galrag serve`` / ``python -m galrag.server``.

Design notes
------------
* stdout belongs to the JSON-RPC transport: logging goes to stderr (WARNING) and to
  ``<data_dir>/logs/galrag.log``; nothing is printed at import time.
* One :class:`~galrag.store.VectorStore` per process (lazy; the embedding model is
  loaded on the first semantic query).  SQLite connections are per thread because
  the SDK runs synchronous tools in worker threads.
* Every passage carries provenance (paper_key, bibcode, arxiv_id, chunk_id,
  section_path, char_start/char_end into the document's ``full_text``, page,
  source) and a citation string "Author et al. (Year), bibcode".
* Tests can point the server at a temporary corpus with :func:`configure`.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _MCPServer
except ImportError:  # pragma: no cover - mcp 1.x
    from mcp.server.fastmcp import FastMCP as _MCPServer

from galrag import service
from galrag.db import Database
from galrag.models import Chunk, Document, Paragraph, Section, Status

log = logging.getLogger("galrag.server")

INSTRUCTIONS = """\
galrag: a local, citable corpus of the galaxy-evolution / high-redshift literature
(NASA ADS metadata for every paper; arXiv full text, chunked with exact character
offsets, for the most-cited and recent refereed papers; abstracts for the rest).

Workflow for literature answers:
1. search_passages (hybrid BM25 + dense retrieval, optional reranker) or
   search_papers (title+abstract) to find evidence; filter by year/author/refereed.
2. read_paragraphs / read_section around a hit to get context.
3. get_quote(ident, text) to confirm any quote is verbatim before presenting it
   as a quotation; it returns the exact span, section, offsets and a similarity.
4. Cite by ADS bibcode, using the `citation` string returned with each passage.

how_to_cite: cite every claim as "Author et al. (Year), <bibcode>"; only put text
in quotation marks if get_quote returned exact=true (or similarity >= 95) for it;
otherwise paraphrase. Passages with kind="abstract" or source="abstract_only"
come from the ADS abstract, not the full text. PDF-derived text (source
arxiv_pdf / pub_pdf) can contain extraction artefacts. If the corpus lacks a
paper, say so (or use ingest_paper) rather than guessing.

Identifiers (`ident`) may be an ADS bibcode, an arXiv id (2405.03781 or
arXiv:2405.03781), a DOI, or the internal paper_key.
"""

mcp = _MCPServer("galrag", instructions=INSTRUCTIONS)

# --------------------------------------------------------------------------- state
_lock = threading.Lock()
_local = threading.local()
_state: dict[str, Any] = {"db_path": None, "db": None, "db_thread": None, "store": None}


def configure(db: Database | None = None, store=None) -> None:
    """Point the server at a specific Database / VectorStore (tests, embedding)."""
    with _lock:
        _state["db"] = db
        _state["db_path"] = db.path if db is not None else None
        _state["db_thread"] = threading.get_ident() if db is not None else None
        _state["store"] = store
        _local.__dict__.clear()


def get_db() -> Database:
    """Database for the current thread (sqlite3 connections are thread-bound)."""
    if _state["db"] is not None and _state["db_thread"] == threading.get_ident():
        return _state["db"]
    db = getattr(_local, "db", None)
    if db is None:
        db = Database(_state["db_path"])
        _local.db = db
    return db


def get_store():
    if _state["store"] is None:
        with _lock:
            if _state["store"] is None:
                from galrag.store import VectorStore

                _state["store"] = VectorStore()
    return _state["store"]


# --------------------------------------------------------------------------- helpers
class _NotFound(Exception):
    pass


def _resolve(ident: str) -> str:
    key = get_db().resolve(ident.strip()) if ident else None
    if key is None:
        raise _NotFound(f"unknown paper {ident!r} (not in the corpus; try resolve_citation or ingest_paper)")
    return key


def _err(e: Exception | str) -> dict:
    return {"error": str(e)}


def _paper_brief(key: str) -> dict:
    row = get_db().get_row(key)
    if row is None:
        return {"paper_key": key}
    return {
        "paper_key": key, "bibcode": row["bibcode"], "arxiv_id": row["arxiv_id"],
        "title": row["title"], "first_author": row["first_author"], "year": row["year"],
        "citation_count": row["citation_count"], "refereed": bool(row["refereed"]),
        "has_fulltext": row["status"] == Status.EMBEDDED,
        "citation": service.citation_string(row["first_author"], row["year"], row["bibcode"], key),
    }


def _load_doc(key: str) -> Document | None:
    from galrag.parse import load_document

    return load_document(key)


def _require_doc(key: str) -> Document:
    doc = _load_doc(key)
    if doc is None:
        row = get_db().get_row(key)
        st = row["status"] if row is not None else "unknown"
        raise _NotFound(f"no full text for {key} (status={st}); use get_paper for the abstract "
                        f"or ingest_paper to fetch it")
    return doc


def _citation_for_key(key: str) -> str:
    row = get_db().get_row(key)
    if row is None:
        return key
    return service.citation_string(row["first_author"], row["year"], row["bibcode"], key)


def _hit_dict(h, *, max_text: int | None = None) -> dict:
    c: Chunk = h.chunk
    text = c.text if not max_text or len(c.text) <= max_text else c.text[:max_text] + " [...]"
    cit = (f"{h.authors_short}, {c.bibcode or c.paper_key}" if h.authors_short
           else service.citation_string(c.first_author, c.year, c.bibcode, c.paper_key))
    return {
        "rank": h.rank, "score": round(float(h.score), 4), "chunk_id": c.chunk_id,
        "paper_key": c.paper_key, "bibcode": c.bibcode, "arxiv_id": c.arxiv_id,
        "citation": cit, "title": h.title, "year": c.year, "section_path": c.section_path,
        "section_id": c.section_id, "para_ids": c.para_ids, "kind": c.kind, "text": text,
        "char_start": c.char_start, "char_end": c.char_end, "page_start": c.page_start,
        "source": c.source,
        # body/caption chunks: text == full_text[char_start:char_end]; abstract chunks are
        # "title\n\nabstract" and offsets (if >= 0) span the abstract paragraphs.
        "verbatim_span": c.kind != "abstract" and c.char_start >= 0,
    }


def _filters(**kw) -> dict:
    return {k: v for k, v in kw.items() if v not in (None, False, [], "")}


def _para_dict(s: Section | None, p: Paragraph) -> dict:
    d = {"para_id": p.id, "kind": p.kind, "text": p.text, "char_start": p.char_start,
         "char_end": p.char_end}
    if s is not None:
        d["section_id"], d["section_title"] = s.id, s.title
    if p.page is not None:
        d["page"] = p.page
    return d


def chunks_of_paper(store, paper_key: str, columns: list[str] | None = None) -> list[dict]:
    """All chunk rows of one paper, ordered by ordinal."""
    t = store.chunks
    if t is None:
        return []
    cols = columns or ["chunk_id", "ordinal", "kind", "section_id", "section_path", "char_start",
                       "char_end", "page_start", "n_tokens"]
    key = paper_key.replace("'", "''")
    rows = t.search().where(f"paper_key = '{key}'").select(cols).limit(100_000).to_list()
    return sorted(rows, key=lambda r: r.get("ordinal", 0))


# =========================================================================== tools
@mcp.tool()
def search_passages(query: str, k: int = 10, mode: str = "hybrid", year_from: int | None = None,
                    year_to: int | None = None, first_author: str | None = None,
                    refereed_only: bool = False, paper_keys: list[str] | None = None,
                    kinds: list[str] | None = None, min_citations: int | None = None,
                    rerank: bool | None = None) -> list[dict] | dict:
    """Retrieve passages (chunks of ~350 tokens) relevant to `query`.

    mode: "hybrid" (BM25 + dense, default), "vector" or "fts" (exact terms, good for
    object names like GN-z11). Filters: year range, first_author surname prefix,
    refereed_only, paper_keys (restrict to papers; any identifiers), kinds
    (["abstract","body","caption","footnote"]), min_citations. rerank: cross-encoder
    rerank (default from settings; slower on first call).
    Each hit carries provenance (bibcode, chunk_id, section_path, char_start/char_end into
    the paper's full text, page) and a `citation` string to cite it by."""
    try:
        if paper_keys:
            paper_keys = [get_db().resolve(p) or p for p in paper_keys]
        f = _filters(year_from=year_from, year_to=year_to, first_author=first_author,
                     refereed_only=refereed_only, paper_keys=paper_keys, kinds=kinds,
                     min_citations=min_citations)
        hits = get_store().search(query, k=max(1, min(int(k), 100)), mode=mode, filters=f or None,
                                  rerank=rerank)
        return [_hit_dict(h) for h in hits]
    except Exception as e:
        log.exception("search_passages failed")
        return _err(e)


@mcp.tool()
def search_papers(query: str, k: int = 10, year_from: int | None = None, year_to: int | None = None,
                  first_author: str | None = None, refereed_only: bool = False,
                  min_citations: int | None = None) -> list[dict] | dict:
    """Paper-level semantic search over title+abstract vectors (covers every paper in the
    corpus, including those without full text). Returns paper_key, bibcode, title,
    authors_short, year, citation_count, refereed, has_fulltext, score."""
    try:
        f = _filters(year_from=year_from, year_to=year_to, first_author=first_author,
                     refereed_only=refereed_only, min_citations=min_citations)
        res = get_store().search_papers(query, k=max(1, min(int(k), 100)), filters=f or None)
        out = []
        for r in res:
            b = _paper_brief(r["paper_key"])
            b["authors_short"] = service.authors_short(r.get("first_author"))
            b["score"] = round(float(r["score"]), 4)
            out.append(b)
        return out
    except Exception as e:
        log.exception("search_papers failed")
        return _err(e)


@mcp.tool()
def find_papers(author: str | None = None, year: int | None = None, title: str | None = None,
                limit: int = 20) -> list[dict]:
    """Metadata lookup (no embeddings): first-author surname prefix, exact year and/or a
    title substring. Sorted by citation count. Useful to find "Smith et al. 2023"."""
    if author:
        author = service.surname(author) or author
    rows = get_db().search_meta(author=author, year=year, title_like=title, limit=max(1, min(limit, 200)))
    return [_paper_brief(r["paper_key"]) for r in rows]


@mcp.tool()
def get_paper(ident: str) -> dict:
    """Metadata, abstract, ingestion status, full-text source, section outline
    (ids, titles, paragraph counts) and ADS/arXiv links for one paper."""
    try:
        key = _resolve(ident)
    except _NotFound as e:
        return _err(e)
    db = get_db()
    meta, row = db.get_meta(key), db.get_row(key)
    out = _paper_brief(key)
    out.update({
        "authors": meta.authors[:30] + (["..."] if len(meta.authors) > 30 else []),
        "n_authors": len(meta.authors), "doi": meta.doi, "pub": meta.pub, "pubdate": meta.pubdate,
        "doctype": meta.doctype, "keywords": meta.keywords[:20], "abstract": meta.abstract,
        "tier": row["tier"], "status": row["status"], "source": row["source"], "n_chunks": row["n_chunks"],
        "ingest_error": row["error"], "n_references": len(meta.references),
        "links": {"ads": service.ads_url(meta.bibcode), "arxiv": service.arxiv_url(meta.arxiv_id)},
    })
    doc = _load_doc(key)
    out["fulltext_available"] = doc is not None
    if doc is not None:
        out["parser_version"] = doc.parser_version
        out["quality_flags"] = doc.quality_flags
        out["n_pages"] = doc.n_pages
        out["n_chars"] = len(doc.full_text)
        out["sections"] = [
            {"section_id": s.id, "title": s.title, "level": s.level, "n_paragraphs": len(s.paragraphs),
             **({"page": s.paragraphs[0].page} if s.paragraphs and s.paragraphs[0].page else {})}
            for s in doc.sections]
        out["n_bibliography"] = len(doc.bibliography)
    return out


_SEC_NUM = re.compile(r"^\s*(?:[0-9]+|[IVXLC]+|[A-Z])(?:\.[0-9A-Za-z]+)*\.?\s+")


@mcp.tool()
def read_section(ident: str, section_id: str, max_chars: int = 20000) -> dict:
    """Paragraphs of one section (by section_id from get_paper, e.g. "S3" or "S3.SS1", or a
    case-insensitive title prefix such as "Results"), with paragraph ids and offsets."""
    try:
        key = _resolve(ident)
        doc = _require_doc(key)
    except _NotFound as e:
        return _err(e)
    sec = next((s for s in doc.sections if s.id == section_id), None)
    if sec is None:
        want = section_id.strip().lower()
        sec = next((s for s in doc.sections if _SEC_NUM.sub("", s.title).lower().startswith(want)
                    or s.title.lower().startswith(want)), None)
    if sec is None:
        return _err(f"no section {section_id!r}; available: {[s.id for s in doc.sections]}")
    paras, used, truncated = [], 0, False
    for p in sec.paragraphs:
        if used + len(p.text) > max_chars and paras:
            truncated = True
            break
        paras.append(_para_dict(None, p))
        used += len(p.text)
    # include subsections' ids so the client can continue
    subs = [s.id for s in doc.sections if s.id.startswith(sec.id + ".")]
    return {"paper_key": key, "citation": _citation_for_key(key), "source": doc.source,
            "section_id": sec.id, "section_title": sec.title, "paragraphs": paras,
            "truncated": truncated, "subsections": subs}


@mcp.tool()
def read_paragraphs(ident: str, para_ids: list[str] | None = None, char_start: int | None = None,
                    char_end: int | None = None, context_chars: int = 0) -> dict:
    """Deterministic read of a paper's full text: either paragraphs by id (from
    search_passages `para_ids` or read_section), or the exact span
    full_text[char_start:char_end] (from a passage's offsets), optionally with
    `context_chars` of surrounding text. Use it to verify a passage verbatim."""
    try:
        key = _resolve(ident)
        doc = _require_doc(key)
    except _NotFound as e:
        return _err(e)
    out: dict[str, Any] = {"paper_key": key, "citation": _citation_for_key(key), "source": doc.source}
    if para_ids:
        found, missing = [], []
        for pid in para_ids:
            hit = next(((s, p) for s, p in doc.iter_paragraphs() if p.id == pid), None)
            if hit is None:
                missing.append(pid)
            else:
                found.append(_para_dict(*hit))
        out["paragraphs"] = found
        if missing:
            out["missing_para_ids"] = missing
        return out
    if char_start is None or char_end is None:
        return _err("give para_ids or char_start and char_end")
    n = len(doc.full_text)
    a, b = max(0, int(char_start)), min(n, int(char_end))
    if a >= b:
        return _err(f"empty span [{char_start}:{char_end}] (full_text has {n} chars)")
    out.update({"char_start": a, "char_end": b, "text": doc.full_text[a:b]})
    if context_chars > 0:
        c = int(context_chars)
        out["context_before"] = doc.full_text[max(0, a - c):a]
        out["context_after"] = doc.full_text[b:min(n, b + c)]
    from galrag.quotes import locate_paragraph

    loc = locate_paragraph(doc, a)
    if loc:
        s, p = loc
        out.update({"section_id": s.id, "section_title": s.title, "para_id": p.id, "page": p.page})
    return out


def _quote_dict(m, key: str) -> dict:
    d = m.model_dump()
    d["section"] = d.pop("section_title")
    d["citation"] = _citation_for_key(key)
    d["similarity"] = round(float(d["similarity"]), 1)
    return d


@mcp.tool()
def get_quote(ident: str, text: str, min_similarity: float = 60.0) -> dict:
    """Locate `text` in a paper's full text: use to confirm that a quote is verbatim
    before citing it. Tries an exact (whitespace/case-insensitive) match first, then a
    fuzzy sentence-window match. Returns found, exact, similarity (0-100), the verbatim
    `quote` from the paper, section, para_id, char offsets, page and a citation string.
    Only present text as a direct quotation when exact=true; otherwise quote the returned
    `quote` or paraphrase."""
    try:
        key = _resolve(ident)
        doc = _require_doc(key)
    except _NotFound as e:
        return _err(e)
    from galrag.quotes import find_quote

    return _quote_dict(find_quote(doc, text, min_similarity=min_similarity), key)


def _abstract_doc(key: str, text: str) -> Document:
    p = Paragraph(id="abs.P1", kind="abstract", text=text, char_start=0, char_end=len(text))
    return Document(paper_key=key, source="abstract_only", full_text=text,
                    sections=[Section(id="abs", title="Abstract", paragraphs=[p])])


@mcp.tool()
def verify_claim(claim: str, k: int = 8, paper_keys: list[str] | None = None,
                 year_from: int | None = None, year_to: int | None = None,
                 first_author: str | None = None, refereed_only: bool = False,
                 rerank: bool | None = None) -> dict:
    """Find the passages that best support (or contradict) `claim`, corpus-wide or
    within `paper_keys`. For each hit, also returns the closest sentence(s) to the
    claim inside that paper (`best_match`: similarity 0-100, verbatim quote, offsets),
    so you can judge whether the paper actually states it. The judgement is yours:
    high similarity means similar wording, not agreement."""
    hits = search_passages(claim, k=k, mode="hybrid", year_from=year_from, year_to=year_to,
                           first_author=first_author, refereed_only=refereed_only,
                           paper_keys=paper_keys, rerank=rerank)
    if isinstance(hits, dict):
        return hits
    from galrag.quotes import find_quote

    best: dict[str, dict] = {}
    for h in hits:
        key = h["paper_key"]
        if key not in best:
            doc = _load_doc(key)
            if doc is None or not doc.full_text:
                m = find_quote(_abstract_doc(key, h["text"]), claim)
                d = _quote_dict(m, key)
                d["offsets_in"] = "abstract text (no full text available)"
            else:
                d = _quote_dict(find_quote(doc, claim), key)
                d["offsets_in"] = "full_text"
            best[key] = {k_: d[k_] for k_ in ("found", "exact", "similarity", "quote", "char_start",
                                              "char_end", "section", "para_id", "page", "offsets_in")}
        h["best_match_in_paper"] = best[key]
    return {"claim": claim, "hits": hits,
            "note": "best_match_in_paper is the closest wording anywhere in that paper; "
                    "read the passage text to judge support vs contradiction."}


@mcp.tool()
def citation_contexts(citing_ident: str, cited_ident: str | None = None,
                      cited_first_author: str | None = None, cited_year: int | None = None,
                      max_contexts: int = 20) -> dict:
    """Where (and whether) paper A cites paper B: the sentences of A that cite B, with
    section, offsets and the citation marker. Identify B by `cited_ident` (bibcode /
    arXiv / DOI) and/or by first-author surname + year. Also reports whether B is in A's
    ADS reference list (`in_ads_reference_list`)."""
    try:
        key = _resolve(citing_ident)
    except _NotFound as e:
        return _err(e)
    db = get_db()
    bibcode = None
    if cited_ident:
        ck = db.resolve(cited_ident)
        if ck is not None:
            row = db.get_row(ck)
            bibcode = row["bibcode"] or ck
            if not cited_first_author:
                cited_first_author = row["first_author"] or None
            if not cited_year:
                cited_year = row["year"]
        else:
            bibcode = cited_ident.strip()
    if not (bibcode or (cited_first_author and cited_year)):
        return _err("give cited_ident or cited_first_author + cited_year")
    out: dict[str, Any] = {"citing": _paper_brief(key), "cited_bibcode": bibcode,
                           "cited_first_author": cited_first_author, "cited_year": cited_year}
    if bibcode:
        out["in_ads_reference_list"] = db.cites(key, bibcode)
    doc = _load_doc(key)
    if doc is None:
        out["contexts"] = []
        out["note"] = "citing paper has no full text; only the ADS reference list was checked"
        return out
    from galrag.quotes import citation_contexts as _cc

    ctxs = _cc(doc, bibcode=bibcode, first_author=service.surname(cited_first_author) or None,
               year=cited_year)
    out["bib_entries"] = [
        {"key": b.key, "raw_text": b.raw_text[:300], "bibcode": b.bibcode}
        for b in doc.bibliography
        if (bibcode and b.bibcode == bibcode)
        or (cited_year and b.year == cited_year and cited_first_author
            and service.surname(b.first_author).lower() == service.surname(cited_first_author).lower())
    ][:5]
    out["n_contexts"] = len(ctxs)
    out["contexts"] = [
        {"sentence": c.sentence, "marker": c.marker, "section_id": c.section_id,
         "section_title": c.section_title, "para_id": c.para_id, "char_start": c.char_start,
         "char_end": c.char_end, "page": c.page}
        for c in ctxs[:max_contexts]]
    return out


_AUTHOR_YEAR = re.compile(
    r"^\s*([A-Z][\w'À-ɏ-]+(?:\s+[A-Z][\w'À-ɏ-]+)?)"
    r"(?:\s+et\s+al\.?|\s*(?:&|and)\s+[A-Z][\w'-]+)?\s*,?\s*\(?((?:19|20)\d{2})[a-z]?\)?")


@mcp.tool()
def resolve_citation(text: str, max_results: int = 5) -> dict:
    """Resolve a free-text citation ("Madau & Dickinson 2014", "Finkelstein et al. 2024,
    ApJ 969 L2", an arXiv id or DOI) to bibcode candidates: first in the local corpus,
    then via the ADS API (needs GALRAG_ADS_TOKEN)."""
    db = get_db()
    out: dict[str, Any] = {"text": text, "local": [], "ads": []}
    key = db.resolve(text.strip())
    if key:
        out["local"].append(_paper_brief(key))
    m = _AUTHOR_YEAR.match(text)
    if m:
        rows = db.search_meta(author=m.group(1), year=int(m.group(2)), limit=max_results)
        seen = {d["paper_key"] for d in out["local"]}
        out["local"] += [_paper_brief(r["paper_key"]) for r in rows if r["paper_key"] not in seen]
    from galrag.config import get_settings

    if not get_settings().ads_token:
        out["ads_error"] = "no ADS token configured (GALRAG_ADS_TOKEN); local results only"
        return out
    try:
        from galrag.ads import ADSClient

        with ADSClient() as c:
            cands = c.resolve_citation(text, max_results=max_results)
        for d in cands:
            d["in_corpus"] = db.resolve(d["bibcode"]) is not None if d.get("bibcode") else False
        out["ads"] = cands
    except Exception as e:  # noqa: BLE001
        out["ads_error"] = str(e)[:300]
    return out


@mcp.tool()
def references_of(ident: str, limit: int = 200) -> dict:
    """The paper's reference list (cited bibcodes from ADS), each flagged `in_corpus` with
    title/year/first_author when the cited paper is in the corpus."""
    try:
        key = _resolve(ident)
    except _NotFound as e:
        return _err(e)
    db = get_db()
    refs = db.references_of(key)
    out = []
    for b in refs[:limit]:
        k = db.resolve(b)
        if k:
            r = db.get_row(k)
            out.append({"bibcode": b, "in_corpus": True, "paper_key": k, "title": r["title"],
                        "year": r["year"], "first_author": r["first_author"],
                        "has_fulltext": r["status"] == Status.EMBEDDED})
        else:
            out.append({"bibcode": b, "in_corpus": False})
    return {"paper_key": key, "citation": _citation_for_key(key), "n_references": len(refs),
            "n_in_corpus": sum(r["in_corpus"] for r in out), "references": out}


@mcp.tool()
def citations_of(ident: str, limit: int = 200, local_only: bool = True) -> dict:
    """Papers citing `ident`: those in the local corpus (from their ADS reference lists),
    and, with local_only=false, the live ADS citation list (needs a token)."""
    db = get_db()
    key = db.resolve(ident)
    bibcode = (db.get_row(key)["bibcode"] if key else None) or ident.strip()
    local = [_paper_brief(k) for k in db.local_citers_of(bibcode)]
    local.sort(key=lambda d: -(d.get("citation_count") or 0))
    out: dict[str, Any] = {"bibcode": bibcode, "in_corpus": key is not None,
                           "n_local_citers": len(local), "local_citers": local[:limit]}
    if not local_only:
        from galrag.config import get_settings

        if not get_settings().ads_token:
            out["ads_error"] = "no ADS token configured"
        else:
            try:
                from galrag.ads import ADSClient, _slim

                with ADSClient() as c:
                    docs = c.citations_of(bibcode, rows=limit)
                out["ads_citers"] = [{**_slim(d), "in_corpus": db.resolve(d["bibcode"]) is not None}
                                     for d in docs]
            except Exception as e:  # noqa: BLE001
                out["ads_error"] = str(e)[:300]
    return out


@mcp.tool()
def get_bibtex(idents: list[str]) -> dict:
    """BibTeX entries keyed by bibcode (cached locally, fetched from ADS when missing)."""
    db = get_db()
    bibcodes, unresolved = [], []
    for i in idents:
        k = db.resolve(i)
        b = db.get_row(k)["bibcode"] if k else None
        if b:
            bibcodes.append(b)
        elif re.fullmatch(r"\d{4}\S{14}\S", i.strip()):
            bibcodes.append(i.strip())
        else:
            unresolved.append(i)
    out: dict[str, Any] = {}
    try:
        from galrag.ads import get_bibtex_cached

        out["bibtex"] = get_bibtex_cached(db, bibcodes)
    except Exception as e:  # noqa: BLE001  (no token / network): return what is cached
        out["bibtex"] = {b: t for b in bibcodes if (t := db.get_bibtex(b))}
        out["error"] = str(e)[:300]
    if unresolved:
        out["unresolved"] = unresolved
    return out


def _ingest_sync(ident: str, embed: bool) -> list[dict]:
    store = get_store()
    return service.ingest_ident(get_db(), store, store.embedder, ident, embed=embed)


@mcp.tool()
async def ingest_paper(ident: str, embed: bool = True) -> dict:
    """Add one paper to the corpus on demand: ADS metadata (any doctype/year), full text
    (arXiv HTML -> ar5iv -> PDF), parsing and embedding. `ident`: bibcode, arXiv id or
    DOI. Takes ~10 s to a few minutes. Returns the outcome of each stage."""
    import anyio.to_thread

    try:
        res = await anyio.to_thread.run_sync(_ingest_sync, ident, embed)
    except Exception as e:
        log.exception("ingest failed")
        return _err(e)
    return {"results": res}


def _update_sync(max_papers: int, process: bool) -> dict:
    store = get_store()
    return service.update_corpus(get_db(), store, lambda: store.embedder, process=process,
                                 max_papers=max_papers)


@mcp.tool()
async def update_corpus(max_papers: int = 200, process: bool = True) -> dict:
    """Incremental ADS update (papers entered since the last harvest), then fetch, parse
    and embed up to `max_papers` new tier-A/B papers (process=true). Slow: several
    seconds per paper."""
    import anyio.to_thread

    try:
        return await anyio.to_thread.run_sync(_update_sync, max_papers, process)
    except Exception as e:
        log.exception("update failed")
        return _err(e)


@mcp.tool()
def corpus_stats() -> dict:
    """Corpus size per ingestion status and tier, vector-store size, last ADS harvest/update
    and the most common failure reasons."""
    return service.corpus_stats(get_db(), get_store(), disk=False)


# =========================================================================== resources
def _paper_markdown(key: str) -> str:
    p = get_paper(key)
    if "error" in p:
        return p["error"]
    lines = [f"# {p['title']}", "", f"**{p['citation']}**", ""]
    who = ", ".join(p["authors"][:8]) + (" et al." if p["n_authors"] > 8 else "")
    lines += [f"- Authors: {who}", f"- Year: {p['year']}  ", f"- Journal: {p.get('pub') or ''}",
              f"- bibcode: {p['bibcode']}  arXiv: {p['arxiv_id']}  DOI: {p.get('doi')}",
              f"- Citations: {p['citation_count']}  refereed: {p['refereed']}",
              f"- Full text: {p['fulltext_available']} (source {p['source']}, status {p['status']})",
              f"- ADS: {p['links']['ads']}", "", "## Abstract", "", p.get("abstract") or "(none)", ""]
    if p.get("sections"):
        lines += ["## Sections", ""]
        for s in p["sections"]:
            lines.append(f"{'  ' * (s['level'] - 1)}- `{s['section_id']}` {s['title']} "
                         f"({s['n_paragraphs']} paragraphs)")
    return "\n".join(lines)


@mcp.resource("paper://{ident}", mime_type="text/markdown")
def paper_resource(ident: str) -> str:
    """Metadata, abstract and section outline of a paper (markdown)."""
    return _paper_markdown(ident)


@mcp.resource("paper://{ident}/section/{section_id}", mime_type="text/markdown")
def section_resource(ident: str, section_id: str) -> str:
    """Full text of one section (markdown, paragraphs prefixed by their ids)."""
    r = read_section(ident, section_id, max_chars=200_000)
    if "error" in r:
        return r["error"]
    body = "\n\n".join(f"[{p['para_id']}] {p['text']}" for p in r["paragraphs"])
    return f"## {r['section_title']}\n\n_{r['citation']}_\n\n{body}"


# =========================================================================== prompts
@mcp.prompt()
def literature_answer(question: str) -> str:
    """Answer a research question from the corpus with verified, bibcode-cited evidence."""
    return f"""Answer the following question using the galrag literature corpus.

Question: {question}

Procedure:
1. Call search_passages with one or more focused queries (try both a natural-language
   query and an "fts" query with key terms / object names). Use year or refereed filters
   if relevant; use search_papers to find whole papers.
2. For promising hits, read context with read_paragraphs (para_ids or offsets) or read_section.
3. Before quoting any text verbatim, call get_quote(paper, text) and only use quotation
   marks if it returns exact=true; otherwise paraphrase.
4. Cite every statement as "Author et al. (Year), bibcode" using the `citation` field.
5. Distinguish full-text evidence from abstract-only evidence, note disagreements between
   papers, and state clearly what the corpus does not cover or where you are uncertain.
"""


@mcp.prompt()
def check_citation(claim: str, bibcode: str) -> str:
    """Check whether a given paper actually supports a claim attributed to it."""
    return f"""Check whether the paper {bibcode} supports this claim:

"{claim}"

Procedure:
1. get_paper("{bibcode}") for metadata and the section outline (ingest_paper if missing).
2. verify_claim(claim, paper_keys=["{bibcode}"]) and get_quote("{bibcode}", <key phrase>) to
   find the closest passages; read their context with read_paragraphs.
3. Answer one of: SUPPORTED (quote the verbatim passage, with section and offsets),
   PARTIALLY SUPPORTED, NOT FOUND, or CONTRADICTED, and explain briefly.
4. If the claim is about something the paper cites, use citation_contexts to see what
   the paper says about the cited work.
"""


# =========================================================================== entry point
def main() -> None:
    """Console entry point ``galrag-mcp``: stdio MCP server."""
    service.setup_logging(logging.INFO, console="plain", console_level=logging.WARNING)
    log.info("starting galrag MCP server (stdio)")
    mcp.run()


if __name__ == "__main__":
    main()
