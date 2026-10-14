"""MCP server tools on a tiny temporary corpus (real parser, chunker and embedding model).

The corpus lives in a temporary GALRAG_DATA_DIR (SQLite incl. documents, LanceDB);
GALRAG_WORK_DIR / GALRAG_LANCEDB_DIR are cleared so the production store is never touched.  The embedding model is read from the real
HF cache (``<real data dir>/models``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from galrag import config
from galrag.models import PaperMeta

FIX = Path(__file__).parent / "fixtures"
K1 = "2024ApJ...999L...1A"      # arxiv_2405.03781.html
K2 = "2016MNRAS.999.0002B"      # ar5iv_1601.00001.html
K3 = "2020A&A...999A...3C"      # abstract only (no full text)
CITED = "2005ApJ...619..697A"   # synthetic bibcode for one bibliography entry of K1


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    real_models = config.Settings().models_dir          # before overriding the data dir
    os.environ.setdefault("HF_HOME", str(real_models))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tmp = tmp_path_factory.mktemp("galrag_data")
    names = ("GALRAG_DATA_DIR", "GALRAG_WORK_DIR", "GALRAG_LANCEDB_DIR")
    old = {n: os.environ.get(n) for n in names}
    os.environ["GALRAG_DATA_DIR"] = str(tmp)
    os.environ["GALRAG_WORK_DIR"] = ""           # empty = unset (overrides a .env value)
    os.environ["GALRAG_LANCEDB_DIR"] = ""
    config.get_settings.cache_clear()
    try:
        yield from _build(tmp)
    finally:
        for n, v in old.items():
            if v is None:
                os.environ.pop(n, None)
            else:
                os.environ[n] = v
        config.get_settings.cache_clear()
        from galrag import server

        server.configure(None, None)


def _build(tmp: Path):
    from galrag import server
    from galrag.db import Database
    from galrag.parse import save_document
    from galrag.parse.html_latexml import parse_latexml_html
    from galrag.pipeline import embed_documents
    from galrag.service import embed_paper_vectors
    from galrag.store import VectorStore

    try:
        from galrag.embed import Embedder

        emb = Embedder(threads=8)
        emb.embed_passages(["warm up"])
    except Exception as exc:  # pragma: no cover  # noqa: BLE001
        pytest.skip(f"embedding model unavailable: {exc}")

    d1 = parse_latexml_html((FIX / "arxiv_2405.03781.html").read_text(), K1, "arxiv_html",
                            "https://arxiv.org/html/2405.03781")
    d2 = parse_latexml_html((FIX / "ar5iv_1601.00001.html").read_text(), K2, "ar5iv",
                            "https://ar5iv.labs.arxiv.org/html/1601.00001")
    # give one cited bibliography entry of d1 a bibcode (as resolve_bibliography would)
    cited_keys = {k for _, p in d1.iter_paragraphs() for c in p.cites for k in c.bib_keys}
    bib = next(b for b in d1.bibliography if b.key in cited_keys and b.first_author and b.year)
    bib.bibcode = CITED

    metas = [
        PaperMeta(paper_key=K1, bibcode=K1, arxiv_id="2405.03781", title=d1.title,
                  authors=["Kim, Alpha", "Lee, Beta", "Park, Gamma"], first_author="Kim, Alpha",
                  year=2024, abstract=d1.abstract, refereed=True, citation_count=12,
                  references=[CITED, K2, "1999ApJ...000..000Z"], tier="A"),
        PaperMeta(paper_key=K2, bibcode=K2, arxiv_id="1601.00001", title=d2.title,
                  authors=["Bergfield, Delta", "Ratner, Eps"], first_author="Bergfield, Delta",
                  year=2016, abstract=d2.abstract, refereed=True, citation_count=40, tier="A"),
        PaperMeta(paper_key=K3, bibcode=K3, title="Dust attenuation in Lyman break galaxies at z~4",
                  authors=["Chen, Zeta"], first_author="Chen, Zeta", year=2020,
                  abstract="We measure the UV continuum slopes of Lyman break galaxies at z~4 and "
                           "find that dust attenuation increases with stellar mass.",
                  refereed=True, citation_count=5, references=[K1], tier="C"),
    ]
    db = Database(tmp / "galrag.sqlite")
    db.upsert_papers(metas)
    for d in (d1, d2):
        save_document(d, db)
    for m in metas[:2]:
        db.set_status(m.paper_key, "parsed", source="arxiv_html" if m.paper_key == K1 else "ar5iv")
    store = VectorStore(embedder=emb, uri=str(tmp / "lancedb"))
    n = embed_documents(db, store, emb, [(metas[0], d1), (metas[1], d2)])
    assert n > 0
    pv = embed_paper_vectors(db, store, emb, [m.paper_key for m in metas])
    assert pv == {"papers": 3, "abstract_chunks": 1}
    store.create_indexes()
    server.configure(db, store)
    yield {"db": db, "store": store, "docs": {K1: d1, K2: d2}, "metas": metas, "tmp": tmp,
           "cited_bib": bib}


# ---------------------------------------------------------------------- tools
def test_search_passages_provenance(corpus):
    from galrag import server

    hits = server.search_passages("overdensity of Lyman break galaxies around a quasar", k=8,
                                  rerank=False)
    assert isinstance(hits, list) and hits
    assert hits[0]["paper_key"] == K1
    need = {"rank", "score", "chunk_id", "paper_key", "bibcode", "arxiv_id", "citation", "title",
            "section_path", "text", "char_start", "char_end", "page_start", "kind", "source"}
    n_checked = 0
    for h in hits:
        assert need <= set(h)
        assert h["chunk_id"].startswith(h["paper_key"] + "#")
        doc = corpus["docs"].get(h["paper_key"])
        if h["kind"] != "abstract" and doc is not None:
            assert h["verbatim_span"]
            assert doc.full_text[h["char_start"]:h["char_end"]] == h["text"]
            n_checked += 1
    assert n_checked
    assert hits[0]["citation"].startswith("Kim et al. (2024), " + K1)

    # filters and identifier resolution in paper_keys
    only2 = server.search_passages("transport junction probe", k=5, rerank=False,
                                   paper_keys=["arXiv:1601.00001"])
    assert only2 and {h["paper_key"] for h in only2} == {K2}
    old = server.search_passages("galaxies", k=20, mode="fts", rerank=False, year_to=2020)
    assert old and all(h["paper_key"] in (K2, K3) for h in old)
    abs_hits = server.search_passages("dust attenuation UV slope", k=3, rerank=False,
                                      kinds=["abstract"])
    assert abs_hits[0]["paper_key"] == K3 and abs_hits[0]["source"] == "abstract_only"
    assert not abs_hits[0]["verbatim_span"]


def test_search_papers_and_find(corpus):
    from galrag import server

    r = server.search_papers("dust attenuation in Lyman break galaxies", k=3)
    assert r[0]["paper_key"] == K3 and r[0]["has_fulltext"] is False
    assert {"bibcode", "title", "authors_short", "year", "citation_count", "score"} <= set(r[0])
    assert any(x["paper_key"] == K1 and x["has_fulltext"] for x in r)
    f = server.find_papers(author="Kim", year=2024)
    assert [x["paper_key"] for x in f] == [K1]
    assert server.find_papers(title="molecular transport")[0]["paper_key"] == K2


def test_get_paper_and_read(corpus):
    from galrag import server

    d1 = corpus["docs"][K1]
    p = server.get_paper("arXiv:2405.03781")
    assert p["paper_key"] == K1 and p["status"] == "embedded" and p["fulltext_available"]
    assert p["n_chunks"] > 0 and p["links"]["ads"].endswith(f"/abs/{K1}/abstract")
    assert [s["section_id"] for s in p["sections"]] == [s.id for s in d1.sections]
    p3 = server.get_paper(K3)
    assert "error" not in p and p3["fulltext_available"] is False and "sections" not in p3 and p3["abstract"]
    assert "error" in server.get_paper("2099XXX...1....1Z")

    sec = server.read_section(K1, "S1")
    assert sec["section_id"] == "S1" and sec["paragraphs"]
    para = sec["paragraphs"][0]
    assert d1.full_text[para["char_start"]:para["char_end"]] == para["text"]
    assert server.read_section(K1, "Introduction")["section_id"] == "S1"
    assert "error" in server.read_section(K3, "S1")

    r = server.read_paragraphs(K1, para_ids=[para["para_id"], "nope"])
    assert r["paragraphs"][0]["text"] == para["text"] and r["missing_para_ids"] == ["nope"]
    a, b = para["char_start"] + 10, para["char_start"] + 80
    span = server.read_paragraphs(K1, char_start=a, char_end=b, context_chars=20)
    assert span["text"] == d1.full_text[a:b]
    assert span["context_before"] == d1.full_text[a - 20:a]
    assert span["para_id"] == para["para_id"]

    rows = server.chunks_of_paper(corpus["store"], K1)
    assert [r["ordinal"] for r in rows] == list(range(len(rows))) and len(rows) == p["n_chunks"]
    assert server.chunks_of_paper(corpus["store"], "it's") == []


def test_get_quote_exact_and_fuzzy(corpus):
    from galrag import server

    d1 = corpus["docs"][K1]
    para = next(p for _, p in d1.iter_paragraphs() if p.kind == "body" and len(p.text) > 300)
    exact = para.text[50:200]
    q = server.get_quote(K1, "  " + exact.upper().replace(" ", "\n ") + " ")
    assert q["found"] and q["exact"] and q["similarity"] == 100.0
    assert q["quote"] == exact and d1.full_text[q["char_start"]:q["char_end"]] == exact
    assert q["para_id"] == para.id and q["citation"].endswith(K1)

    from galrag.quotes import split_sentences

    sents = [para.text[a:b] for a, b in split_sentences(para.text) if b - a > 60]
    target = sents[0]
    words = target.split()
    paraphrase = " ".join(words[:3] + ["notably"] + words[3:-2] + [words[-1]])
    f = server.get_quote(K1, paraphrase)
    assert f["found"] and not f["exact"] and 60 <= f["similarity"] < 100
    assert target in f["quote"] or f["quote"] in para.text
    miss = server.get_quote(K1, "Completely unrelated sentence about cooking pasta al dente.",
                            min_similarity=90)
    assert not miss["found"]


def test_verify_claim(corpus):
    from galrag import server

    d1 = corpus["docs"][K1]
    para = next(p for _, p in d1.iter_paragraphs() if p.kind == "body" and len(p.text) > 300)
    claim = para.text[:150]
    r = server.verify_claim(claim, k=5, rerank=False)
    top = r["hits"][0]
    assert top["paper_key"] == K1
    bm = top["best_match_in_paper"]
    assert bm["exact"] and bm["offsets_in"] == "full_text"
    assert d1.full_text[bm["char_start"]:bm["char_end"]] == bm["quote"]


def test_references_citations_contexts(corpus):
    from galrag import server

    r = server.references_of(K1)
    assert r["n_references"] == 3 and r["n_in_corpus"] == 1
    ref = {x["bibcode"]: x for x in r["references"]}
    assert ref[K2]["in_corpus"] and ref[K2]["has_fulltext"] and ref[K2]["year"] == 2016
    assert not ref[CITED]["in_corpus"]

    c = server.citations_of(K1)
    assert [x["paper_key"] for x in c["local_citers"]] == [K3]

    ctx = server.citation_contexts(K1, cited_ident=CITED)
    assert ctx["in_ads_reference_list"] is True and ctx["n_contexts"] >= 1
    d1 = corpus["docs"][K1]
    for x in ctx["contexts"]:
        assert d1.full_text[x["char_start"]:x["char_end"]] == x["sentence"]
    bib = corpus["cited_bib"]
    ctx2 = server.citation_contexts(K1, cited_first_author=bib.first_author, cited_year=bib.year)
    assert ctx2["n_contexts"] >= 1
    assert "error" in server.citation_contexts(K1)


def test_resolve_citation_local(corpus, monkeypatch):
    from galrag import server

    monkeypatch.setattr(config.get_settings(), "ads_token", "")
    r = server.resolve_citation("Kim et al. 2024")
    assert [x["paper_key"] for x in r["local"]] == [K1] and "ads_error" in r
    assert server.resolve_citation("arXiv:1601.00001")["local"][0]["paper_key"] == K2


def test_corpus_stats_and_resources(corpus):
    from galrag import server

    s = server.corpus_stats()
    assert s["papers_total"] == 3
    assert s["by_status"] == {"embedded": 2, "metadata": 1}
    assert s["vector_papers"] == 3 and s["vector_chunks"] == corpus["store"].count()
    assert s["documents"] == 2
    md = server.paper_resource(K1)
    assert md.startswith("# ") and K1 in md and "`S1`" in md
    sec = server.section_resource(K1, "S1")
    assert sec.startswith("## ") and "[S1.P1]" in sec
    assert "get_quote" in server.literature_answer("What is X?")
    assert K1 in server.check_citation("claim", K1)


async def test_tool_registry():
    from galrag import server

    names = {t.name for t in await server.mcp.list_tools()}
    assert {"search_passages", "search_papers", "find_papers", "get_paper", "read_section",
            "read_paragraphs", "get_quote", "verify_claim", "citation_contexts", "resolve_citation",
            "references_of", "citations_of", "get_bibtex", "ingest_paper", "update_corpus",
            "corpus_stats"} <= names


async def test_stdio_smoke(tmp_path):
    """Spawn the real stdio server and list its tools (no model is loaded)."""
    import anyio
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "galrag.server"],
        env={"GALRAG_DATA_DIR": str(tmp_path), "GALRAG_WORK_DIR": "", "GALRAG_LANCEDB_DIR": "",
             "GALRAG_ADS_TOKEN": ""},
        cwd=str(tmp_path))
    with anyio.fail_after(600):
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert {"search_passages", "get_quote", "corpus_stats"} <= names
            res = await session.call_tool("corpus_stats", {})
            assert not res.is_error
            import json

            assert json.loads(res.content[0].text)["papers_total"] == 0
