"""Chunker offset contract + LanceDB store round trip (real embedding model)."""

from __future__ import annotations

import pytest

from galrag.chunk import chunk_abstract_only, chunk_document, count_tokens, split_sentences
from galrag.config import Settings
from galrag.models import Document, PaperMeta, Paragraph, Section


def build_doc(key: str, title: str, abstract: str, sections: list[tuple[str, str, int, list]],
              source="arxiv_html") -> Document:
    """sections: (id, title, level, [(kind, text), ...]); abstract becomes section 'abs'."""
    parts: list[str] = []
    pos = 0
    secs: list[Section] = []
    all_secs = [("abs", "Abstract", 1, [("abstract", abstract)])] + sections
    for sid, stitle, lvl, paras in all_secs:
        sec = Section(id=sid, title=stitle, level=lvl)
        for i, (kind, text) in enumerate(paras, 1):
            if parts:
                pos += 2
            sec.paragraphs.append(Paragraph(id=f"{sid}.P{i}", kind=kind, text=text,
                                            char_start=pos, char_end=pos + len(text), page=1))
            parts.append(text)
            pos += len(text)
        secs.append(sec)
    ft = "\n\n".join(parts)
    return Document(paper_key=key, source=source, title=title, abstract=abstract, sections=secs,
                    full_text=ft)


LONG = " ".join(
    f"Sentence number {i} reports that Smith et al. (2020) found z = {i}.5 in Fig. 3, i.e. a bright "
    f"source with [O III] emission, e.g. at 5007 A." for i in range(60))
PARA = ("We observed the field with NIRCam in several filters and reduced the data with the "
        "standard pipeline, obtaining images reaching 30 AB mag depth. ") * 4


def make_docs():
    d1 = build_doc(
        "2023A&A...1..1A", "Bright galaxies at z>10", "We report GN-z11 like sources at cosmic dawn.",
        [("S1", "Introduction", 1, [("body", PARA)] * 6),
         ("S2", "Observations", 1, [("body", PARA)]),
         ("S2.SS1", "NIRCam imaging", 2, [("body", PARA), ("caption", "Figure 1: " + PARA[:300]),
                                          ("caption", "Fig. 2: tiny."), ("body", LONG)]),
         ("S3", "Results", 1, [("footnote", "Footnote about the cosmology adopted here " * 3),
                               ("body", "Short para."), ("list", "- item one\n- item two")])])
    d2 = build_doc("2020ApJ...2..2B", "Dust in early galaxies", "ALMA constraints on dust at z~7.",
                   [("S1", "Introduction", 1, [("body", "The photon escape fraction governs "
                                                "reionization. " * 20)])])
    d3 = build_doc("arXiv:2405.00003", "Quasar luminosity function", "A study of quasars.",
                   [("S1", "Intro", 1, [("body", "The quasar luminosity function is steep. " * 25)])])
    return d1, d2, d3


METAS = {
    "2023A&A...1..1A": PaperMeta(paper_key="2023A&A...1..1A", bibcode="2023A&A...1..1A", year=2023,
                                 first_author="Alpha, A.", refereed=True, citation_count=10,
                                 title="Bright galaxies at z>10", abstract="x"),
    "2020ApJ...2..2B": PaperMeta(paper_key="2020ApJ...2..2B", year=2020, first_author="Beta, B.",
                                 title="Dust in early galaxies"),
    "arXiv:2405.00003": PaperMeta(paper_key="arXiv:2405.00003", arxiv_id="2405.00003", year=2024,
                                  first_author="Gamma, G.", title="Quasar luminosity function"),
}


def test_sentence_splitter():
    t = "Smith et al. (2020) used Fig. 3, e.g. this one. We find z = 7.5 in JWST. J. Smith agrees."
    s = [t[a:b] for a, b in split_sentences(t)]
    assert s == ["Smith et al. (2020) used Fig. 3, e.g. this one.", "We find z = 7.5 in JWST.",
                 "J. Smith agrees."]


def test_offsets_contract_and_structure():
    cfg = Settings(chunk_target_tokens=120, chunk_max_tokens=200, chunk_overlap_paragraphs=1)
    for d in make_docs():
        chunks = chunk_document(d, METAS[d.paper_key], cfg)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))
        assert all(c.chunk_id == f"{d.paper_key}#{c.ordinal:04d}" for c in chunks)
        for c in chunks:
            if c.kind == "abstract":
                continue
            assert c.char_start >= 0
            assert d.full_text[c.char_start:c.char_end] == c.text, c.chunk_id
            assert c.n_tokens <= cfg.chunk_max_tokens + 5
            assert c.year == METAS[d.paper_key].year and c.source == d.source
        assert chunks[0].kind == "abstract"
        assert chunks[0].text == f"{d.title}\n\n{d.abstract}"
        # abstract is in full_text as paragraph -> real offsets
        assert d.full_text[chunks[0].char_start:chunks[0].char_end] == d.abstract
        assert sum(c.kind == "abstract" for c in chunks) == 1


def test_doc1_details():
    cfg = Settings(chunk_target_tokens=120, chunk_max_tokens=200, chunk_overlap_paragraphs=1)
    d1 = make_docs()[0]
    ch = chunk_document(d1, METAS[d1.paper_key], cfg)
    paths = {c.section_path for c in ch}
    assert "2 Observations > 2.1 NIRCam imaging" in paths
    assert any(c.kind == "caption" for c in ch)
    # tiny caption ("Fig. 2: tiny.") is merged, not its own chunk
    assert not any(c.text.strip() == "Fig. 2: tiny." for c in ch)
    # the long paragraph is split into several sub-chunks of the same para id
    long_id = "S2.SS1.P4"
    subs = [c for c in ch if c.para_ids == [long_id]]
    assert len(subs) >= 3
    # overlap: the intro paragraphs repeat across consecutive chunks
    cfg2 = Settings(chunk_target_tokens=300, chunk_max_tokens=400, chunk_overlap_paragraphs=1)
    intro = [c for c in chunk_document(d1, METAS[d1.paper_key], cfg2) if c.section_id == "S1"]
    assert len(intro) >= 2 and set(intro[0].para_ids) & set(intro[1].para_ids)
    assert any(c.kind == "footnote" for c in ch)


def test_abstract_not_in_fulltext_and_abstract_only():
    d = Document(paper_key="k", source="arxiv_pdf", title="T", abstract="Abs text.",
                 sections=[Section(id="S1", title="Intro", paragraphs=[
                     Paragraph(id="S1.P1", text="Body text here.", char_start=0, char_end=15)])],
                 full_text="Body text here.")
    ch = chunk_document(d, None)
    assert ch[0].kind == "abstract" and (ch[0].char_start, ch[0].char_end) == (-1, -1)
    assert ch[1].text == "Body text here."
    m = PaperMeta(paper_key="k2", title="T", abstract="A.", year=2021, first_author="X")
    one = chunk_abstract_only(m)
    assert len(one) == 1 and one[0].source == "abstract_only" and one[0].text == "T\n\nA."
    assert chunk_abstract_only(PaperMeta(paper_key="k3")) == []


def test_count_tokens_positive():
    assert count_tokens("hello world, JWST sees galaxies") > 3


# ------------------------------------------------------------------ store

@pytest.fixture(scope="module")
def embedder():
    try:
        from galrag.embed import Embedder

        e = Embedder()
        e.embed_passages(["test"])
        return e
    except Exception as exc:  # pragma: no cover  # noqa: BLE001
        pytest.skip(f"embedding model unavailable: {exc}")


@pytest.fixture()
def store(tmp_path, embedder):
    from galrag.store import VectorStore

    cfg = Settings(data_dir=tmp_path)
    return VectorStore(cfg, embedder=embedder)


def ingest(store, embedder, docs):
    cfg = Settings(chunk_target_tokens=120, chunk_max_tokens=200)
    all_chunks = []
    for d in docs:
        all_chunks += chunk_document(d, METAS[d.paper_key], cfg)
    vecs = embedder.embed_passages([c.text for c in all_chunks])
    store.upsert_chunks(all_chunks, vecs)
    return all_chunks


def test_store_roundtrip(store, embedder):
    assert store.count() == 0 and store.search("anything") == []
    d1, d2, d3 = make_docs()
    chunks = ingest(store, embedder, [d1, d2, d3])
    n = len(chunks)
    assert store.count() == n and store.has_paper(d2.paper_key)
    assert not store.has_paper("nope")

    for mode in ("vector", "fts", "hybrid"):
        hits = store.search("quasar luminosity function is steep", k=3, mode=mode, rerank=False)
        assert hits and hits[0].chunk.paper_key == d3.paper_key, mode
        assert hits[0].rank == 1 and hits[0].scores
        c = hits[0].chunk
        assert c.para_ids is not None and c.chunk_id.startswith(d3.paper_key)
    h = store.search("photon escape fraction governs reionization", k=3, mode="hybrid", rerank=False)
    assert h[0].chunk.paper_key == d2.paper_key and {"vector", "fts", "rrf"} <= set(h[0].scores)

    # exact identifier phrase + weird characters never raise
    assert store.search("GN-z11", mode="fts", rerank=False)[0].chunk.paper_key == d1.paper_key
    store.search('[O III] 88 um (z ~ 7.5) "x" AND a:b', mode="hybrid", rerank=False)

    # filters
    q = "galaxies"
    assert {h.chunk.paper_key for h in store.search(q, k=50, mode="vector", rerank=False,
                                                    filters={"year_from": 2023, "year_to": 2023})} == {d1.paper_key}
    assert {h.chunk.paper_key for h in store.search(q, k=50, mode="hybrid", rerank=False,
                                                    filters={"year_to": 2020})} == {d2.paper_key}
    got = store.search("galaxy", k=50, mode="fts", rerank=False, filters={"paper_keys": [d2.paper_key]})
    assert {h.chunk.paper_key for h in got} <= {d2.paper_key}
    assert {h.chunk.paper_key for h in store.search(q, k=50, mode="vector", rerank=False,
                                                    filters={"first_author": "gam"})} == {d3.paper_key}
    assert {h.chunk.paper_key for h in store.search(q, k=50, mode="vector", rerank=False,
                                                    filters={"refereed_only": True})} == {d1.paper_key}
    assert {h.chunk.kind for h in store.search(q, k=50, mode="vector", rerank=False,
                                               filters={"kinds": ["abstract"]})} == {"abstract"}
    assert store.search(q, mode="vector", rerank=False, filters={"paper_keys": ["it's"]}) == []

    # idempotent re-ingest
    ingest(store, embedder, [d1])
    assert store.count() == n
    # rows appended after the FTS index exists are searchable without re-indexing
    extra = build_doc("2025X", "New", "abs", [("S1", "I", 1, [("body", "zyxwv unique token here " * 5)])])
    METAS["2025X"] = PaperMeta(paper_key="2025X", year=2025, first_author="Z")
    ingest(store, embedder, [extra])
    assert store.search("zyxwv", mode="fts", rerank=False)[0].chunk.paper_key == "2025X"
    store.refresh_indexes()
    assert store.search("zyxwv", mode="fts", rerank=False)[0].chunk.paper_key == "2025X"
    store.create_indexes()

    store.delete_paper(d2.paper_key)
    assert not store.has_paper(d2.paper_key)
    assert store.count() == n + len(chunk_document(extra, METAS["2025X"])) - sum(
        c.paper_key == d2.paper_key for c in chunks)


def test_papers_table(store, embedder):
    assert store.search_papers("dust") == []
    items = [METAS["2020ApJ...2..2B"], METAS["arXiv:2405.00003"]]
    vecs = embedder.embed_passages(["Dust in early galaxies. ALMA dust", "Quasar luminosity function"])
    store.upsert_papers(items, vecs)
    store.upsert_papers(items, vecs)
    r = store.search_papers("dust emission in galaxies", k=5)
    assert len(r) == 2 and r[0]["paper_key"] == "2020ApJ...2..2B"
    assert [x["paper_key"] for x in store.search_papers("dust", filters={"year_from": 2024})] == [
        "arXiv:2405.00003"]


def test_rerank(store, embedder):
    try:
        from galrag.embed import Reranker

        rr = Reranker()
        rr.rerank("q", ["a"])
    except Exception as exc:  # pragma: no cover  # noqa: BLE001
        pytest.skip(f"reranker unavailable: {exc}")
    store._reranker = rr
    d1, d2, d3 = make_docs()
    ingest(store, embedder, [d1, d2, d3])
    h = store.search("quasar luminosity function", k=2, rerank=True)
    assert h[0].chunk.paper_key == d3.paper_key and "rerank" in h[0].scores
