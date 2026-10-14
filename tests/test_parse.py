"""Parser tests on fixture HTML/PDF, the offset contract and (optionally) live fetching."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from galrag.models import BibEntry, CiteMarker, Document, Paragraph, Section
from galrag.parse.common import (
    HARD_BREAK,
    build_full_text,
    check_offsets,
    dehyphenate,
    map_span,
    normalize_text,
    normalize_with_map,
    sentence_spans,
    split_sentences,
)

FIX = Path(__file__).parent / "fixtures"


# ------------------------------------------------------------------ helpers
def assert_offsets(doc: Document) -> None:
    ft = doc.full_text
    n = 0
    for _, p in doc.iter_paragraphs():
        assert p.text, p.id
        assert ft[p.char_start:p.char_end] == p.text, p.id
        for c in p.cites:
            assert p.char_start <= c.char_start < c.char_end <= p.char_end
            assert ft[c.char_start:c.char_end] == c.text
            assert c.bib_keys
            n += 1
    ids = [p.id for _, p in doc.iter_paragraphs()]
    assert len(ids) == len(set(ids)), "paragraph ids must be unique"
    # JSON round trip keeps the contract
    doc2 = Document.model_validate_json(doc.model_dump_json())
    check_offsets(doc2)


def stats(doc: Document) -> dict:
    paras = [p for _, p in doc.iter_paragraphs()]
    return {
        "sections": len([s for s in doc.sections if s.id not in ("abs", "front")]),
        "paragraphs": len(paras),
        "cites": sum(len(p.cites) for p in paras),
        "bib": len(doc.bibliography),
        "bib_year": sum(b.year is not None for b in doc.bibliography),
    }


# --------------------------------------------------------------- common.py
def test_normalize_whitespace_and_unicode():
    raw = "  The z = 6 gal­axies   are 𝑧-dropouts:  ﬁne  α  "
    assert normalize_text(raw) == "The z = 6 galaxies are z-dropouts: fine α"
    assert normalize_text(f"item one {HARD_BREAK} item two") == "item one\nitem two"


def test_normalize_map_spans():
    raw = "  see  (Bunker   et al.,  2023)  for"
    text, m = normalize_with_map(raw)
    a, b = raw.index("(Bunker"), raw.index("2023)") + 5
    s, e = map_span(text, m, a, b)
    assert text[s:e] == "(Bunker et al., 2023)"
    s, e = map_span(text, m, a - 2, b + 2)        # whitespace at the edges is trimmed
    assert text[s:e] == "(Bunker et al., 2023)"


def test_sentence_split():
    t = ("Bunker et al. (2023) found z = 10.6 for GN-z11. See Fig. 3 and e.g. Eq. 2 for details. "
         "J. Smith disagreed! The end.")
    sents = split_sentences(t)
    assert sents == ["Bunker et al. (2023) found z = 10.6 for GN-z11.",
                     "See Fig. 3 and e.g. Eq. 2 for details.",
                     "J. Smith disagreed!", "The end."]
    for (a, b), s in zip(sentence_spans(t), sents):
        assert t[a:b] == s


def test_dehyphenate():
    assert dehyphenate("gal-\naxies at high red-\nshift, Lyman-\nAlpha") == \
        "galaxies at high redshift, Lyman-Alpha"


def test_build_full_text_contract():
    secs = [
        Section(id="S1", title="1 Intro", paragraphs=[
            Paragraph(id="S1.P1", text="Alpha (A et al. 2020) beta.", char_start=0, char_end=27,
                      cites=[CiteMarker(bib_keys=["b1"], char_start=6, char_end=21, text="(A et al. 2020)")]),
            Paragraph(id="S1.P2", text="Gamma.", char_start=0, char_end=6),
        ]),
        Section(id="S2", title="2 Data", paragraphs=[
            Paragraph(id="S2.P1", text="Delta [1].", char_start=0, char_end=10,
                      cites=[CiteMarker(bib_keys=["b2"], char_start=6, char_end=9, text="[1]")]),
        ]),
    ]
    ft = build_full_text(secs)
    assert ft == "Alpha (A et al. 2020) beta.\n\nGamma.\n\nDelta [1]."
    ft2 = build_full_text(secs)                    # idempotent
    assert ft2 == ft
    doc = Document(paper_key="k", source="arxiv_html", sections=secs, full_text=ft)
    check_offsets(doc)
    assert doc.span(secs[1].paragraphs[0].cites[0].char_start, secs[1].paragraphs[0].cites[0].char_end) == "[1]"


# ------------------------------------------------------------ LaTeXML HTML
@pytest.fixture(scope="module")
def arxiv_doc() -> Document:
    from galrag.parse.html_latexml import parse_latexml_html

    html = (FIX / "arxiv_2405.03781.html").read_text(encoding="utf-8")
    return parse_latexml_html(html, "arXiv:2405.03781", "arxiv_html", "https://arxiv.org/html/2405.03781")


@pytest.fixture(scope="module")
def ar5iv_doc() -> Document:
    from galrag.parse.html_latexml import parse_latexml_html

    html = (FIX / "ar5iv_1601.00001.html").read_text(encoding="utf-8")
    return parse_latexml_html(html, "arXiv:1601.00001", "ar5iv", "https://ar5iv.labs.arxiv.org/html/1601.00001")


def test_arxiv_html_structure(arxiv_doc: Document):
    d = arxiv_doc
    assert_offsets(d)
    st = stats(d)
    assert st["sections"] > 5
    assert st["paragraphs"] > 40
    assert st["cites"] > 50
    assert st["bib"] > 30
    assert st["bib_year"] > 0.5 * st["bib"]
    assert d.parser_version == "latexml-1"
    assert d.title.startswith("Large Scale Overdensity of Lyman Break Galaxies")
    assert "$z=6.3$" in d.title
    assert d.abstract.startswith("We study the environment of the $z=6.33$ ultraluminous quasar")
    assert "short_text" not in d.quality_flags and "no_sections" not in d.quality_flags
    ids = {s.id for s in d.sections}
    assert {"abs", "S1", "S2.SS1", "S3.SS1", "A1"} <= ids
    s1 = next(s for s in d.sections if s.id == "S1")
    assert s1.title == "I Introduction" and s1.level == 1
    assert next(s for s in d.sections if s.id == "S2.SS1").level == 2
    # site chrome is not in the text
    assert "Report Issue" not in d.full_text and "Report issue" not in d.full_text
    assert "Why HTML?" not in d.full_text


def test_arxiv_html_content(arxiv_doc: Document):
    d = arxiv_doc
    p = d.paragraph("S1.P1")
    assert p.text.startswith("Over the last two decades of quasar research")
    assert "$\\geq 10^{9}\\,M_{\\odot}$" in p.text          # math kept as TeX
    c = p.cites[0]
    assert c.text == "[83, 93, 7, 46, 94, 87, 95, 33]"
    assert c.bib_keys[:3] == ["bib.bib83", "bib.bib93", "bib.bib7"]
    # footnotes are moved out of the running text into their own paragraphs
    p3 = d.paragraph("S2.SS1.P3")
    assert "Telescope Access Program of NAOC during the 2017B semester" in p3.text
    fn = [q for _, q in d.iter_paragraphs() if q.kind == "footnote"]
    assert any(q.text == "https://tap.china-vo.org" for q in fn)
    caps = {q.id: q for _, q in d.iter_paragraphs() if q.kind == "caption"}
    assert "S3.F1.cap" in caps and caps["S3.F1.cap"].text.startswith("Figure 1: In black, transmission curves")
    assert "S2.T1.cap" in caps
    assert any(c.bib_keys == ["bib.bib45"] for c in caps["S3.F1.cap"].cites)
    # every cite key exists in the bibliography
    keys = {b.key for b in d.bibliography}
    assert all(k in keys for _, q in d.iter_paragraphs() for c in q.cites for k in c.bib_keys)
    # display equations are inline TeX, not dropped
    assert "$i-z>1.5$" in d.paragraph("S3.SS1.P1").text
    # bibliography parsing
    b1 = d.bibliography[0]
    assert b1.key == "bib.bib1" and b1.first_author == "Adelberger" and b1.year == 2005
    assert b1.journal == "ApJ" and b1.doi == "10.1086/426580"
    assert sum(b.doi is not None for b in d.bibliography) > 0.5 * len(d.bibliography)


def test_ar5iv(ar5iv_doc: Document):
    d = ar5iv_doc
    assert_offsets(d)
    st = stats(d)
    assert st["sections"] >= 5
    assert st["paragraphs"] > 30
    assert st["cites"] > 30
    assert st["bib"] > 30 and st["bib_year"] > 0.5 * st["bib"]
    assert d.title.startswith("Inelastic effects in molecular transport junctions")
    assert d.abstract.startswith("We extend the Landauer-Büttiker probe formalism")
    p = d.paragraph("S1.P1")
    assert p.cites[0].bib_keys == ["bib.bib1"]
    assert "Report" not in d.full_text and "Copyright" not in d.full_text
    b = d.bibliography[0]
    assert b.first_author == "Hihath" and b.year == 2012


# ---------------------------------------------------------------------- PDF
@pytest.fixture(scope="module")
def pdf_doc() -> Document:
    from galrag.parse.pdf import parse_pdf

    return parse_pdf(FIX / "arxiv_1601.00001.pdf", "arXiv:1601.00001", "arxiv_pdf",
                     "https://arxiv.org/pdf/1601.00001")


def test_pdf(pdf_doc: Document):
    d = pdf_doc
    assert_offsets(d)
    assert d.n_pages and d.n_pages > 0
    assert d.parser_version == "pdf-1"
    paras = [p for _, p in d.iter_paragraphs()]
    assert len(paras) > 20
    assert all(p.page is not None and 1 <= p.page <= d.n_pages for p in paras)
    assert d.title.startswith("Inelastic effects in molecular transport junctions")
    assert d.abstract.startswith("We extend the Landauer-Büttiker probe formalism")
    titles = [s.title for s in d.sections]
    assert "I INTRODUCTION" in titles and any(t.startswith("V CONCLUSIONS") for t in titles)
    assert "pdf_no_headings" not in d.quality_flags
    # references are separated from the body
    assert len(d.bibliography) > 50
    assert d.bibliography[0].first_author == "Hihath" and d.bibliography[0].year == 2012
    body = [p for p in paras if p.kind == "body"]
    refish = re.compile(r"Phys\. Rev\. [BE] \d+, \d+ \(\d{4}\)")
    assert sum(1 for p in body if refish.search(p.text)) == 0
    # no arXiv side stamp, no page numbers, dehyphenated text
    assert "arXiv:1601.00001v1" not in d.full_text
    assert "molecule-based electronic systems" in d.full_text
    assert "opera- tion" not in d.full_text and "operation of molecule-based" in d.full_text
    # numeric superscript citations are linked
    assert sum(len(p.cites) for p in paras) > 20


# ----------------------------------------------------------- persistence / bib
def test_save_load_and_fs_safe(tmp_path, arxiv_doc):
    from galrag.db import Database
    from galrag.models import PaperMeta
    from galrag.parse import fs_safe, load_document, save_document

    assert fs_safe("astro-ph/0601001") == "astro-ph_0601001"
    assert fs_safe("arXiv:2405.03781") == "arXiv_2405.03781"
    db = Database(tmp_path / "galrag.sqlite")
    db.upsert_papers([PaperMeta(paper_key=arxiv_doc.paper_key, title="t")])
    save_document(arxiv_doc, db)
    d2 = load_document(arxiv_doc.paper_key, db)
    assert d2 is not None and d2.full_text == arxiv_doc.full_text
    save_document(arxiv_doc, db)                 # replace, not duplicate
    assert db.count_documents() == 1
    assert load_document("nope", db) is None
    assert not (tmp_path / "docs").exists()      # no per-document files any more
    db.close()


def test_parse_bibcode():
    from galrag.parse import parse_bibcode

    b = parse_bibcode("2005ApJ...619..697A")
    assert b["year"] == 2005 and b["bibstem"] == "ApJ" and b["volume"] == "619" and b["page"] == "697"
    assert b["initial"] == "A"
    assert parse_bibcode("2023arXiv230512345B")["arxiv_id"] == "2305.12345"
    assert parse_bibcode("2013A&A...558A..33A")["page"] == "33"


def test_resolve_bibliography(tmp_path, arxiv_doc):
    from galrag.db import Database
    from galrag.models import PaperMeta
    from galrag.parse import resolve_bibliography

    db = Database(tmp_path / "t.sqlite")
    refs = ["2005ApJ...619..697A", "2014MNRAS.439.2146C", "2017A&A...606A..23B", "2013ApJ...773..178B",
            "2016ApJS..227...11B", "2010ApJ...721.1680U", "1999ADIL...KK.....X"]
    meta = PaperMeta(paper_key="2024ApJ...968...16P", bibcode="2024ApJ...968...16P", arxiv_id="2405.03781",
                     references=refs)
    db.upsert_papers([meta])
    doc = arxiv_doc.model_copy(deep=True)
    doc.paper_key = meta.paper_key
    for b in doc.bibliography:
        b.bibcode = None
    n = resolve_bibliography(doc, db)
    got = {b.key: b.bibcode for b in doc.bibliography if b.bibcode}
    assert n == len(got) >= 6
    assert got["bib.bib1"] == "2005ApJ...619..697A"
    assert got["bib.bib25"] == "2014MNRAS.439.2146C"
    assert got["bib.bib9"] == "2017A&A...606A..23B"
    assert got["bib.bib7"] == "2016ApJS..227...11B"
    # Bañados 2013 ApJ 773, 178 (bib6) vs Bañados 2016 ApJS (bib7): no cross-matching
    assert got.get("bib.bib6") == "2013ApJ...773..178B"
    db.close()


def test_resolve_by_arxiv_id(tmp_path):
    from galrag.db import Database
    from galrag.models import PaperMeta
    from galrag.parse import resolve_bibliography

    db = Database(tmp_path / "t.sqlite")
    cited = PaperMeta(paper_key="2023Natur.616..266C", bibcode="2023Natur.616..266C", arxiv_id="2212.04480")
    citing = PaperMeta(paper_key="arXiv:2401.00001", arxiv_id="2401.00001", references=["2023arXiv230512345B"])
    db.upsert_papers([cited, citing])
    doc = Document(paper_key="arXiv:2401.00001", source="arxiv_html", bibliography=[
        BibEntry(key="b1", raw_text="Curtis-Lake, E., et al. 2023, arXiv:2212.04480", arxiv_id="2212.04480",
                 year=2023, first_author="Curtis-Lake"),
        BibEntry(key="b2", raw_text="Bunker, A. 2023, arXiv e-prints, arXiv:2305.12345", arxiv_id="2305.12345",
                 year=2023, first_author="Bunker"),
        BibEntry(key="b3", raw_text="Nobody, X. 1999, Obscure J., 1, 1", year=1999, first_author="Nobody"),
    ])
    assert resolve_bibliography(doc, db) == 2
    assert doc.bibliography[0].bibcode == "2023Natur.616..266C"
    assert doc.bibliography[1].bibcode == "2023arXiv230512345B"
    assert doc.bibliography[2].bibcode is None
    db.close()


# --------------------------------------------------------------- fetch utils
def test_rate_limiter_spacing_and_retry_after():
    import httpx

    from galrag.fetch import RateLimiter

    t = [0.0]
    slept: list[float] = []

    def sleep(s):
        slept.append(s)
        t[0] += s

    rl = RateLimiter({"arxiv.org": 4.0}, default_delay=1.0, sleep=sleep, clock=lambda: t[0], backoff_base=2)
    rl.wait("arxiv.org")
    rl.wait("arxiv.org")
    assert slept == [4.0]
    calls = []

    def handler(request):
        calls.append(request.url)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        if len(calls) == 2:
            return httpx.Response(503)
        return httpx.Response(200, text="ok")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    slept.clear()
    r = rl.request(client, "GET", "https://example.org/x")
    assert r.status_code == 200 and len(calls) == 3
    assert 7 in slept and 4 in slept      # Retry-After honoured, then exponential backoff (2*2**1)


def test_usable_latexml_and_raw_paths(tmp_path):
    from galrag.config import Settings
    from galrag.fetch import fs_safe, raw_path, usable_latexml

    assert usable_latexml((FIX / "arxiv_2405.03781.html").read_text())
    assert usable_latexml((FIX / "ar5iv_1601.00001.html").read_text())
    assert not usable_latexml("<html><body><h1>No HTML for '2405.99999'</h1></body></html>")
    s = Settings(data_dir=tmp_path, work_dir=None)
    assert raw_path("arxiv_html", "astro-ph/0601001", s).name == "astro-ph_0601001.html.gz"
    assert raw_path("arxiv_pdf", "2405.03781", s) == tmp_path / "raw" / "pdf" / "2405.03781.pdf"
    assert fs_safe("2024A&A...684A..40X") == "2024A&A...684A..40X"


def test_fetch_uses_cache(tmp_path):
    import gzip

    from galrag.config import Settings
    from galrag.fetch import fetch_fulltext, raw_path
    from galrag.models import PaperMeta

    s = Settings(data_dir=tmp_path, work_dir=None)
    s.ensure_dirs()
    p = raw_path("arxiv_html", "2405.03781", s)
    p.write_bytes(gzip.compress((FIX / "arxiv_2405.03781.html").read_bytes()))
    meta = PaperMeta(paper_key="arXiv:2405.03781", arxiv_id="2405.03781")
    res = fetch_fulltext(meta, settings=s)           # must not touch the network
    assert res.source == "arxiv_html" and res.path == p and res.from_cache

    from galrag.parse import parse_fetched

    doc = parse_fetched(res, meta)
    assert doc.source == "arxiv_html" and len(doc.bibliography) > 30


@pytest.mark.skipif(os.environ.get("GALRAG_LIVE") != "1", reason="set GALRAG_LIVE=1 for network tests")
def test_live_fetch_arxiv_html():
    from galrag.fetch import fetch_fulltext
    from galrag.models import PaperMeta
    from galrag.parse import parse_fetched

    meta = PaperMeta(paper_key="arXiv:2405.03781", arxiv_id="2405.03781")
    res = fetch_fulltext(meta)
    assert res.source == "arxiv_html"
    assert res.path.exists()
    doc = parse_fetched(res, meta)
    assert_offsets(doc)
    assert len(doc.bibliography) > 30
    # second call is served from disk
    res2 = fetch_fulltext(meta)
    assert res2.from_cache and res2.path == res.path
