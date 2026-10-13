"""Live ADS tests (skipped without a token). Small queries only."""

import pytest

from galrag.ads import ADSClient, build_citation_query, get_bibtex_cached, split_bibtex
from galrag.config import get_settings
from galrag.db import Database

pytestmark = pytest.mark.skipif(not get_settings().ads_token, reason="no GALRAG_ADS_TOKEN")


@pytest.fixture(scope="module")
def ads():
    with ADSClient() as c:
        yield c


def test_count_topic(ads):
    assert ads.count(get_settings().topic_query) > 0
    assert ads.remaining is not None


def test_search_has_references(ads):
    docs = list(ads.search("first_author:Bunker year:2023 title:JADES", rows=5, max_records=5))
    assert docs and len(docs) <= 5
    assert any("reference" in d for d in docs)


def test_export_bibtex_cached(ads, tmp_path):
    out = ads.export_bibtex(["2024ApJ...968..118P"])
    assert "2024ApJ...968..118P" in out
    db = Database(tmp_path / "t.sqlite")
    r = get_bibtex_cached(db, ["2024ApJ...968..118P"], client=ads)
    assert "2024ApJ...968..118P" in r
    assert db.get_bibtex("2024ApJ...968..118P")


def test_resolve_citation(ads):
    res = ads.resolve_citation("Bunker et al. 2023")
    assert res and res[0]["bibcode"].startswith("2023")
    res = ads.resolve_citation("Finkelstein+23")
    assert res and res[0]["year"] == 2023


def test_resolve_by_ids(ads):
    res = ads.resolve_citation("2024ApJ...968..118P")
    assert res and res[0]["bibcode"] == "2024ApJ...968..118P"


def test_citations_refs(ads):
    assert ads.citations_of("2024ApJ...968..118P", rows=3)
    assert ads.references_of_live("2024ApJ...968..118P")


def test_harvest(ads, tmp_path):
    db = Database(tmp_path / "h.sqlite")
    n_new, n_upd = ads.harvest(db, max_records=10)
    assert (n_new, n_upd) == (10, 0)
    assert db.count_by_status() == {"metadata": 10}
    row = db.conn.execute("SELECT paper_key, arxiv_id FROM papers WHERE arxiv_id IS NOT NULL LIMIT 1").fetchone()
    if row:
        assert db.resolve("arXiv:" + row["arxiv_id"]) == row["paper_key"]
        assert isinstance(db.references_of(row["paper_key"]), list)
    key = db.conn.execute("SELECT paper_key FROM papers LIMIT 1").fetchone()[0]
    assert isinstance(db.references_of(key), list)
    # incremental update path with a since date works
    n2 = ads.harvest(db, max_records=5, since_entdate=db.max_entdate())
    assert sum(n2) <= 5


def test_query_parsing_offline():
    q, _ = build_citation_query("Finkelstein+23")
    assert q == 'first_author:"Finkelstein" year:2023'
    q, _ = build_citation_query("Bunker, Saxena, Cameron 2023 A&A")
    assert 'first_author:"Bunker"' in q and "year:2023" in q
    assert build_citation_query("arXiv:2405.03781v2")[0] == 'identifier:"arXiv:2405.03781"'
    assert build_citation_query("doi:10.1051/0004-6361/202346159")[0].startswith('identifier:"10.1051')
    assert split_bibtex("@ARTICLE{2023A&A...000..1B,\n a=1}\n@ARTICLE{x,\n}")["x"]
