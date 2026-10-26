"""arXiv RSS feed screening: parsing, topic rule, re-keying, ingest_arxiv_feed.
Everything runs in tmp_path; no network, no models."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import feedparser
import numpy as np
import pytest

from galrag import fetch as fetch_mod
from galrag.arxiv_feed import (
    _api_entry,
    ads_name,
    fetch_feed,
    fetch_listing,
    meta_from_feed,
    parse_feed,
    split_authors,
    topic_keywords,
)
from galrag.config import TOPIC_TERMS
from galrag.db import Database
from galrag.models import Document, PaperMeta, Status
from galrag.service import ingest_arxiv_feed, purge_rekeyed

FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_astro-ph.rss"
ACCEPT = {"2610.32307", "2610.32497", "2610.32786"}


class FakeStore:
    def __init__(self, sim: float | None = None):
        self.sim = sim
        self.deleted: list[str] = []
        self.refreshed = 0

    @property
    def reranker(self):
        return self

    def rerank(self, query, passages):
        assert query and passages
        if self.sim is None:
            raise RuntimeError("no reranker in this test")
        return [self.sim] * len(passages)

    def delete_paper(self, key):
        self.deleted.append(key)

    def refresh_indexes(self):
        self.refreshed += 1

    def upsert_chunks(self, chunks, vectors, paper_info=None):
        return len(chunks)

    def upsert_papers(self, metas, vectors):
        return len(metas)


class FakeEmbedder:
    def embed_passages(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)


def _factory():
    return FakeEmbedder()


def _entries():
    return {e.arxiv_id: e for e in parse_feed(FIXTURE.read_bytes())[1]}


@pytest.fixture()
def db(tmp_path):
    return Database(tmp_path / "g.sqlite")


@pytest.fixture()
def fetch_calls(monkeypatch):
    calls = []

    def boom(db, meta, *, force=False):
        calls.append((meta.paper_key, force))
        db.set_status(meta.paper_key, Status.FAILED_FETCH, error="no network in tests")
        raise fetch_mod.FetchError("no network in tests")

    monkeypatch.setattr("galrag.pipeline.fetch_and_parse_one", boom)
    return calls


def _run(db, store=None, **kw):
    kw.setdefault("feed_path", str(FIXTURE))
    kw.setdefault("use_ads", False)
    return ingest_arxiv_feed(db, store or FakeStore(), _factory, **kw)


# ------------------------------------------------------------------ 1. parsing
def test_parse_feed_fixture():
    info, entries = parse_feed(FIXTURE.read_bytes())
    assert len(entries) == 6 and info["n_items"] == 6
    by = {e.arxiv_id: e for e in entries}
    assert set(by) == {"2610.31688", "2610.32307", "2610.32497", "2610.32786", "2610.27052", "2610.30049"}
    assert {k: v.announce_type for k, v in by.items()} == {
        "2610.31688": "new", "2610.32307": "new", "2610.32497": "new", "2610.32786": "new",
        "2610.27052": "replace", "2610.30049": "cross"}
    assert by["2610.27052"].version == 2 and by["2610.32307"].version == 1
    assert by["2610.27052"].is_replacement and not by["2610.30049"].is_replacement
    for e in entries:
        assert e.published == "2026-10-26"
        assert e.categories and e.title
        assert not e.abstract.startswith("arXiv:") and "Announce Type" not in e.abstract
        assert e.abstract and not e.abstract.startswith("Abstract:")
    assert by["2610.32307"].authors == ["Sun, Bangzheng", "Yan, Haojing"]
    assert by["2610.27052"].authors[0] == "Herard-Demanche, Thomas"
    assert "van der Werf, Paul" in by["2610.27052"].authors
    assert "astro-ph.GA" in by["2610.32307"].categories


# ------------------------------------------------------------------ 2. names
def test_ads_name_and_split_authors():
    assert ads_name("Arjen van der Wel") == "van der Wel, Arjen"
    assert ads_name("Paul van der Werf") == "van der Werf, Paul"
    assert ads_name("Jane Q. Doe Jr.") == "Doe, Jane Q., Jr."
    assert ads_name("Doe, Jane") == "Doe, Jane"
    assert ads_name("Plato") == "Plato"
    assert ads_name("Jane Doe") == "Doe, Jane"
    assert split_authors("Jane Doe, Arjen van der Wel (MPIA), Bob Ross (Inst A, Dept B)") == [
        "Doe, Jane", "van der Wel, Arjen", "Ross, Bob"]
    assert split_authors("") == []


# ------------------------------------------------------------------ 3. topic rule
def test_topic_keywords():
    assert topic_keywords("Galaxies at High-Redshift", "") == ["high redshift"]
    assert "high redshift" in topic_keywords("A galaxy", "at high redshift")
    assert "reionization" in topic_keywords("Galaxies", "during reionisation")
    assert "jwst" in [k.lower() for k in topic_keywords("Galaxy", "seen by JWST")]
    assert topic_keywords("A galaxy cluster", "dynamics of nearby spirals") == []
    assert topic_keywords("JWST observations", "of exoplanet atmospheres and stars") == []
    assert topic_keywords("Cosmic dawn", "star formation") == []          # no galaxy word
    assert topic_keywords("Galaxy", "x", keywords=("foo",)) == []
    assert topic_keywords("Galaxy", "foo bar", keywords=("foo",)) == ["foo"]


def test_topic_terms_literal():
    assert TOPIC_TERMS == (
        '"high redshift" OR "high-redshift" OR "early universe" OR reionization OR "cosmic dawn" OR '
        '"cosmic noon" OR "Lyman break" OR "Lyman-break" OR "Lyman alpha emitter" OR '
        '"Lyman-alpha emitter" OR JWST OR "epoch of reionization"')


# ------------------------------------------------------------------ 4. meta
def test_meta_from_feed():
    e = _entries()["2610.32307"]
    m = meta_from_feed(e)
    assert m.paper_key == "arXiv:2610.32307" and m.arxiv_id == "2610.32307" and m.bibcode is None
    assert m.year == 2026 and m.pubdate == "2026-10-26"
    assert m.tier.value == "B" and not m.refereed
    assert m.first_author == "Sun, Bangzheng"
    assert m.raw["source"] == "arxiv_rss" and m.raw["announce_type"] == "new"


# ------------------------------------------------------------------ 5. re-key
def _ads_meta(**kw):
    return PaperMeta(paper_key="2026arXiv261032307S", bibcode="2026arXiv261032307S",
                     arxiv_id="2610.32307", title="ADS title", year=2026, **kw)


def test_rekey_on_ads_record(db):
    from galrag.parse import save_document

    m = meta_from_feed(_entries()["2610.32307"])
    m.references = ["2020ApJ...1....1X"]
    db.upsert_papers([m])
    old, new = "arXiv:2610.32307", "2026arXiv261032307S"
    save_document(Document(paper_key=old, source="arxiv_html", title="t", abstract="a", full_text="body " * 20), db)
    db.set_status(old, Status.EMBEDDED, n_chunks=5)
    db.conn.execute("UPDATE papers SET paper_vec=1, abs_chunk=1 WHERE paper_key=?", (old,))
    db.conn.commit()

    assert db.upsert_papers([_ads_meta()]) == (0, 1)
    assert db.get_row(old) is None
    row = db.get_row(new)
    assert row is not None and row["status"] == "parsed" and row["n_chunks"] == 0 and row["paper_vec"] == 0
    assert db.resolve("2610.32307") == new and db.resolve(new) == new
    assert db.resolve("arXiv:2610.32307") == new
    assert not db.has_document(old) and db.has_document(new)
    assert db.get_document(new).paper_key == new
    refs = db.conn.execute("SELECT citing_key, cited_bibcode FROM refs").fetchall()
    assert [(r[0], r[1]) for r in refs] == [(new, "2020ApJ...1....1X")]
    assert db.pending_rekeys() == [(old, new)]

    store = FakeStore()
    assert purge_rekeyed(db, store) == [(old, new)] and store.deleted == [old]
    assert db.pending_rekeys() == []
    assert purge_rekeyed(db, store) == []

    # same ADS record again: plain update
    assert db.upsert_papers([_ads_meta()]) == (0, 1)
    assert db.pending_rekeys() == []
    assert db.conn.execute("SELECT COUNT(*) FROM rekeyed").fetchone()[0] == 1


def test_rekey_not_triggered_without_feed_row(db):
    assert db.upsert_papers([_ads_meta()]) == (1, 0)
    assert db.pending_rekeys() == []


# ------------------------------------------------------------------ 6. ingest_arxiv_feed
def test_feed_dry_run(db, fetch_calls):
    out = _run(db, FakeStore(sim=0.5), dry_run=True)
    assert db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0
    assert fetch_calls == []
    assert {a["arxiv_id"] for a in out["accepted"]} == ACCEPT
    assert out["n_entries"] == 6 and out["by_type"] == {"new": 4, "replace": 1, "cross": 1}
    cands = {c["arxiv_id"]: c for c in out["candidates"]}
    assert "2610.31688" in cands and "2610.31688" not in {a["arxiv_id"] for a in out["accepted"]}
    for k in ACCEPT:
        assert cands[k]["keywords"] and cands[k]["topic_score"] == 0.5
    skipped = {s["arxiv_id"]: s for s in out["skipped"]}
    assert skipped["2610.27052"]["note"] == "replacement"
    rows = {r["arxiv_id"]: r for r in db.feed_history(limit=50)}
    assert rows["2610.27052"]["decision"] == "skipped"
    assert {rows[k]["decision"] for k in cands} == {"dry-run"}
    assert rows["2610.32307"]["keywords"]


def test_feed_real_run_and_rerun(db, fetch_calls):
    store = FakeStore()
    out = _run(db, store, embed=False)
    assert {a["arxiv_id"] for a in out["accepted"]} == ACCEPT
    for k in ACCEPT:
        row = db.get_row(f"arXiv:{k}")
        assert row is not None and row["status"] == Status.FAILED_FETCH
    assert db.get_row("arXiv:2610.31688") is None
    assert {r["metadata"] for r in out["ingested"]} == {"feed"}
    assert all(str(r["fulltext"]).startswith("unavailable") for r in out["ingested"])
    dec = {r["arxiv_id"]: r["decision"] for r in db.feed_history(limit=50)}
    assert {dec[k] for k in ACCEPT} == {"accepted"}
    assert dec["2610.31688"] == "rejected" and dec["2610.27052"] == "skipped"
    assert db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 3

    # failed:fetch is not "already in corpus": they are candidates again
    n = len(fetch_calls)
    out2 = _run(db, store, embed=False)
    assert {a["arxiv_id"] for a in out2["accepted"]} == ACCEPT
    assert all(s["note"] != "already in corpus" for s in out2["skipped"])
    assert len(fetch_calls) == n + 3

    # once parsed they are skipped
    for k in ACCEPT:
        db.set_status(f"arXiv:{k}", Status.PARSED)
    out3 = _run(db, store, embed=False)
    assert out3["accepted"] == []
    assert {s["arxiv_id"] for s in out3["skipped"] if s["note"] == "already in corpus"} == ACCEPT


def test_feed_max_papers(db, fetch_calls):
    out = _run(db, embed=False, max_papers=1)
    assert len(out["accepted"]) == 1 and out["accepted"][0]["arxiv_id"] in ACCEPT
    assert db.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
    notes = {r["arxiv_id"]: r["note"] for r in db.feed_history(limit=50)}
    assert sum(1 for v in notes.values() if v == "over max_papers") == 2


def test_feed_min_score(db, fetch_calls):
    out = _run(db, FakeStore(sim=0.9), embed=False, min_score=0.8)
    assert {a["arxiv_id"] for a in out["accepted"]} >= ACCEPT | {"2610.31688"}
    assert db.get_row("arXiv:2610.31688") is not None
    out = _run(db, FakeStore(sim=0.5), embed=False, min_score=0.8)
    assert {a["arxiv_id"] for a in out["accepted"]} == ACCEPT


def test_feed_include_replacements(db, fetch_calls):
    from galrag.parse import save_document

    m = meta_from_feed(_entries()["2610.27052"])
    db.upsert_papers([m])
    save_document(Document(paper_key=m.paper_key, source="arxiv_html", title="t", abstract="a",
                           full_text="body " * 20), db)
    db.set_status(m.paper_key, Status.PARSED)

    out = _run(db, embed=False)
    assert "2610.27052" in {s["arxiv_id"] for s in out["skipped"]}
    assert (m.paper_key, True) not in fetch_calls

    out = _run(db, embed=False, include_replacements=True)
    assert "2610.27052" in {a["arxiv_id"] for a in out["accepted"]}
    assert (m.paper_key, True) in fetch_calls


# ------------------------------------------------------------------ 7. fetch_feed
def test_fetch_feed_http(monkeypatch):
    seen = {}

    class R:
        status_code = 200
        content = FIXTURE.read_bytes()

        def raise_for_status(self):
            pass

    def fake_get(url, **kw):
        seen["url"] = url
        return R()

    monkeypatch.setattr("galrag.arxiv_feed.httpx.get", fake_get)
    info, entries = fetch_feed()
    assert info["source"] == "https://rss.arxiv.org/rss/astro-ph.GA" == seen["url"]
    assert len(entries) == 6
    info, _ = fetch_feed("astro-ph.CO")
    assert info["source"].endswith("/astro-ph.CO")


# ------------------------------------------------------------------ 7. arXiv API listing
API_FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_api_astro-ph.GA.atom"


def _api_entries():
    return [_api_entry(e, "astro-ph.GA") for e in feedparser.parse(str(API_FIXTURE)).entries]


def test_api_entry_mapping():
    by = {e.arxiv_id: e for e in _api_entries()}
    assert list(by) == ["2610.28515", "2610.28073", "2610.27526"]
    assert {k: v.announce_type for k, v in by.items()} == {
        "2610.28515": "new", "2610.28073": "cross", "2610.27526": "new"}
    assert all(e.version == 1 for e in by.values())
    assert "astro-ph.GA" in by["2610.28515"].categories
    assert by["2610.28073"].categories[0] == "astro-ph.HE" and "astro-ph.GA" in by["2610.28073"].categories
    assert by["2610.28515"].authors == ["Doe, Jane", "van der Wel, Arjen", "Ross, Bob"]
    assert by["2610.28515"].published == "2026-10-20" and by["2610.27526"].published == "2026-10-21"
    assert by["2610.28515"].title == "A JWST view of compact star-forming galaxies at z>10"
    assert "\n" not in by["2610.28515"].abstract and "  " not in by["2610.28515"].abstract
    assert by["2610.28515"].abstract.startswith("We present JWST/NIRSpec")


class FakeApiClient:
    """Serves the fixture entries as API pages, honouring start/max_results."""
    calls: ClassVar[list[dict]] = []

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, params=None):
        params = dict(params)
        self.calls.append(params)
        f = feedparser.parse(str(API_FIXTURE))
        start, n = int(params["start"]), int(params["max_results"])
        items = f.entries[start:start + n]
        raw = API_FIXTURE.read_text()
        head = raw.partition("<entry>")[0]
        blocks = re.findall(r"<entry>.*?</entry>", raw, re.DOTALL)[start:start + n]
        body = head + "\n".join(blocks) + "\n</feed>\n"
        assert len(items) == len(blocks)
        return SimpleNamespace(status_code=200, content=body.encode(), raise_for_status=lambda: None)


def test_fetch_listing_pages(monkeypatch):
    FakeApiClient.calls = []
    monkeypatch.setattr("galrag.arxiv_feed.httpx.Client", FakeApiClient)
    info, entries = fetch_listing("astro-ph.GA", since="2026-10-20", until="2026-10-21", page=2, delay_s=0)
    assert [e.arxiv_id for e in entries] == ["2610.28515", "2610.28073", "2610.27526"]
    assert info["total_results"] == 3 and info["n_items"] == 3
    assert info["since"] == "2026-10-20" and info["until"] == "2026-10-21"
    assert [c["start"] for c in FakeApiClient.calls] == [0, 2]
    assert "submittedDate:[202610200000 TO 202610212359]" in FakeApiClient.calls[0]["search_query"]
    assert "cat:astro-ph.GA" in FakeApiClient.calls[0]["search_query"]
    with pytest.raises(ValueError):
        fetch_listing("astro-ph.GA", since="2026-10-22", until="2026-10-21", delay_s=0)
    FakeApiClient.calls = []
    _, one = fetch_listing("astro-ph.GA", since="2026-10-20", until="2026-10-21", page=2, delay_s=0,
                           max_entries=1)
    assert len(one) == 1


def test_ingest_since_uses_listing(db, fetch_calls, monkeypatch):
    entries = _api_entries()
    seen = {}

    def fake_listing(category=None, *, since, until=None, **kw):
        seen.update(category=category, since=since, until=until)
        info = {"title": f"arXiv API: astro-ph.GA {since} to {until}", "updated": None,
                "source": "test", "n_items": len(entries), "since": since, "until": until}
        return info, entries

    monkeypatch.setattr("galrag.arxiv_feed.fetch_listing", fake_listing)
    out = ingest_arxiv_feed(db, FakeStore(sim=0.05), _factory, since="2026-10-20", until="2026-10-21",
                            dry_run=True, use_ads=False)
    assert seen["since"] == "2026-10-20" and seen["until"] == "2026-10-21"
    assert [a["arxiv_id"] for a in out["accepted"]] == ["2610.28515"]
    assert out["rejected"] == 2 and out["n_entries"] == 3
    assert "2026-10-20" in out["feed"]["title"] and "2026-10-21" in out["feed"]["title"]
    assert len(db.feed_history(limit=50)) == 3
    assert fetch_calls == []


def test_cli_days_option():
    from typer.testing import CliRunner

    from galrag.cli import app

    res = CliRunner().invoke(app, ["rss", "--days", "3", "--since", "2026-10-20"])
    assert res.exit_code != 0
    assert "either --since or --days" in " ".join(res.output.split())
