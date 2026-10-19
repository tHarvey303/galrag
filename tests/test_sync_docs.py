"""Documents table, legacy-file import, local<->archive sync, store-driven embedding
candidates and reconcile.  Everything runs in temporary directories; no model is loaded."""

from __future__ import annotations

import gzip
import shutil
import sqlite3

import numpy as np
import pytest

from galrag.config import Settings
from galrag.db import Database
from galrag.models import Document, PaperMeta, Status


def _doc(key: str, text: str = "Body text. " * 50) -> Document:
    return Document(paper_key=key, source="arxiv_html", title=f"Title {key}", abstract="An abstract.",
                    full_text=text)


def _db_with_papers(path, keys) -> Database:
    db = Database(path)
    db.upsert_papers([PaperMeta(paper_key=k, bibcode=k, title=f"Title {k}", abstract="An abstract.",
                                year=2024) for k in keys])
    return db


@pytest.fixture()
def settings(tmp_path):
    data, work = tmp_path / "data", tmp_path / "work"
    s = Settings(data_dir=data, work_dir=work, lancedb_dir_override=work / "lancedb")
    s.ensure_dirs()
    return s


# ------------------------------------------------------------------ settings
def test_paths_with_and_without_work_dir(tmp_path):
    s = Settings(data_dir=tmp_path / "d", work_dir=None, lancedb_dir_override=None)
    assert s.sqlite_path == s.archive_sqlite_path == tmp_path / "d" / "galrag.sqlite"
    assert s.raw_dir == tmp_path / "d" / "raw" and s.lancedb_dir == tmp_path / "d" / "lancedb"
    s = Settings(data_dir=tmp_path / "d", work_dir=tmp_path / "w", lancedb_dir_override=None)
    assert s.sqlite_path == tmp_path / "w" / "galrag.sqlite"
    assert s.archive_sqlite_path == tmp_path / "d" / "galrag.sqlite"
    assert s.raw_dir == tmp_path / "w" / "raw"
    assert s.lancedb_dir == tmp_path / "d" / "lancedb"          # LanceDB stays on data_dir by default
    assert Settings(data_dir=tmp_path, work_dir="", lancedb_dir_override="").work_dir is None


def test_env_lancedb_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GALRAG_LANCEDB_DIR", str(tmp_path / "local_lance"))
    monkeypatch.setenv("GALRAG_WORK_DIR", "")
    s = Settings(data_dir=tmp_path / "d")
    assert s.lancedb_dir == tmp_path / "local_lance" and s.work_dir is None
    assert s.archive_lancedb_dir == tmp_path / "d" / "lancedb"


# ------------------------------------------------------------------ documents
def test_documents_round_trip(tmp_path):
    from galrag.parse import load_document, save_document

    db = _db_with_papers(tmp_path / "g.sqlite", ["K1", "K2"])
    assert not db.has_document("K1") and db.get_document("K1") is None
    save_document(_doc("K1"), db)
    db.put_document(_doc("K2", "other"))
    assert db.has_document("K1") and sorted(db.iter_document_keys()) == ["K1", "K2"]
    d = load_document("K1", db)
    assert d is not None and d.full_text == _doc("K1").full_text and d.title == "Title K1"
    row = db.conn.execute("SELECT parser_version, source, json_gz FROM documents WHERE paper_key='K1'").fetchone()
    assert row["source"] == "arxiv_html" and gzip.decompress(row["json_gz"]).startswith(b"{")
    db.delete_document("K2")
    assert db.count_documents() == 1 and not db.has_document("K2")


def test_import_document_files(tmp_path):
    from galrag.parse import fs_safe

    docs = tmp_path / "docs"
    docs.mkdir()
    db = _db_with_papers(tmp_path / "g.sqlite", ["2024ApJ...1A", "arXiv:2405.03781", "K3"])
    (docs / (fs_safe("2024ApJ...1A") + ".json")).write_text(_doc("2024ApJ...1A").model_dump_json())
    (docs / (fs_safe("arXiv:2405.03781") + ".json.gz")).write_bytes(
        gzip.compress(_doc("arXiv:2405.03781").model_dump_json().encode()))
    (docs / "K3.json").write_text(_doc("K3", "already there").model_dump_json())
    (docs / "broken.json").write_text("{not json")
    (docs / "unknown.json").write_text(_doc("NOT_IN_PAPERS").model_dump_json())
    (docs / "notes.txt").write_text("ignored")
    db.put_document(_doc("K3", "newer version"))          # present keys are skipped
    assert db.import_document_files(docs, batch=1) == 2
    assert db.count_documents() == 3
    assert db.get_document("K3").full_text == "newer version"
    assert db.get_document("arXiv:2405.03781").title == "Title arXiv:2405.03781"
    assert db.import_document_files(docs) == 0               # idempotent
    assert db.import_document_files(tmp_path / "missing") == 0
    assert (docs / "K3.json").exists()                       # files are never deleted here


# ------------------------------------------------------------------ sync
def test_push_pull_sqlite_and_lancedb(settings, monkeypatch):
    from galrag import sync

    monkeypatch.setattr(shutil, "which", lambda name: None)   # exercise the copytree fallback
    db = _db_with_papers(settings.sqlite_path, ["K1"])
    db.put_document(_doc("K1"))
    (settings.lancedb_dir / "t.lance").mkdir(parents=True)
    (settings.lancedb_dir / "t.lance" / "frag").write_bytes(b"x" * 100)

    res = sync.push(settings)
    assert res["sqlite"]["bytes"] > 0 and "copytree" in res["lancedb"]["method"]
    arch = settings.archive_sqlite_path
    assert arch.exists() and not arch.with_name(arch.name + ".tmp").exists()
    assert not (arch.parent / (arch.name + "-wal")).exists()
    c = sqlite3.connect(arch)
    assert c.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
    c.close()
    assert (settings.archive_lancedb_dir / "t.lance" / "frag").read_bytes() == b"x" * 100

    # newer local writes go out with the next push (the live db stays open meanwhile)
    db.upsert_papers([PaperMeta(paper_key="K2", title="t2")])
    sync.push(settings, lancedb=False)
    assert Database(arch).get_row("K2") is not None

    # pull refuses to clobber the local db without force
    with pytest.raises(FileExistsError):
        sync.pull(settings)
    db.close()
    settings.sqlite_path.unlink()
    shutil.rmtree(settings.lancedb_dir)
    res = sync.pull(settings)
    assert res["sqlite"]["bytes"] > 0
    assert Database(settings.sqlite_path).get_document("K1").title == "Title K1"
    assert (settings.lancedb_dir / "t.lance" / "frag").exists()
    assert sync.pull(settings, lancedb=False, force=True)["sqlite"]["bytes"] > 0


def test_push_skips_without_work_dir(tmp_path):
    from galrag import sync

    s = Settings(data_dir=tmp_path, work_dir=None, lancedb_dir_override=None)
    _db_with_papers(s.sqlite_path, ["K1"]).close()
    res = sync.push(s)
    assert res["sqlite"].startswith("skipped") and res["lancedb"].startswith("skipped")


def test_sync_push_every(settings, monkeypatch):
    from galrag import sync

    calls = []
    monkeypatch.setattr(sync, "push", lambda s, **kw: calls.append(kw) or {"ok": True})
    p = sync.sync_push_every(settings, 3600, sqlite=False, lancedb=True)
    assert p() is None and calls == []                      # interval not reached
    assert p(force=True) == {"ok": True} and calls == [{"sqlite": False, "lancedb": True}]
    p0 = sync.sync_push_every(settings, 0)
    assert p0() == {"ok": True}

    def boom(s, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(sync, "push", boom)
    assert p0() is None                                     # errors are logged, not raised


# ------------------------------------------------------------------ candidates / reconcile
class FakeStore:
    """Minimal in-memory stand-in for VectorStore."""

    def __init__(self):
        self.chunks: dict[str, list[tuple[str, str]]] = {}   # key -> [(kind, source)]
        self.paper_vecs: set[str] = set()

    def fulltext_paper_keys(self):
        return {k for k, cs in self.chunks.items() if any(kind != "abstract" for kind, _ in cs)}

    def paper_vec_keys(self):
        return set(self.paper_vecs)

    def chunk_counts_by_paper(self):   # same semantics as VectorStore.chunk_counts_by_paper
        return {k: (sum(s != "abstract_only" for _, s in cs) if any(kind != "abstract" for kind, _ in cs) else 0,
                    sum(s == "abstract_only" for _, s in cs))
                for k, cs in self.chunks.items()}

    def upsert_chunks(self, chunks, vectors, paper_info=None):
        for k in {c.paper_key for c in chunks}:
            self.chunks[k] = []
        for c in chunks:
            self.chunks[c.paper_key].append((c.kind, c.source))
        return len(chunks)

    def upsert_papers(self, metas, vectors):
        self.paper_vecs |= {m.paper_key for m in metas}
        return len(metas)


class FakeEmbedder:
    def embed_passages(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)


def test_fulltext_candidates_use_store(tmp_path):
    from galrag.pipeline import fulltext_candidates

    db = _db_with_papers(tmp_path / "g.sqlite", ["A", "B", "C", "D", "E"])
    for k in ("A", "B", "C", "D"):
        db.put_document(_doc(k))
    db.set_status("A", Status.PARSED)
    db.set_status("B", Status.EMBEDDED)          # snapshot says embedded, store does not have it
    db.set_status("C", Status.PARSED)            # store already has it (embedded elsewhere)
    db.set_status("D", Status.FAILED_FETCH)
    db.set_status("E", Status.PARSED)            # no document
    store = FakeStore()
    store.chunks["C"] = [("abstract", "arxiv_html"), ("body", "arxiv_html")]
    store.chunks["A"] = [("abstract", "abstract_only")]   # abstract-only chunk does not count
    assert sorted(fulltext_candidates(db, store)) == ["A", "B"]
    assert fulltext_candidates(db, store, paper_keys=["C", "E", "nope"]) == ["C"]


def test_run_embed_abstracts_skips_store_keys_and_keeps_fulltext(tmp_path):
    from galrag.pipeline import run_embed_abstracts

    db = _db_with_papers(tmp_path / "g.sqlite", ["A", "B", "C"])
    db.set_status("B", Status.PARSED)           # stale snapshot: B is full-text embedded in the store
    store = FakeStore()
    store.paper_vecs.add("A")
    store.chunks["B"] = [("body", "arxiv_html")]
    res = run_embed_abstracts(db, store, FakeEmbedder())
    assert res == {"papers": 2, "abstract_chunks": 1}       # B and C; only C gets an abstract chunk
    assert store.chunks["B"] == [("body", "arxiv_html")]
    assert store.chunks["C"] == [("abstract", "abstract_only")]
    assert run_embed_abstracts(db, store, FakeEmbedder()) == {"papers": 0, "abstract_chunks": 0}


def test_reconcile_from_store(tmp_path):
    db = _db_with_papers(tmp_path / "g.sqlite", ["A", "B", "C", "D"])
    db.set_status("A", Status.PARSED)
    db.set_status("B", Status.EMBEDDED, n_chunks=3)
    store = FakeStore()
    store.chunks["A"] = [("abstract", "arxiv_html")] + [("body", "arxiv_html")] * 4
    store.chunks["B"] = [("body", "arxiv_html")] * 3
    store.chunks["C"] = [("abstract", "abstract_only")]
    store.paper_vecs |= {"A", "C"}
    res = db.reconcile_from_store(store)
    assert res["embedded"] == 1 and res["paper_vec"] == 2 and res["abs_chunk"] == 1
    a, b, c, d = (db.get_row(k) for k in "ABCD")
    assert a["status"] == Status.EMBEDDED and a["n_chunks"] == 5 and a["paper_vec"] == 1
    assert b["status"] == Status.EMBEDDED and b["n_chunks"] == 3
    assert c["status"] == Status.METADATA and c["abs_chunk"] == 1 and c["paper_vec"] == 1
    assert d["paper_vec"] == 0 and d["abs_chunk"] == 0
    assert db.reconcile_from_store(store)["embedded"] == 0   # idempotent


def test_store_key_scans(tmp_path):
    """The real VectorStore projected scans (tiny LanceDB table, no model)."""
    from galrag.models import Chunk
    from galrag.store import VectorStore

    s = Settings(data_dir=tmp_path, work_dir=None, lancedb_dir_override=None)
    store = VectorStore(settings=s, uri=str(tmp_path / "lancedb"))
    assert store.fulltext_paper_keys() == set() and store.paper_vec_keys() == set()

    def ch(key, i, kind, source):
        return Chunk(chunk_id=f"{key}#{i:04d}", paper_key=key, ordinal=i, kind=kind, section_id="s",
                     section_title="", section_path="", para_ids=[], text=f"text {key} {i}",
                     char_start=0, char_end=1, n_tokens=2, source=source)

    cs = [ch("A", 0, "abstract", "arxiv_html"), ch("A", 1, "body", "arxiv_html"),
          ch("B", 0, "abstract", "abstract_only")]
    store.upsert_chunks(cs, np.eye(3, 4, dtype=np.float32))
    store.upsert_papers([{"paper_key": "A"}, {"paper_key": "B"}], np.ones((2, 4), dtype=np.float32))
    assert store.fulltext_paper_keys() == {"A"}
    assert store.paper_vec_keys() == {"A", "B"}
    assert store.chunk_counts_by_paper() == {"A": (2, 0), "B": (0, 1)}


def test_archive_raw_tar_and_delete(tmp_path):
    import tarfile

    from galrag.config import Settings
    from galrag.sync import archive_raw

    data, work = tmp_path / "data", tmp_path / "work"
    s = Settings(ads_token="x", data_dir=data, work_dir=work)
    raw = s.raw_dir / "html"
    raw.mkdir(parents=True)
    for i in range(3):
        (raw / f"p{i}.html.gz").write_bytes(b"x" * 1000)
    assert "skipped" in archive_raw(s, min_age_s=0)["raw"]  # below the size threshold
    res = archive_raw(s, min_mb=0, min_age_s=0)["raw"]
    assert res["files"] == 3 and not list(raw.glob("*"))
    with tarfile.open(res["tar"]) as t:
        assert sorted(t.getnames()) == [f"html/p{i}.html.gz" for i in range(3)]
    assert "skipped" in archive_raw(s, min_mb=0, min_age_s=0)["raw"]


def test_embed_documents_marks_papers_without_body_text(tmp_path):
    """A parsed document with no body text yields only an abstract chunk; such a paper
    must become failed:embed, not embedded, or it is re-embedded on every pass."""
    from galrag.models import Paragraph, Section
    from galrag.pipeline import embed_documents

    db = _db_with_papers(tmp_path / "g.sqlite", ["A", "B"])
    meta_a, meta_b = (PaperMeta.model_validate_json(db.get_row(k)["meta_json"]) for k in ("A", "B"))
    body = "We observed the field and reduced the data with the standard pipeline. " * 6
    sec = Section(id="S1", title="Observations", level=1,
                  paragraphs=[Paragraph(id="S1.P1", kind="body", text=body, char_start=0,
                                        char_end=len(body), page=None)])
    doc_a = Document(paper_key="A", source="arxiv_html", title="Title A", abstract="An abstract.",
                     sections=[sec], full_text=body)
    doc_b = _doc("B", text="")                     # parser produced no body text at all
    store = FakeStore()
    n = embed_documents(db, store, FakeEmbedder(), [(meta_a, doc_a), (meta_b, doc_b)])
    assert n > 0 and "A" in store.fulltext_paper_keys()
    assert db.get_row("A")["status"] == Status.EMBEDDED
    assert db.get_row("B")["status"] == Status.FAILED_EMBED
    assert "no body text" in (db.get_row("B")["error"] or "")
    assert "B" not in store.chunks
