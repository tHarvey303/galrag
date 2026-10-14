"""``galrag`` command-line interface (typer).

Pipeline order: ``harvest`` -> ``fetch`` -> ``embed`` -> ``index``; later ``update``
(new ADS records) and ``ingest`` (specific papers).  ``serve`` starts the MCP server.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="galrag", help="Citable RAG over the high-redshift galaxy-evolution literature.",
                  no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False)
console = Console()


def _split(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


@app.callback()
def _main(
    data_dir: Annotated[Path | None, typer.Option("--data-dir", envvar="GALRAG_DATA_DIR",
                                                  help="Corpus directory (default <repo>/data).")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Only warnings on the console.")] = False,
) -> None:
    from galrag import config
    from galrag.service import setup_logging

    if data_dir is not None:
        os.environ["GALRAG_DATA_DIR"] = str(Path(data_dir).resolve())
        config.get_settings.cache_clear()
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    setup_logging(level, console="rich")


def _db():
    from galrag.db import Database

    return Database()


def _store(**kw):
    from galrag.store import VectorStore

    return VectorStore(**kw)


def _print_counts(title: str, d: dict) -> None:
    t = Table(title=title, show_header=False, title_justify="left")
    for k, v in sorted(d.items(), key=lambda kv: str(kv[0])):
        t.add_row(str(k), f"{v:,}" if isinstance(v, int) else str(v))
    console.print(t)


# ------------------------------------------------------------------ ADS
@app.command("archive-raw")
def archive_raw_cmd(
    source: Annotated[Path | None, typer.Option(help="Directory to archive (default: the raw cache).")] = None,
    min_mb: Annotated[float, typer.Option(help="Only archive if at least this many MB (0 = always).")] = 0,
    min_age_s: Annotated[float, typer.Option(help="Skip files modified less than this many seconds ago.")] = 600,
    keep: Annotated[bool, typer.Option("--keep", help="Keep the local files after archiving.")] = False,
):
    """Tar the raw HTML/PDF cache into data_dir/raw_archive (bulk move to Nexus)."""
    from galrag.config import get_settings
    from galrag.sync import archive_raw

    res = archive_raw(get_settings(), source=source, min_mb=min_mb, min_age_s=min_age_s, delete=not keep)
    console.print(res)


@app.command()
def harvest(
    query: Annotated[str | None, typer.Option(help="ADS query (default: settings.topic_query).")] = None,
    max_records: Annotated[int | None, typer.Option(help="Stop after N records.")] = None,
) -> None:
    """Full ADS harvest of metadata (+ reference lists) into SQLite."""
    from galrag.ads import ADSClient

    db = _db()
    t0 = time.time()
    with ADSClient() as c:
        n_new, n_upd = c.harvest(db, query=query, max_records=max_records)
    console.print(f"[bold]harvest:[/] {n_new:,} new, {n_upd:,} updated in {time.time() - t0:.0f} s")
    _print_counts("papers by tier", db.count_by_tier())


@app.command()
def update(
    process: Annotated[bool, typer.Option("--process/--no-process",
                                          help="Fetch/parse/embed the new papers.")] = True,
    max_papers: Annotated[int | None, typer.Option(help="Max new tier-A/B papers to fetch.")] = None,
    workers: Annotated[int, typer.Option(help="Embedding worker processes.")] = 1,
    threads: Annotated[int | None, typer.Option(help="Threads per embedding worker.")] = None,
) -> None:
    """Incremental ADS update (records entered since the last run), then process new papers."""
    from galrag.embed import PoolEmbedder
    from galrag.service import update_corpus

    db, store = _db(), _store()
    with PoolEmbedder(n_workers=workers, threads_per_worker=threads) as pe:
        res = update_corpus(db, store, lambda: pe, process=process, max_papers=max_papers)
    console.print_json(json.dumps(res, default=str))


# ------------------------------------------------------------------ pipeline
@app.command()
def fetch(
    tiers: Annotated[str, typer.Option(help="Comma-separated tiers.")] = "A,B",
    limit: Annotated[int | None, typer.Option(help="Max papers.")] = None,
    retry_failed: Annotated[bool, typer.Option(help="Also retry failed:fetch / failed:parse.")] = False,
    keys: Annotated[str | None, typer.Option(help="Comma-separated paper keys / identifiers.")] = None,
) -> None:
    """Fetch and parse full text (arXiv HTML -> ar5iv -> PDF), resumable, in priority order."""
    from galrag.pipeline import run_fetch_parse

    db = _db()
    pk = [db.resolve(k) or k for k in _split(keys)] if keys else None
    t0 = time.time()
    counts = run_fetch_parse(db, tiers=tuple(_split(tiers) or ["A", "B"]), limit=limit, paper_keys=pk,
                             retry_failed=retry_failed)
    console.print(f"[bold]fetch/parse:[/] {counts} in {time.time() - t0:.0f} s")


@app.command()
def embed(
    limit: Annotated[int | None, typer.Option(help="Max papers per stage.")] = None,
    keys: Annotated[str | None, typer.Option(help="Comma-separated paper keys / identifiers.")] = None,
    workers: Annotated[int | None, typer.Option(help="Embedding worker processes.")] = None,
    threads: Annotated[int | None, typer.Option(help="Torch threads per worker.")] = None,
    abstracts: Annotated[bool, typer.Option("--abstracts/--no-abstracts",
                                            help="Title+abstract vectors for every paper.")] = True,
    fulltext: Annotated[bool, typer.Option("--fulltext/--no-fulltext",
                                           help="Chunk+embed parsed full texts.")] = True,
) -> None:
    """Embed paper vectors (title+abstract) and parsed full texts into LanceDB, then index."""
    from galrag.embed import PoolEmbedder
    from galrag.pipeline import run_embed, run_embed_abstracts
    from galrag.service import embed_paper_vectors

    db, store = _db(), _store()
    pk = [db.resolve(k) or k for k in _split(keys)] if keys else None
    t0 = time.time()
    with PoolEmbedder(n_workers=workers, threads_per_worker=threads) as pe:
        if fulltext:
            r = run_embed(db, store, pe, limit=limit, paper_keys=pk)
            dt = time.time() - t0
            console.print(f"[bold]full text:[/] {r['papers']:,} papers, {r['chunks']:,} chunks, "
                          f"{r['failed']} failed ({r['chunks'] / max(dt, 1e-6):.1f} chunks/s)")
        if abstracts:
            t1 = time.time()
            r = (embed_paper_vectors(db, store, pe, pk) if pk
                 else run_embed_abstracts(db, store, pe, limit=limit))
            dt = time.time() - t1
            console.print(f"[bold]paper vectors:[/] {r['papers']:,} papers, "
                          f"{r['abstract_chunks']:,} abstract-only chunks "
                          f"({r['papers'] / max(dt, 1e-6):.1f} papers/s)")
    console.print(f"index: {store.create_indexes()}")
    store.refresh_indexes()
    console.print(f"chunks in store: {store.count():,}  ({time.time() - t0:.0f} s total)")


@app.command()
def index(force: Annotated[bool, typer.Option(help="Rebuild indexes from scratch.")] = False) -> None:
    """Create/refresh the FTS, scalar and (large tables) IVF-PQ indexes."""
    store = _store()
    t0 = time.time()
    res = store.create_indexes(force=force)
    store.refresh_indexes()
    console.print(f"{res}  chunks={store.count():,}  ({time.time() - t0:.0f} s)")


@app.command()
def ingest(
    idents: Annotated[list[str], typer.Argument(help="Bibcodes, arXiv ids, DOIs or 'query:<ADS query>'.")],
    embed: Annotated[bool, typer.Option("--embed/--no-embed", help="Embed after parsing.")] = True,
    force: Annotated[bool, typer.Option(help="Re-fetch even if already parsed.")] = False,
    max_records: Annotated[int, typer.Option(help="Max papers for a query: ident.")] = 20,
) -> None:
    """Add specific papers (any year/doctype): metadata -> full text -> embedding."""
    from galrag.embed import Embedder
    from galrag.service import ingest_ident

    db = _db()
    store = _store(embedder=Embedder()) if embed else _store()
    for ident in idents:
        for r in ingest_ident(db, store, store.embedder if embed else None, ident, embed=embed,
                              force_fetch=force, max_records=max_records):
            console.print_json(json.dumps(r, default=str))
    if embed:
        try:
            store.refresh_indexes()
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]index refresh failed: {e}")


# ------------------------------------------------------------------ query
@app.command()
def search(
    query: Annotated[str, typer.Argument()],
    k: Annotated[int, typer.Option("-k", help="Number of hits.")] = 10,
    mode: Annotated[str, typer.Option(help="hybrid | vector | fts")] = "hybrid",
    year_from: Annotated[int | None, typer.Option()] = None,
    year_to: Annotated[int | None, typer.Option()] = None,
    author: Annotated[str | None, typer.Option(help="First-author surname prefix.")] = None,
    refereed: Annotated[bool, typer.Option(help="Refereed papers only.")] = False,
    rerank: Annotated[bool, typer.Option("--rerank/--no-rerank")] = True,
    papers: Annotated[bool, typer.Option(help="Paper-level (title+abstract) search instead.")] = False,
) -> None:
    """Search the corpus (debugging aid; the MCP server exposes the same search)."""
    store = _store()
    f = {k_: v for k_, v in {"year_from": year_from, "year_to": year_to, "first_author": author,
                             "refereed_only": refereed}.items() if v}
    t0 = time.time()
    if papers:
        for i, r in enumerate(store.search_papers(query, k=k, filters=f or None), 1):
            console.print(f"[bold]{i:2d}[/] {r['score']:.3f}  {r['first_author']} ({r['year']})  "
                          f"[cyan]{r['title']}[/]  {r['paper_key']}")
        return
    hits = store.search(query, k=k, mode=mode, filters=f or None, rerank=rerank)
    for h in hits:
        c = h.chunk
        txt = " ".join(c.text.split())
        console.rule(f"[bold]{h.rank}[/]  score {h.score:.3f}")
        console.print(f"[bold]{h.authors_short}[/]  [cyan]{h.title}[/]")
        console.print(f"[dim]{c.section_path}  |  {c.chunk_id}  |  {c.source}  "
                      f"[{c.char_start}:{c.char_end}]" + (f" p.{c.page_start}" if c.page_start else ""))
        console.print(txt[:300] + ("..." if len(txt) > 300 else ""), highlight=False)
    console.print(f"[dim]{len(hits)} hits in {time.time() - t0:.1f} s")


@app.command()
def quote(paper: Annotated[str, typer.Argument(help="Paper identifier.")],
          text: Annotated[str, typer.Argument(help="Text to locate.")],
          min_similarity: Annotated[float, typer.Option()] = 60.0) -> None:
    """Locate a (possibly paraphrased) quote in a paper's full text."""
    from galrag.parse import load_document
    from galrag.quotes import find_quote

    db = _db()
    key = db.resolve(paper)
    if key is None:
        console.print(f"[red]unknown paper {paper!r}")
        raise typer.Exit(1)
    doc = load_document(key, db)
    if doc is None:
        console.print(f"[red]no full text for {key}")
        raise typer.Exit(1)
    m = find_quote(doc, text, min_similarity=min_similarity)
    colour = "green" if m.exact else ("yellow" if m.found else "red")
    console.print(f"[{colour}]found={m.found} exact={m.exact} similarity={m.similarity:.1f}[/]")
    console.print(f"section {m.section_id} {m.section_title!r}  para {m.para_id}  "
                  f"[{m.char_start}:{m.char_end}]  page {m.page}  source {m.source}")
    console.print(m.quote, highlight=False)


@app.command()
def paper(ident: Annotated[str, typer.Argument()]) -> None:
    """Show a paper's metadata, ingestion status and section outline."""
    from galrag.parse import load_document

    db = _db()
    key = db.resolve(ident)
    if key is None:
        console.print(f"[red]unknown paper {ident!r}")
        raise typer.Exit(1)
    m, row = db.get_meta(key), db.get_row(key)
    console.print(f"[bold cyan]{m.title}")
    au = "; ".join(m.authors[:6]) + (" ..." if len(m.authors) > 6 else "")
    console.print(f"{au} ({m.year})  {m.pub or ''}")
    t = Table(show_header=False)
    for k_, v in [("paper_key", key), ("bibcode", m.bibcode), ("arXiv", m.arxiv_id), ("DOI", m.doi),
                  ("doctype", m.doctype), ("refereed", m.refereed), ("citations", m.citation_count),
                  ("references", len(m.references)), ("tier", row["tier"]),
                  ("priority", f"{row['priority']:.2f}"), ("status", row["status"]),
                  ("source", row["source"]), ("n_chunks", row["n_chunks"]), ("error", row["error"])]:
        t.add_row(k_, str(v))
    console.print(t)
    doc = load_document(key, db)
    if doc is not None:
        console.print(f"[bold]sections[/] ({len(doc.full_text):,} chars, {len(doc.bibliography)} bib entries, "
                      f"flags {doc.quality_flags})")
        for s in doc.sections:
            console.print(f"{'  ' * (s.level - 1)}{s.id:10s} {s.title}  [dim]({len(s.paragraphs)} paras)")


@app.command()
def stats(disk: Annotated[bool, typer.Option("--disk/--no-disk", help="Measure disk usage.")] = True) -> None:
    """Corpus statistics (does not load any model)."""
    from galrag.service import corpus_stats

    s = corpus_stats(_db(), disk=disk)
    console.print(f"[bold]data dir:[/] {s['data_dir']}   papers: {s['papers_total']:,}   "
                  f"paper vectors: {s['with_paper_vector']:,}   chunks in LanceDB: {s['vector_chunks']}   "
                  f"documents: {s['documents']:,}")
    console.print(f"[bold]sqlite:[/] {s['sqlite_path']}   [bold]lancedb:[/] {s['lancedb_dir']}"
                  + (f"   [bold]work dir:[/] {s['work_dir']}" if s["work_dir"] else ""), highlight=False)
    _print_counts("by status", s["by_status"])
    _print_counts("by tier", s["by_tier"])
    if s["fulltext_embedded_by_tier"]:
        _print_counts("full-text embedded by tier", s["fulltext_embedded_by_tier"])
    for kind, r in s["last_runs"].items():
        console.print(f"last {kind}: {r['finished_at']}  new={r['n_new']} updated={r['n_updated']}"
                      + (f"  note={r['note']}" if r["note"] else ""))
    console.print(f"max entdate: {s['max_entdate']}")
    if s["top_failures"]:
        t = Table(title="top failure reasons", title_justify="left")
        t.add_column("status")
        t.add_column("n", justify="right")
        t.add_column("error")
        for f in s["top_failures"]:
            t.add_row(f["status"], str(f["n"]), f["error"])
        console.print(t)
    if disk:
        _print_counts("disk usage", s["disk"])


@app.command(name="eval")
def eval_(
    queries: Annotated[Path, typer.Option(help="YAML query set.")] = Path("tests/queries.yaml"),
    k: Annotated[int, typer.Option("-k")] = 10,
    mode: Annotated[str, typer.Option()] = "hybrid",
    rerank: Annotated[bool, typer.Option("--rerank/--no-rerank")] = True,
    papers: Annotated[bool, typer.Option("--papers/--no-papers", help="Also evaluate paper search.")] = True,
    json_out: Annotated[Path | None, typer.Option(help="Write full results as JSON.")] = None,
) -> None:
    """Retrieval evaluation: recall@k / MRR over tests/queries.yaml."""
    from galrag.evaluate import load_queries, run_eval

    qs = load_queries(queries)
    res = run_eval(_store(), _db(), qs, k=k, mode=mode, rerank=rerank, papers=papers)
    t = Table(title=f"retrieval eval (k={k}, mode={mode}, rerank={rerank})", title_justify="left")
    for c in ("query", "found", "paper", "missing (not in corpus / abstract only)", "RR"):
        t.add_column(c)
    for r in res["rows"]:
        n = len(r["expected"])
        miss = [b for b in r["expected"] if b not in r["found"]]
        tag = [b + (" (NC)" if b in r["not_in_corpus"] else " (abs)" if b in r["abstract_only"] else "")
               for b in miss]
        t.add_row(r["query"][:60], f"{len(r['found'])}/{n}", f"{len(r['paper_found'])}/{n}",
                  ", ".join(tag), f"{r['rr']:.2f}")
    console.print(t)
    _print_counts("summary", {k_: (f"{v:.3f}" if isinstance(v, float) else v)
                              for k_, v in res["summary"].items()})
    if json_out:
        json_out.write_text(json.dumps(res, indent=1, default=str))


# ------------------------------------------------------------------ storage
@app.command(name="migrate-docs")
def migrate_docs(
    docs_dir: Annotated[Path | None, typer.Option("--docs-dir",
                                                  help="Legacy JSON dir (default <data_dir>/docs).")] = None,
    delete_files: Annotated[bool, typer.Option("--delete-files",
                                               help="Delete each JSON file once its document is in SQLite.")] = False,
) -> None:
    """Import legacy docs/*.json[.gz] files into the SQLite documents table."""
    from galrag.config import get_settings

    db = _db()
    d = docs_dir or get_settings().docs_dir
    t0 = time.time()
    n = db.import_document_files(d)
    total = db.count_documents()
    console.print(f"[bold]migrate-docs:[/] imported {n:,} from {d}; documents table now holds {total:,} "
                  f"({time.time() - t0:.0f} s)", highlight=False)
    if delete_files and d.is_dir():
        from galrag.parse import fs_safe

        removed = 0
        for key in db.iter_document_keys():
            for suffix in (".json", ".json.gz"):
                p = d / (fs_safe(key) + suffix)
                if p.exists():
                    p.unlink()
                    removed += 1
        console.print(f"deleted {removed:,} migrated files")


sync_app = typer.Typer(help="Move results between local scratch (work_dir) and the shared data_dir.",
                       no_args_is_help=True)
app.add_typer(sync_app, name="sync")


def _only(only: str | None) -> dict[str, bool]:
    if only is None:
        return {"sqlite": True, "lancedb": True}
    if only not in ("sqlite", "lancedb"):
        raise typer.BadParameter("--only must be sqlite or lancedb")
    return {"sqlite": only == "sqlite", "lancedb": only == "lancedb"}


@sync_app.command("push")
def sync_push(only: Annotated[str | None, typer.Option(help="sqlite | lancedb")] = None) -> None:
    """Local work_dir/lancedb -> data_dir (SQLite via the backup API, LanceDB via rsync)."""
    from galrag import sync
    from galrag.config import get_settings

    console.print_json(json.dumps(sync.push(get_settings(), **_only(only)), default=str))


@sync_app.command("pull")
def sync_pull(only: Annotated[str | None, typer.Option(help="sqlite | lancedb")] = None,
              force: Annotated[bool, typer.Option(help="Overwrite an existing local SQLite.")] = False) -> None:
    """data_dir -> local work_dir/lancedb (refuses to overwrite a local SQLite without --force)."""
    from galrag import sync
    from galrag.config import get_settings

    try:
        res = sync.pull(get_settings(), force=force, **_only(only))
    except FileExistsError as e:
        console.print(f"[red]{e}")
        raise typer.Exit(1) from None
    console.print_json(json.dumps(res, default=str))


@app.command()
def reconcile() -> None:
    """Set embedded/n_chunks and paper_vec/abs_chunk in SQLite from the LanceDB contents."""
    db, store = _db(), _store()
    t0 = time.time()
    res = db.reconcile_from_store(store)
    console.print(f"[bold]reconcile:[/] {res} ({time.time() - t0:.0f} s)", highlight=False)


# ------------------------------------------------------------------ MCP
@app.command()
def serve() -> None:
    """Run the MCP server on stdio (what MCP clients launch)."""
    from galrag.server import main

    main()


@app.command(name="mcp-config")
def mcp_config() -> None:
    """Print how to register the MCP server with Claude Code and other MCP clients."""
    from galrag.config import get_settings

    exe = shutil.which("galrag-mcp") or str(Path(sys.executable).parent / "galrag-mcp")
    env = {"GALRAG_DATA_DIR": str(get_settings().data_dir)}
    env_flags = " ".join(f"-e {k}={v}" for k, v in env.items())
    console.print("[bold]Claude Code[/] (user scope; drop `-s user` for the current project only):")
    console.print(f"  claude mcp add -s user galrag {env_flags + ' ' if env_flags else ''}-- {exe}",
                  highlight=False, soft_wrap=True)
    console.print("\n[bold]Generic MCP client config[/] (Claude Desktop, Cursor, ...):")
    cfg = {"mcpServers": {"galrag": {"command": exe, "args": [], **({"env": env} if env else {})}}}
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    app()
