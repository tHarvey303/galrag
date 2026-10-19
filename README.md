# galrag

A local, citation-verifiable retrieval system over the galaxy-evolution / high-redshift
literature, served to LLM clients (Claude Code, Claude Desktop, any MCP host) through an
MCP server.

* **Metadata** for every matching paper from NASA ADS / SciX (abstracts, authors,
  citation counts and reference lists) in SQLite.
* **Full text** from arXiv (LaTeXML HTML, then ar5iv, then PDF, then open-access
  publisher PDFs), parsed into sections and paragraphs. Every paragraph and chunk stores
  exact character offsets into the document text, so `full_text[char_start:char_end]`
  reproduces a passage exactly.
* **Hybrid search** in LanceDB: BM25 full-text search plus `BAAI/bge-base-en-v1.5`
  dense vectors over passages and over title+abstract, fused with reciprocal-rank fusion
  and a mild citation prior, then reranked with the `bge-reranker-v2-m3` cross-encoder
  (see "Ranking" below).
* **Verification tools**: locate quotes verbatim (`get_quote`), check claims
  (`verify_claim`), find where paper A cites paper B (`citation_contexts`), resolve
  free-text citations, and export BibTeX.

The design and its rationale are in [docs/PLAN.md](docs/PLAN.md).

## Install

```bash
conda create -n galrag -c conda-forge --override-channels python=3.12
conda activate galrag
git clone <this repo> galrag && cd galrag
pip install -e ".[dev]"
```

This installs two console scripts: `galrag` (the CLI) and `galrag-mcp` (the stdio MCP server).

### Configuration (`.env`)

Settings are read from environment variables prefixed `GALRAG_`, or from a `.env`
file in the working directory or the repo root. Keep `.env` out of git; it is already
listed in `.gitignore`.

```dotenv
# NASA ADS API token: https://ui.adsabs.harvard.edu/user/settings/token
GALRAG_ADS_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Where the corpus archive lives (default: <repo>/data); on shared/network storage
GALRAG_DATA_DIR=/path/to/galrag/data
# Machine-local scratch for the live SQLite and the raw HTML/PDF cache (see "Storage")
GALRAG_WORK_DIR=/tmp/$USER/galrag
# LanceDB location if not <data_dir>/lancedb (e.g. local disk on the GPU machine)
# GALRAG_LANCEDB_DIR=/scratch/galrag/lancedb
# optional
GALRAG_CONTACT=you@example.org          # added to the HTTP User-Agent
GALRAG_EMBED_WORKERS=6                  # embedding processes
GALRAG_EMBED_THREADS_PER_WORKER=24
GALRAG_RERANK_ENABLED=true
```

You can tune the corpus definition (ADS `topic_query`, `year_min`, tier thresholds),
the chunk sizes, the models and the crawl delays in `galrag/config.py` or through the
matching `GALRAG_*` variables.

## Building the corpus

Run the stages in this order. Each one is idempotent and resumable, because progress
is tracked in each paper's `status` column.

```bash
galrag harvest                 # ADS metadata + reference lists for the topic query -> SQLite
galrag fetch --tiers A,B       # full text (arXiv HTML -> ar5iv -> PDF), parsed into SQLite
galrag embed --workers 6 --threads 24   # full-text chunks + title/abstract vectors -> LanceDB
galrag index                   # FTS / scalar / IVF-PQ indexes (embed already does this)
galrag stats                   # counts per status/tier, failures, disk usage
```

`fetch` waits 4 s between arXiv requests (configurable), so a large backlog takes days.
`scripts/run_fetch_parse.py` runs it detached and pushes a SQLite snapshot to the data
dir every `--sync-interval` seconds (default 1800) and at exit. You can run `embed`
repeatedly while fetching continues: each run embeds the parsed papers that have no
full-text chunks in LanceDB yet.

### Keeping it up to date and adding papers

```bash
galrag update                  # ADS records entered since the last run; fetch/parse/embed new tier A/B
galrag update --no-process     # metadata only
galrag ingest 2405.03781 2023A&A...677A..88B 10.3847/2041-8213/ad4cf3   # any year / doctype
galrag ingest "query:author:\"^Bunker\" year:2023"  --max-records 10
```

`ingest` resolves each identifier through ADS. It does not apply the corpus year or
doctype filters, and it then fetches, parses and embeds the paper. If no full text is
available, it embeds the abstract only.

### Querying from the shell

```bash
galrag search "UV luminosity function at z>10 with JWST" -k 5 --year-from 2022
galrag search "GN-z11" --mode fts --no-rerank
galrag search "dust in Lyman break galaxies" --papers          # paper-level search
galrag paper 2302.07256                                         # metadata + section outline
galrag quote 2023A&A...677A..88B "nitrogen abundance"           # locate a quote
galrag eval -k 10 --no-rerank                                   # recall@k / MRR on tests/queries.yaml
```

## MCP server

### Claude Code

```bash
claude mcp add -s user galrag -e GALRAG_DATA_DIR=/abs/path/to/data -- /abs/path/envs/galrag/bin/galrag-mcp
```

`galrag mcp-config` prints this command with the right paths for your environment. Use
`-s project` or omit `-s` to register the server for one project only.

### Other MCP clients (Claude Desktop, Cursor, ...)

```json
{
  "mcpServers": {
    "galrag": {
      "command": "/abs/path/envs/galrag/bin/galrag-mcp",
      "args": [],
      "env": {"GALRAG_DATA_DIR": "/abs/path/to/data"}
    }
  }
}
```

The server uses stdio. It logs to stderr (warnings only) and to `data/logs/galrag.log`.
The first semantic query loads the embedding model, which can take a while on a cold
network filesystem.

### Tools

| Tool | What it does |
|---|---|
| `search_passages` | Hybrid / vector / FTS passage search with year, author, refereed, paper, kind and citation filters. Every hit includes provenance and a citation string. |
| `search_papers` | Paper-level search over title+abstract vectors. Covers every paper, including those without full text. |
| `find_papers` | Metadata lookup by first author, year and title substring. |
| `get_paper` | Metadata, abstract, status, full-text source, section outline and ADS/arXiv links. |
| `read_section` | One section's paragraphs, with paragraph ids and offsets. |
| `read_paragraphs` | Paragraphs by id, or the exact span `full_text[a:b]` with optional context. |
| `get_quote` | Locate a quote. It tries an exact (whitespace/case-insensitive) match, then a fuzzy match, and returns the verbatim text, section, offsets, page and similarity. |
| `verify_claim` | The passages that best support a claim, each with the closest wording in that paper. |
| `citation_contexts` | The sentences in paper A that cite paper B, and whether B is in A's ADS reference list. |
| `resolve_citation` | Turn free text such as "Madau & Dickinson 2014" or an arXiv id or DOI into bibcode candidates, checking the local corpus first and then ADS. |
| `references_of` | A paper's reference list, with each entry flagged by whether it is in the corpus. |
| `citations_of` | Citing papers in the corpus, plus the live ADS list on request. |
| `get_bibtex` | BibTeX by bibcode, cached locally. |
| `ingest_paper` | Add a paper on demand: metadata, then full text, then embedding. |
| `update_corpus` | Incremental ADS update, then processing of the new papers. |
| `corpus_stats` | Counts per status and tier, index size, last runs and failure reasons. |

Resources: `paper://{ident}` returns metadata, the abstract and the section outline as
markdown, and `paper://{ident}/section/{section_id}` returns the text of one section.
Prompts: `literature_answer(question)` and `check_citation(claim, bibcode)`.

An `ident` can be an ADS bibcode, an arXiv id (`2405.03781` or `arXiv:2405.03781`), a
DOI, or the internal `paper_key`.

### Verification and citation workflow

1. `search_passages` (or `search_papers`) finds candidate evidence. Each hit carries
   `paper_key`, `bibcode`, `arxiv_id`, `chunk_id`, `section_path`, `char_start`,
   `char_end`, `page_start`, `source` and a `citation` string such as
   `"Bunker et al. (2023), 2023A&A...677A..88B"`.
2. `read_paragraphs(ident, para_ids=...)` or `read_paragraphs(ident, char_start=..., char_end=..., context_chars=300)`
   reads the surrounding text deterministically.
3. `get_quote(ident, text)` checks a quote before it is presented as verbatim. Use
   quotation marks only when the result has `exact=true`.
4. Cite by bibcode.

Hits with `kind="abstract"` or `source="abstract_only"` come from the ADS abstract, not
from the full text. For these, `verbatim_span` is false.

## Storage layout

### Shared archive vs local scratch

`/nexus` is shared network storage. Heavy I/O there (thousands of small files, a hot
database written by several loops) slows the whole platform and causes SQLite
"database is locked" stalls. galrag therefore splits its storage:

* **`GALRAG_DATA_DIR`** (on Nexus) is the *archive*: what the MCP server reads and what
  the loops push finished results into, in bulk.
* **`GALRAG_WORK_DIR`** (machine-local disk, optional) holds the hot files: the live
  SQLite and the raw HTML/PDF cache. The raw cache is never synced.
* **`GALRAG_LANCEDB_DIR`** (optional) moves LanceDB off the data dir. LanceDB writes a
  few large fragment files, which is fine on Nexus, so only the GPU machine sets it.

Without `GALRAG_WORK_DIR` everything lives in the data dir (single-machine layout).

```
$GALRAG_DATA_DIR/                 (Nexus, the archive)
  galrag.sqlite         papers, identifiers (bibcode/arXiv/DOI -> paper_key), refs, bibtex, runs,
                        documents (parsed Document per paper, gzipped JSON)
  lancedb/              chunks__<model> (vectors + FTS index), papers__<model> (title+abstract vectors)
  models/               Hugging Face cache (HF_HOME) for the embedding and reranker models
  logs/                 galrag.log (CLI, MCP server), fetch_parse.log, embed.log
$GALRAG_WORK_DIR/                 (local disk)
  galrag.sqlite         the live database the loops write (WAL)
  raw/html, raw/pdf     fetched raw files (cache; nothing is fetched twice on this machine)
```

Parsed documents live in the SQLite `documents` table, not in thousands of
`docs/*.json` files. To import an old `docs/` directory, run `galrag migrate-docs`
(`--docs-dir PATH`, and optionally `--delete-files` once you have checked the count).

### Sync commands

```bash
galrag sync pull                 # archive -> local: SQLite (backup API) + LanceDB (rsync); refuses
                                 # to overwrite an existing local SQLite without --force
galrag sync push                 # local -> archive: SQLite snapshot (backup API into a .tmp file,
                                 # then an atomic rename) + rsync -a --delete of a local LanceDB
galrag sync push --only sqlite   # or --only lancedb
galrag reconcile                 # set embedded/n_chunks, paper_vec/abs_chunk from the LanceDB contents
```

`push` never copies the live SQLite file or its `-wal`/`-shm` files. It writes one
consistent snapshot sequentially, so the archive is always a complete database. Treat the
archive SQLite as read-only: anything written to it directly, for example by the MCP
server's `ingest_paper`, is overwritten by the next push from the fetch node.

### Typical setup on this node (fetch/parse)

```bash
# .env: GALRAG_DATA_DIR=<repo>/data   GALRAG_WORK_DIR=/tmp/$USER/galrag
galrag sync pull                                   # once: local working copy of the SQLite
nohup python scripts/run_fetch_parse.py --tiers A,B >> data/logs/fetch_parse.stdout 2>&1 &
```

The fetch loop pushes `sqlite,lancedb` (`--push`) every `--sync-interval` seconds (default
1800) and again when it stops, including after SIGTERM/SIGINT. LanceDB is skipped when it
already lives in the data dir.

### Embedding on a GPU machine

Embedding needs SQLite only as a read-mostly snapshot. Which papers still need embedding
is decided from LanceDB itself: full-text candidates are papers with status
`parsed`/`embedded` that have a document but no full-text chunks in the store, and
paper-vector candidates are papers missing from the paper-level table. So the GPU machine
can work from a stale snapshot.

```bash
# same conda env (its torch is a CUDA wheel, which also runs on CPU-only nodes)
export GALRAG_DATA_DIR=/shared/path/to/galrag/data      # the shared archive
export GALRAG_WORK_DIR=/dev/shm/$USER/galrag            # local disk or tmpfs: only a snapshot lives here
galrag sync pull --only sqlite                           # SQLite snapshot (~11 s onto tmpfs)
# one pass (e.g. after `galrag update`):
python scripts/run_embed.py --push none --chunks-per-batch 20000
# or keep up with a running fetch loop; restart every 50 passes (see below):
nohup sh -c 'while python scripts/run_embed.py --loop --refresh-snapshot --push none \
      --chunks-per-batch 20000 --max-passes 50; do :; done' >> data/logs/embed.stdout 2>&1 &
```

This writes straight into the archive's LanceDB on Nexus (no `GALRAG_LANCEDB_DIR`), so
the MCP server and the fetch node see new chunks immediately and nothing has to be
rsynced. `--refresh-snapshot` re-pulls the SQLite snapshot before every pass after the
first (`galrag sync pull --only sqlite --force`) and reconciles it with the store, so
papers the fetch node parsed and pushed meanwhile get embedded without a restart and the
MCP server reading the snapshot reports correct statuses. `--push none` because the
snapshot must never overwrite the archive. `--chunks-per-batch 20000` keeps the H100 busy
and produces fewer LanceDB fragments (an H100 embeds bge-base at a few thousand chunks/s;
the loop is then bound by chunking and NFS writes, roughly 400-600 chunks/s end to end;
the whole tier A+B corpus, 4.5M chunks, took about four hours of GPU time).
`--max-passes` exists because LanceDB's full-table scans retain 100-200 MB per pass
inside the library (the process reached 68 GB after 378 passes); the shell loop restarts
it with fresh memory.

**Disk: run `galrag vacuum` after a long embedding run.** Every upsert, index build and
optimize creates a new LanceDB table version, and old vector-index builds are large.
`optimize()` only prunes versions older than 7 days, so a few days of embedding left
470 GB of dead index files next to a 55 GB table. `galrag vacuum` (keep only the current
version) reclaims them; any MCP server still holding an older version fails once and
reopens.

If you prefer a local LanceDB (`GALRAG_LANCEDB_DIR` on local disk), run
`galrag sync pull` and `--push lancedb` instead: the local LanceDB is then rsynced into
`$GALRAG_DATA_DIR/lancedb` every `--sync-interval` seconds and at exit.

Back on the fetch node, run `galrag reconcile` to record in the local SQLite what the
GPU machine embedded. The next push carries those statuses to the archive. Never run a
fetch loop on the GPU machine while it uses `--refresh-snapshot`: the refresh would
discard that loop's unpushed work.

## Paper status state machine

```
metadata ──fetch──> fetched ──parse──> parsed ──chunk+embed──> embedded
    │                  │                  │
    └> failed:fetch    └> failed:parse    └> failed:embed        (skipped: not wanted)
```

`galrag fetch --retry-failed` retries papers whose fetch or parse failed. Papers that
stay at `metadata`, mostly tier C, are still searchable through their title+abstract
vector and an abstract-only chunk.

## Ranking

`search_passages` (and `galrag search`) ranks in four steps; every knob is a `GALRAG_*`
setting in `galrag/config.py` and the defaults were chosen with `galrag eval` on
`tests/queries.yaml` (55 questions, 88 expected papers, all in the corpus with full text).

1. **Candidates.** `search_candidates` (100) passages from the dense index and 100 from
   BM25, plus the top 100 papers by title+abstract vector, each represented by its
   abstract chunk (`paper_rrf_weight`, 1.0). Reciprocal-rank fusion (`k = 60`) merges
   the three lists.
2. **Citation prior.** `citation_boost` (1.0) adds `log(1+citations)/log(1001)/60` to the
   fused score, so a 1000-citation paper gains as much as a rank-1 hit from one
   retriever. It favours the canonical paper on a topic over the many later papers that
   discuss it.
3. **Rerank.** The top `rerank_top_n` (100) passages go through the cross-encoder. With
   `rerank_blend` (1.0) the final order is an RRF of the reranker's rank and the fused
   rank, so the retrievers' agreement still counts; at 0 the reranker alone decides.
4. **Diversify.** At most `max_chunks_per_paper` (2) passages per paper in the top-k; the
   rest fill up behind. `read_section` / `read_paragraphs` give the rest of a paper.

Effect on the eval set (k = 10, distinct papers among the top-10 passages):

| configuration | recall@10 | hit rate@10 | MRR |
|---|---|---|---|
| before (bge-reranker-base, top-50, no fusion extras) | 0.533 | 0.636 | 0.462 |
| same, reranker off | 0.564 | 0.655 | 0.419 |
| paper-level fusion + 2-per-paper cap | 0.585 | 0.691 | 0.465 |
| + citation prior (0.5) | 0.609 | 0.727 | 0.489 |
| bge-reranker-v2-m3 alone | 0.597 | 0.691 | 0.498 |
| v2-m3 + fusion + cap + citation prior (1.0) | 0.667 | 0.800 | 0.548 |
| + rerank blend (current defaults) | 0.694 | 0.836 | 0.642 |

Two findings worth knowing: feeding the old `bge-reranker-base` more candidates made
results *worse* (it promoted passages from later papers that merely cite the expected
one), and the blended order beats trusting either the retrievers or the reranker alone.
The remaining misses are mostly questions whose "expected" papers are three canonical
ones among dozens of relevant ones, and questions phrased unlike the paper (e.g.
"dust-free outflows" for Ferrara et al. 2023). The reranker costs about 1 s per query
on a GPU; on a CPU-only MCP host set `GALRAG_RERANK_MODEL=BAAI/bge-reranker-base` and
`GALRAG_RERANK_TOP_N=50`, or `GALRAG_RERANK_ENABLED=false`.

## Known limitations

* **Tier C papers have abstracts only** unless you promote them with `galrag ingest` or
  `ingest_paper`. Full text is fetched in priority order, so recent, well-cited and
  refereed papers are searchable first.
* **PDF-derived text** (`source` = `arxiv_pdf` / `pub_pdf`) has a weaker section
  structure and occasional extraction artefacts such as ligatures, hyphenation and
  column breaks. Offsets are exact into `full_text`, but `full_text` itself is less
  clean than text from the LaTeXML HTML. Citation contexts in PDFs use a textual
  "Author et al. YEAR" fallback.
* **Embedding throughput is CPU-bound.** Expect roughly tens to a few hundred chunks
  per second, depending on node load and the workers × threads setting (see
  `scripts/bench_embed.py`). The first query after a cold start also has to load the
  model.
* Bibliography entries are linked to bibcodes heuristically against the ADS reference
  list. Unresolved entries are still matched by author and year in
  `citation_contexts`.
* The ADS API allows about 5000 requests a day. `resolve_citation`, `citations_of`
  with `local_only=false`, `get_bibtex` for uncached entries, `ingest_paper` and
  `update_corpus` all need `GALRAG_ADS_TOKEN`.

## Development

```bash
ruff check galrag scripts tests
pytest tests/test_parse.py tests/test_sync_docs.py   # parsing, documents table, sync, reconcile (no model)
pytest tests/test_server.py                          # loads the embedding model (slow cold)
pytest tests/test_ads.py                             # live ADS tests (need a token)
```
