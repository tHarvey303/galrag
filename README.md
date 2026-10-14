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
  dense vectors (CPU), fused with reciprocal-rank fusion, with an optional
  `bge-reranker-base` cross-encoder.
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
# Where the corpus lives (default: <repo>/data)
GALRAG_DATA_DIR=/path/to/galrag/data
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
galrag fetch --tiers A,B       # full text (arXiv HTML -> ar5iv -> PDF), parse to docs/*.json
galrag embed --workers 6 --threads 24   # full-text chunks + title/abstract vectors -> LanceDB
galrag index                   # FTS / scalar / IVF-PQ indexes (embed already does this)
galrag stats                   # counts per status/tier, failures, disk usage
```

`fetch` waits 4 s between arXiv requests (configurable), so a large backlog takes days.
`scripts/run_fetch_parse.py` runs it detached. You can run `embed` repeatedly while
fetching continues: each run picks up whatever has been parsed since the last one.

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

## Storage layout (`GALRAG_DATA_DIR`)

```
data/
  galrag.sqlite         papers, identifiers (bibcode/arXiv/DOI -> paper_key), refs, bibtex, runs (WAL)
  raw/html, raw/pdf     fetched raw files (cache; nothing is fetched twice)
  docs/<key>.json       parsed Document: sections, paragraphs, offsets, bibliography, full_text
  lancedb/              chunks__<model> (vectors + FTS index), papers__<model> (title+abstract vectors)
  models/               Hugging Face cache (HF_HOME) for the embedding and reranker models
  logs/galrag.log       rotating log (CLI and MCP server)
```

## Paper status state machine

```
metadata ──fetch──> fetched ──parse──> parsed ──chunk+embed──> embedded
    │                  │                  │
    └> failed:fetch    └> failed:parse    └> failed:embed        (skipped: not wanted)
```

`galrag fetch --retry-failed` retries papers whose fetch or parse failed. Papers that
stay at `metadata`, mostly tier C, are still searchable through their title+abstract
vector and an abstract-only chunk.

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
ruff check galrag tests
pytest tests/test_parse.py tests/test_server.py      # test_server loads the embedding model (slow cold)
pytest tests/test_ads.py                             # live ADS tests (need a token)
```
