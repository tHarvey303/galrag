# galrag — MCP-based RAG over the high-redshift galaxy-evolution literature

**Status:** v0.1 design + implementation plan (2026-10-13). This document is the
source of truth for architecture decisions; module docstrings cover the details.

## 1. Goals

1. Build a *local*, citable corpus of the galaxy-evolution literature with an
   emphasis on high redshift (z ≳ 3: reionization, cosmic dawn, cosmic noon,
   LBG/LAE populations, JWST results), including bibliographic metadata and
   full text.
2. Expose the corpus to LLM clients (Claude Code, Claude Desktop, any MCP host)
   through an MCP server whose tools are designed for **verifiable** answers:
   every retrieved passage carries its bibcode/arXiv id, section, paragraph,
   page (for PDFs) and exact character offsets so a direct quote can be
   reproduced and checked.
3. Support claim/citation checking: "does paper X actually say Y?", "which
   papers support Y?", "does X cite Z and in what context?".
4. Be incrementally updatable: new arXiv postings and new ADS records can be
   ingested at any time without rebuilding the index.
5. Run entirely on CPU on this cluster (no GPU, no Docker), install with
   `pip install -e .` inside a conda env, and be versioned with git.

Non-goals (v0.1): answer generation inside the server (the MCP *client* LLM
reasons over retrieved evidence), OCR of scanned PDFs, figure/table
understanding, publisher-HTML scraping.

## 2. Data sources and why

| Need | Source | Notes |
|---|---|---|
| Bibliographic metadata, abstracts, keywords, citation counts, **reference lists** | NASA ADS / SciX API (`/v1/search/query`) | 5000 req/day; 2000 rows/page. The `reference` field gives the cited bibcodes, which is what makes citation verification possible offline. |
| BibTeX | ADS export service (`/v1/export/bibtex`) | Batched, cached in SQLite. |
| Full text (preferred) | `https://arxiv.org/html/<id>` (native LaTeXML HTML, served for recent *and* many older papers) | Structured: sections, paragraphs, `\cite` markers linked to bibliography entries, equations, footnotes. |
| Full text (fallback 1) | `https://ar5iv.labs.arxiv.org/html/<id>` | Same LaTeXML markup for older papers. |
| Full text (fallback 2) | `https://arxiv.org/pdf/<id>` via PyMuPDF | Page-aware, two-column aware via `pymupdf4llm`; weaker section structure. |
| Full text (fallback 3) | Publisher PDF through the ADS link gateway when open access | Only when no arXiv version exists. |
| New-paper discovery | ADS `entdate:[...]` queries on the same topic query; optional arXiv RSS/OAI for same-day coverage | `galrag update`. |

Crawl etiquette: one fetch worker per host, configurable inter-request delay
(default 4 s for arxiv.org, matching arXiv's "one request every few seconds"
guidance; the robots.txt generic crawl-delay is 15 s and the setting can be
raised to that for unattended bulk runs), identifying User-Agent, exponential
backoff on 429/5xx, raw responses cached on disk so nothing is fetched twice.

## 3. Corpus definition

The ADS query is configuration, not code (`galrag/config.py` → `TOPIC_QUERY`).
Measured sizes on 2026-10-13:

| Query | numFound |
|---|---|
| topic ∧ refereed ∧ 2000–2026 | 60 128 |
| topic ∧ refereed ∧ 2015–2026 | 36 096 |
| topic ∧ refereed ∧ citations ≥ 10 | 44 404 |
| topic ∧ 2023–2026 (incl. eprints) | 18 258 |

where *topic* = `abs:(galaxy OR galaxies) AND abs:("high redshift" OR
"high-redshift" OR "early universe" OR reionization OR "cosmic dawn" OR
"cosmic noon" OR "Lyman break" OR "Lyman-break" OR "Lyman alpha emitter" OR
JWST OR "epoch of reionization")`, restricted to `collection:astronomy`.

Tiering (all tiers get metadata + abstract; full text is fetched in priority
order so the most useful papers are searchable first):

* **Tier A (full text first):** refereed, 2015–2026, plus reviews (ARA&A,
  A&ARv, Phys. Rep.) at any year, plus anything with ≥ 50 citations.
* **Tier B:** remaining refereed 2000–2014 and recent un-refereed eprints
  (2024–2026, these are the JWST-era results people ask about).
* **Tier C (metadata only unless requested):** everything else matched.

Priority score for the fetch queue = `log1p(citation_count) + 0.4·(year−2000)/26
+ 2·is_review + 1·is_tierA`. Any paper can be promoted on demand via the
`ingest_paper` tool.

## 4. Storage layout (`GALRAG_DATA_DIR`, default `<repo>/data`)

```
data/
  galrag.sqlite            metadata, reference lists, bibtex, ingestion state (WAL mode)
  raw/html/<arxiv_id>.html raw fetched HTML (gzip)      raw/pdf/<id>.pdf
  docs/<paper_key>.json    canonical parsed Document (see §5)
  lancedb/                 LanceDB tables: chunks (vector + FTS), papers (vector of title+abstract)
  models/                  HF cache for embedding / reranker models
  logs/
```

`paper_key` = bibcode when known, else `arXiv:<id>`. The SQLite `papers`
table maps bibcode ⇄ arXiv id ⇄ DOI so any identifier resolves.

## 5. Canonical document model (`galrag/models.py`)

```
Document
  paper_key, source (arxiv_html | ar5iv | arxiv_pdf | pub_pdf), fetched_at, parser_version
  title, abstract
  sections: [Section(id, title, level, paragraphs: [Paragraph(id, text, char_start, char_end, page?, cites: [CiteMarker])])]
  bibliography: [BibEntry(key, raw_text, authors?, year?, arxiv_id?, doi?, bibcode?)]
  full_text: str          # concatenation; all char offsets index into this
  figures/tables captions kept as paragraphs flagged kind="caption"
```

Offsets are *into `full_text`*, so a quote is `full_text[a:b]` and can be
re-verified byte-for-byte. For PDFs, each paragraph also records its page.

Citation markers in LaTeXML HTML (`<cite class="ltx_cite">` → `#bib.bibN`)
are resolved to `BibEntry` and, where possible, to a bibcode by matching the
entry's arXiv id / DOI / (first author, year, journal) against the ADS
reference list of the citing paper. This gives *citation contexts* for free.

## 6. Chunking

Section-aware, paragraph-based: paragraphs are packed into chunks of ~350
tokens with a 1-paragraph overlap; paragraphs longer than the limit are split
at sentence boundaries. Every chunk stores `paper_key, section_title,
section_path, para_ids, char_start, char_end, page_start, page_end, n_tokens`
and a `kind` (body | abstract | caption | title). Title+abstract always form
their own chunk so abstract-only papers (tier C) are still searchable.

## 7. Embeddings and index

* Embedding model: `BAAI/bge-base-en-v1.5` (768-d, 512-token window, no
  `trust_remote_code`), run with sentence-transformers on CPU. Query prefix
  "Represent this sentence for searching relevant passages: " as the model
  card recommends. Configurable; the table name includes the model name so
  re-embedding with another model is additive.
* Throughput plan on this node (192 cores, no GPU): N worker processes ×
  M intra-op threads, batch 64, fp32; `scripts/bench_embed.py` picks N×M.
  Budget: ~2 M chunks for tier A+B.
* Vector store: **LanceDB** (embedded, Arrow-native, no server/Docker,
  works on the NFS mount). `chunks` table with IVF-PQ ANN index plus a
  tantivy full-text index on `text` → hybrid search with reciprocal-rank
  fusion, optional cross-encoder rerank (`BAAI/bge-reranker-base`) of the top 50.
* Why hybrid: astronomy queries contain identifiers (`GN-z11`, `MoM-z14`,
  `[O III] 88 µm`, `z ∼ 7.5`) that dense models blur; BM25 keeps them exact.
* A second small table `papers` embeds title+abstract for paper-level
  similarity and for "find the paper for this citation" queries.

## 8. MCP server (`galrag serve`, stdio; FastMCP from the official `mcp` SDK)

Tools (all return JSON with provenance fields):

| Tool | Purpose |
|---|---|
| `search_passages(query, k, year_from, year_to, authors, refereed_only, paper_keys, section_filter, rerank)` | Hybrid retrieval → passages with paper metadata and exact offsets. |
| `search_papers(query, k, filters)` | Paper-level retrieval over title+abstract. |
| `get_paper(id)` | Metadata, abstract, available sections, source/quality flags. |
| `read_section(id, section)` / `read_paragraphs(id, para_ids)` | Deterministic reads for the LLM to quote from. |
| `get_quote(id, text)` | Locate a (possibly paraphrased) string in the paper; return the exact span, section, page, and a similarity score. The verifiable-quote primitive. |
| `verify_claim(claim, paper_key?, k)` | Retrieve the best supporting/contradicting passages (corpus-wide or within one paper) for the client to judge. |
| `citation_contexts(citing, cited)` | Does paper A cite paper B (ADS reference list)? If so return the sentences where it does. |
| `resolve_citation(text)` | "Finkelstein et al. 2023" / arXiv id / DOI → bibcode candidates via ADS. |
| `references_of(id)` / `citations_of(id)` | Reference list (local) / citing papers (ADS live). |
| `bibtex(ids)` | BibTeX entries (cached). |
| `ingest_paper(id)` / `ingest_query(ads_query, max_papers)` / `update_corpus(since)` | Grow the corpus from inside the client. |
| `corpus_stats()` | Counts per tier/status, index freshness. |

Resources: `paper://<key>` (full Document JSON), `paper://<key>/bibtex`.
Prompts: `cite_check` and `literature_summary` templates that instruct the
client to quote with `get_quote` and to cite bibcodes.

## 9. Pipeline and CLI

`galrag` (typer):
`harvest` (ADS → SQLite) · `fetch` (full text, resumable queue) · `parse` ·
`chunk` · `embed` · `index` · `run-all` · `update` (new papers since last
run) · `ingest <ids>` · `search "<q>"` (debug) · `stats` · `serve`.

Each paper row carries `status` ∈ {metadata, fetched, parsed, chunked,
embedded, failed:<stage>} and the pipeline stages are idempotent, so a crash
or a new model just resumes. Stages that call the network are serial per host;
parsing/chunking/embedding are multiprocess.

## 10. Verification and quality

* Unit tests on parsers with fixture HTML/PDF; round-trip test that
  `full_text[a:b] == paragraph.text`.
* Retrieval smoke-test set (`tests/queries.yaml`): ~30 queries with expected
  bibcodes (e.g. UV luminosity function at z∼10 → Finkelstein+23, Donnan+24;
  GN-z11 → Bunker+23; little red dots → Matthee+24; Balmer breaks at z>8 →
  Looser+23). Recall@10 reported by `galrag eval`.
* Every tool output includes `source` and `parser_version` so the client can
  down-weight PDF-derived text.

## 11. Milestones

1. Skeleton, config, SQLite schema, ADS harvest (metadata for the full
   topic set).  ← today
2. Fetch + parse (HTML then PDF), chunking, embeddings, LanceDB, CLI.
3. MCP server + quote/citation tools; register with Claude Code.
4. Background bulk ingestion (tier A first), eval set, docs.
5. Scheduled `galrag update` (cron/SLURM) and reranker tuning.

## 12. Open decisions / later

* Swap embedding model to a long-context one (nomic-embed-text-v1.5 or
  bge-m3) if GPU time becomes available; the table-per-model design allows it.
* Parse LaTeX sources (`/e-print/`) for papers with neither HTML nor clean PDF.
* Publisher OA HTML (IOP/A&A) as an additional full-text source.
* Entity index (object names via SIMBAD/NED regex) for exact-match filters.
