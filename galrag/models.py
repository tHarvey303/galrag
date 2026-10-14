"""Pydantic data models shared by every stage of the pipeline.

Identifier conventions
----------------------
* ``paper_key``: the canonical key used everywhere (SQLite, docs/, LanceDB).
  It is the ADS bibcode when known (``2024ApJ...968..118P``) and otherwise
  ``arXiv:<id>`` (``arXiv:2405.03781``).  Use :func:`galrag.db.Database.resolve`
  to map any identifier (bibcode, arXiv id with or without version, DOI) to it.
* ``arxiv_id``: bare arXiv identifier without version (``2405.03781`` or
  ``astro-ph/0601001``).

Offsets
-------
Every ``Paragraph`` and ``Chunk`` carries ``char_start``/``char_end`` that index
into ``Document.full_text`` so that ``full_text[char_start:char_end]`` is the
exact text.  This is the basis for verifiable quotes.
"""

from __future__ import annotations

import html
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

FullTextSource = Literal["arxiv_html", "ar5iv", "arxiv_pdf", "pub_pdf", "abstract_only"]
ParagraphKind = Literal["title", "abstract", "body", "caption", "footnote", "equation", "list"]


class Status(StrEnum):
    """Ingestion state of a paper; stages are strictly ordered and idempotent."""

    METADATA = "metadata"        # harvested from ADS, no full text yet
    FETCHED = "fetched"          # raw HTML/PDF on disk
    PARSED = "parsed"            # docs/<key>.json written
    CHUNKED = "chunked"          # chunks computed (stored transiently in docs json sidecar)
    EMBEDDED = "embedded"        # chunks in LanceDB
    FAILED_FETCH = "failed:fetch"
    FAILED_PARSE = "failed:parse"
    FAILED_EMBED = "failed:embed"
    SKIPPED = "skipped"          # tier C / no full text wanted


class Tier(StrEnum):
    A = "A"
    B = "B"
    C = "C"


class PaperMeta(BaseModel):
    """One ADS record, normalised."""

    paper_key: str
    bibcode: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    first_author: str = ""
    year: int | None = None
    pubdate: str | None = None          # YYYY-MM-DD (ADS uses -00 for unknown)
    entdate: str | None = None          # date entered in ADS
    abstract: str = ""
    keywords: list[str] = Field(default_factory=list)
    arxiv_class: list[str] = Field(default_factory=list)
    bibstem: str | None = None
    pub: str | None = None
    volume: str | None = None
    page: str | None = None
    doctype: str | None = None
    refereed: bool = False
    esources: list[str] = Field(default_factory=list)
    citation_count: int = 0
    read_count: int = 0
    references: list[str] = Field(default_factory=list)   # cited bibcodes
    tier: Tier = Tier.C
    priority: float = 0.0
    raw: dict = Field(default_factory=dict)                 # the untouched ADS doc

    @field_validator("title", "abstract", mode="before")
    @classmethod
    def _unescape_html(cls, v):
        # ADS returns HTML entities (e.g. "z &gt; 8") in titles and abstracts
        return html.unescape(v) if isinstance(v, str) and "&" in v else v


class CiteMarker(BaseModel):
    """A citation occurrence inside a paragraph."""

    bib_keys: list[str]           # keys into Document.bibliography
    char_start: int               # offsets into full_text
    char_end: int
    text: str                     # rendered marker, e.g. "(Bunker et al., 2023)"


class Paragraph(BaseModel):
    id: str                       # e.g. "S2.P3", "S2.SS1.P1", "abs.P1", "fig3.cap"
    kind: ParagraphKind = "body"
    text: str
    char_start: int
    char_end: int
    page: int | None = None       # 1-based, PDFs only
    page_end: int | None = None   # set when a PDF paragraph crosses a page
    cites: list[CiteMarker] = Field(default_factory=list)


class Section(BaseModel):
    id: str                       # "S2", "S2.SS1", "abs", "A1"
    title: str
    level: int = 1                # 1 = section, 2 = subsection, ...
    paragraphs: list[Paragraph] = Field(default_factory=list)


class BibEntry(BaseModel):
    key: str                      # e.g. "bib.bib12"
    raw_text: str
    authors: list[str] = Field(default_factory=list)
    first_author: str | None = None
    year: int | None = None
    title: str | None = None
    journal: str | None = None
    volume: str | None = None
    page: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    bibcode: str | None = None    # resolved against the citing paper's ADS reference list


class Document(BaseModel):
    """Canonical parsed full text; stored gzipped in the SQLite ``documents`` table."""

    paper_key: str
    source: FullTextSource
    source_url: str | None = None
    fetched_at: datetime | None = None
    parser_version: str = "0"
    title: str = ""
    abstract: str = ""
    sections: list[Section] = Field(default_factory=list)
    bibliography: list[BibEntry] = Field(default_factory=list)
    full_text: str = ""
    n_pages: int | None = None
    quality_flags: list[str] = Field(default_factory=list)   # e.g. "no_sections", "ocr_suspect"

    # convenience ----------------------------------------------------------
    def iter_paragraphs(self):
        for s in self.sections:
            for p in s.paragraphs:
                yield s, p

    def paragraph(self, para_id: str) -> Paragraph | None:
        for _, p in self.iter_paragraphs():
            if p.id == para_id:
                return p
        return None

    def span(self, char_start: int, char_end: int) -> str:
        return self.full_text[char_start:char_end]


class Chunk(BaseModel):
    """A retrieval unit.  ``chunk_id`` = f"{paper_key}#{ordinal:04d}"."""

    chunk_id: str
    paper_key: str
    ordinal: int
    kind: ParagraphKind = "body"
    section_id: str
    section_title: str
    section_path: str             # "2 Observations > 2.1 NIRCam imaging"
    para_ids: list[str]
    text: str
    char_start: int
    char_end: int
    page_start: int | None = None
    page_end: int | None = None
    n_tokens: int = 0
    # denormalised metadata for filtering inside LanceDB
    year: int | None = None
    bibcode: str | None = None
    arxiv_id: str | None = None
    first_author: str = ""
    refereed: bool = False
    citation_count: int = 0
    source: FullTextSource = "abstract_only"


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    rank: int
    title: str = ""
    authors_short: str = ""       # "Bunker et al."
    scores: dict[str, float] = Field(default_factory=dict)   # vector / fts / rerank components


class QuoteMatch(BaseModel):
    """Result of locating a quote inside a Document."""

    paper_key: str
    found: bool
    exact: bool
    similarity: float             # 0-100 (rapidfuzz partial ratio)
    quote: str                    # the exact text from full_text
    char_start: int
    char_end: int
    section_id: str | None = None
    section_title: str | None = None
    para_id: str | None = None
    page: int | None = None
    source: FullTextSource | None = None
