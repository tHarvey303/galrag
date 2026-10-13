"""Full-text parsing: raw HTML/PDF -> canonical :class:`~galrag.models.Document`.

Public API
----------
* :func:`parse_fetched` – dispatch a :class:`~galrag.fetch.FetchResult` to the right parser.
* :func:`resolve_bibliography` – link ``BibEntry.bibcode`` using the citing paper's ADS
  reference list (and exact arXiv/DOI matches against the corpus).
* :func:`save_document` / :func:`load_document` – ``docs/<fs_safe(key)>.json[.gz]``.
* :func:`fs_safe` – filesystem-safe key.
"""

from __future__ import annotations

import gzip
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from galrag.config import get_settings
from galrag.models import BibEntry, Document, PaperMeta
from galrag.parse.bib import journal_bibstem, parse_bib_text, strip_accents

if TYPE_CHECKING:
    from galrag.db import Database
    from galrag.fetch import FetchResult

__all__ = [
    "doc_path",
    "fs_safe",
    "load_document",
    "parse_bibcode",
    "parse_fetched",
    "resolve_bibliography",
    "save_document",
]


def fs_safe(key: str) -> str:
    """Filesystem-safe version of a paper key (``/`` and ``:`` -> ``_``)."""
    from galrag.fetch import fs_safe as _fs

    return _fs(key)


# ----------------------------------------------------------------------- dispatch
def parse_fetched(result: FetchResult, meta: PaperMeta) -> Document:
    """Parse a fetched raw file into a :class:`Document` keyed by ``meta.paper_key``."""
    from galrag.fetch import read_raw_text

    if result.source in ("arxiv_html", "ar5iv"):
        from galrag.parse.html_latexml import parse_latexml_html

        doc = parse_latexml_html(read_raw_text(result.path), meta.paper_key, result.source,
                                 result.url, fetched_at=result.fetched_at)
    elif result.source in ("arxiv_pdf", "pub_pdf"):
        from galrag.parse.pdf import parse_pdf

        doc = parse_pdf(result.path, meta.paper_key, result.source, result.url)
        doc.fetched_at = result.fetched_at
    else:
        raise ValueError(f"cannot parse source {result.source!r}")
    if not doc.title and meta.title:
        doc.title = meta.title
    if not doc.abstract and meta.abstract:
        # keep the ADS abstract as metadata; it is *not* inserted into full_text
        doc.abstract = meta.abstract
        doc.quality_flags.append("abstract_from_ads")
    return doc


# ------------------------------------------------------------------ persistence
def doc_path(paper_key: str, *, gz: bool = False) -> Path:
    d = get_settings().docs_dir
    return d / (fs_safe(paper_key) + (".json.gz" if gz else ".json"))


def save_document(doc: Document, *, gz: bool = False) -> Path:
    """Write ``doc`` atomically to ``docs/<fs_safe(key)>.json`` (or ``.json.gz``)."""
    p = doc_path(doc.paper_key, gz=gz)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = doc.model_dump_json().encode("utf-8")
    if gz:
        data = gzip.compress(data, 6)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, p)
    other = doc_path(doc.paper_key, gz=not gz)        # avoid stale twins
    if other.exists():
        other.unlink()
    return p


def load_document(paper_key: str) -> Document | None:
    """Load a saved document (``.json`` or ``.json.gz``); None if not parsed yet."""
    for gz in (False, True):
        p = doc_path(paper_key, gz=gz)
        if p.exists():
            data = p.read_bytes()
            if gz:
                data = gzip.decompress(data)
            return Document.model_validate_json(data)
    return None


# ------------------------------------------------------------ bibliography links
_BIBCODE = re.compile(r"^(\d{4})(.{5})(.{4})(.)(.{4})(.)$")


def parse_bibcode(bibcode: str) -> dict | None:
    """Split an ADS bibcode ``YYYYJJJJJVVVVMPPPPA`` into its fields."""
    m = _BIBCODE.match(bibcode or "")
    if not m:
        return None
    year, stem, vol, qual, page, init = m.groups()
    out = {"year": int(year), "bibstem": stem.rstrip("."), "volume": vol.strip("."),
           "qualifier": qual, "page": page.strip("."), "initial": init.upper(), "arxiv_id": None}
    if qual.isdigit():                       # 5-digit pages overflow into the qualifier slot
        out["page"] = (qual + page).lstrip(".")
    if out["bibstem"] == "arXiv":
        digits = (vol + qual + page).replace(".", "")
        if len(digits) >= 8:
            out["arxiv_id"] = f"{digits[:4]}.{digits[4:]}"
    return out


def _surname_initial(name: str | None) -> str | None:
    if not name:
        return None
    s = strip_accents(name).strip()
    s = re.sub(r"^(?:the\s+)", "", s, flags=re.IGNORECASE)
    for ch in s:
        if ch.isalpha():
            return ch.upper()
    return None


def _norm_page(p: str | None) -> str | None:
    if not p:
        return None
    return re.sub(r"^[A-Za-z]+", "", p).lstrip("0") or None


def _score(entry: BibEntry, vol: str | None, page: str | None, ref: dict) -> float:
    """Evidence that ``entry`` is the ADS reference ``ref``; < 0 means incompatible."""
    if entry.arxiv_id and ref["arxiv_id"]:
        return 10.0 if entry.arxiv_id == ref["arxiv_id"] else -1.0
    if entry.year is None:
        return -1.0
    dy = abs(entry.year - ref["year"])
    if dy > 1:
        return -1.0
    ini = _surname_initial(entry.first_author)
    if ini and ini != ref["initial"]:
        return -1.0
    score = 1.0 if dy == 0 else 0.0
    stem = journal_bibstem(entry.journal)
    if stem and ref["bibstem"]:
        rs = ref["bibstem"]
        same = stem == rs or (stem == "ApJL" and rs == "ApJ") or (stem == "ApJ" and rs == "ApJL") \
            or (stem == "MNRAS" and rs == "MNRAS")
        if same:
            score += 1.0
        elif stem != "arXiv" and rs != "arXiv":
            return -1.0
    rv = ref["volume"]
    if vol and rv:
        if vol.lstrip("0") == rv.lstrip("0"):
            score += 2.0
        elif stem and stem not in ("arXiv",) and ref["bibstem"] != "arXiv":
            return -1.0
    rp = _norm_page(ref["page"])
    ep = _norm_page(page)
    if ep and rp and (ep == rp or (len(rp) == 4 and ep.endswith(rp))):   # bibcode pages: 4 chars
        score += 2.0
    if dy == 1 and score < 3.0:            # preprint vs published year: need hard evidence
        return -1.0
    return score


def resolve_bibliography(doc: Document, db: Database, meta: PaperMeta | None = None) -> int:
    """Fill ``BibEntry.bibcode`` for as many entries as possible; return the number resolved.

    Strategy, in order of reliability:

    1. exact arXiv id / DOI of the entry known in the corpus (``db.resolve``);
    2. arXiv id against ``YYYYarXivNNNNNNNNNX`` bibcodes in the ADS reference list;
    3. heuristic match against the citing paper's ADS reference list
       (``db.references_of``): same year (±1 with volume/page evidence), same
       first-author initial (last bibcode char), compatible journal bibstem,
       volume and page.  Ambiguous matches are left unresolved.
    """
    key = meta.paper_key if meta else doc.paper_key
    refs = db.references_of(key)
    parsed = [(b, parse_bibcode(b)) for b in refs]
    parsed = [(b, p) for b, p in parsed if p]
    n = 0
    for e in doc.bibliography:
        if e.bibcode:
            n += 1
            continue
        # 1. exact identifiers against the corpus
        for ident in ([f"arXiv:{e.arxiv_id}"] if e.arxiv_id else []) + ([f"doi:{e.doi}"] if e.doi else []):
            pk = db.resolve(ident)
            if pk:
                row = db.get_row(pk)
                if row is not None and row["bibcode"]:
                    e.bibcode = row["bibcode"]
                    break
        if e.bibcode:
            n += 1
            continue
        if not parsed:
            continue
        f = parse_bib_text(e.raw_text)
        vol, page = f.get("volume"), f.get("page")
        scored = sorted(((s, b) for b, p in parsed if (s := _score(e, vol, page, p)) >= 0),
                        reverse=True)
        if not scored:
            continue
        best, bib = scored[0]
        runner = scored[1][0] if len(scored) > 1 else -1.0
        if best >= 2.0 and best > runner:
            e.bibcode = bib
        elif best >= 1.0 and len(scored) == 1 and _surname_initial(e.first_author):
            e.bibcode = bib          # the only reference with that year+initial
        if e.bibcode:
            n += 1
    return n
