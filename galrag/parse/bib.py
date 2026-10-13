"""Best-effort parsing of free-text bibliography entries.

Used by both the LaTeXML and the PDF parsers.  Everything here is heuristic;
the authoritative link to ADS is made later by
:func:`galrag.parse.resolve_bibliography` using the citing paper's ADS
reference list.
"""

from __future__ import annotations

import re
import unicodedata

from galrag.models import BibEntry

_ARXIV_NEW = re.compile(r"(?:arXiv\s*[:/]?\s*|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
_ARXIV_BARE = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d.])")
_ARXIV_OLD = re.compile(r"\b((?:astro-ph|gr-qc|hep-ph|hep-th|physics|cond-mat|nucl-th|math|quant-ph|"
                        r"hep-ex|nlin|cs)(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?")
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.IGNORECASE)
_YEAR = re.compile(r"(?<![\d./])((?:19[0-9]|20[0-4])\d)([a-z])?(?![\d/])")
_URL = re.compile(r"https?://\S+")

# Common journal names/abbreviations -> ADS bibstem.  Keys are normalised by _jkey().
JOURNAL_BIBSTEMS = {
    "apj": "ApJ", "astrophysicaljournal": "ApJ", "theastrophysicaljournal": "ApJ",
    "apjl": "ApJL", "apjletters": "ApJL", "astrophysicaljournalletters": "ApJL",
    "theastrophysicaljournalletters": "ApJL",
    "apjs": "ApJS", "astrophysicaljournalsupplement": "ApJS",
    "astrophysicaljournalsupplementseries": "ApJS", "theastrophysicaljournalsupplementseries": "ApJS",
    "mnras": "MNRAS", "monthlynoticesoftheroyalastronomicalsociety": "MNRAS",
    "mnrasl": "MNRAS", "a&a": "A&A", "aa": "A&A", "astronomyandastrophysics": "A&A",
    "astronomy&astrophysics": "A&A", "a&as": "A&AS", "aj": "AJ", "astronomicaljournal": "AJ",
    "theastronomicaljournal": "AJ", "nature": "Natur", "natur": "Natur", "natureastronomy": "NatAs",
    "natastron": "NatAs", "natas": "NatAs", "science": "Sci", "sci": "Sci", "pasp": "PASP",
    "pasj": "PASJ", "pasa": "PASA", "ara&a": "ARA&A", "araa": "ARA&A", "a&arv": "A&ARv",
    "jcap": "JCAP", "prd": "PhRvD", "physrevd": "PhRvD", "physicalreviewd": "PhRvD",
    "prl": "PhRvL", "physrevlett": "PhRvL", "physicalreviewletters": "PhRvL",
    "physrep": "PhR", "physicsreports": "PhR", "newar": "NewAR", "newastronomyreviews": "NewAR",
    "newa": "NewA", "ssrv": "SSRv", "spacescirev": "SSRv", "spacesciencereviews": "SSRv",
    "aap": "A&A", "apss": "Ap&SS", "ap&ss": "Ap&SS", "joss": "JOSS",
    "journalofopensourcesoftware": "JOSS", "spie": "SPIE", "procspie": "SPIE",
    "arxiv": "arXiv", "arxive-prints": "arXiv", "arxivpreprint": "arXiv", "rnaas": "RNAAS",
    "researchnotesoftheaas": "RNAAS", "baas": "BAAS", "aas": "AAS", "nat": "Natur",
    "natastronomy": "NatAs", "openjastrophys": "OJAp", "theopenjournalofastrophysics": "OJAp",
    "ojap": "OJAp", "annualreviewofastronomyandastrophysics": "ARA&A",
    "publicationsoftheastronomicalsocietyofthepacific": "PASP",
    "publicationsoftheastronomicalsocietyofjapan": "PASJ",
}


def journal_bibstem(journal: str | None) -> str | None:
    """Map a journal name/abbreviation to an ADS bibstem (None if unknown)."""
    if not journal:
        return None
    k = re.sub(r"[\s.,~\\{}()]", "", journal).lower()
    if k in JOURNAL_BIBSTEMS:
        return JOURNAL_BIBSTEMS[k]
    k2 = k.replace("the", "", 1) if k.startswith("the") else k
    return JOURNAL_BIBSTEMS.get(k2)


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


_INITIALS = re.compile(r"^(?:[A-Z][a-z]?\.?[\s\-~]*)+(?=[A-Z][\w'’\-]+)")


def _surname_from_name(name: str) -> str:
    """'J. C. Cuevas' -> 'Cuevas'; 'Di Ventra, M.' -> 'Di Ventra'; 'M. Di Ventra' -> 'Di Ventra'."""
    name = name.strip().strip(",;:").strip()
    if not name:
        return ""
    if "," in name:
        return name.split(",", 1)[0].strip()
    m = _INITIALS.match(name)
    if m and m.end() < len(name):
        return name[m.end():].strip()
    return name


def parse_bib_text(raw: str, hrefs: list[str] | None = None) -> dict:
    """Extract ``first_author, authors, year, title, journal, arxiv_id, doi, volume, page``."""
    out: dict = {}
    hay = raw + " " + " ".join(hrefs or [])
    m = _ARXIV_NEW.search(hay)
    if m:
        out["arxiv_id"] = m.group(1)
    else:
        m = _ARXIV_OLD.search(hay)
        if m:
            out["arxiv_id"] = m.group(1)
    for h in hrefs or []:
        hm = re.search(r"doi\.org/(10\.\S+)", h)
        if hm:
            out["doi"] = hm.group(1).rstrip(".,;")
            break
    if "doi" not in out:
        dm = _DOI.search(raw)
        if dm:
            out["doi"] = dm.group(1).rstrip(".,;)")
    if out.get("doi", "").lower().startswith("10.48550/arxiv."):
        out.setdefault("arxiv_id", out["doi"].split("arxiv.", 1)[-1].split("arXiv.")[-1])
        del out["doi"]

    # remove identifiers / URLs before looking for the year
    clean = _URL.sub(" ", raw)
    clean = _DOI.sub(" ", clean)
    clean = _ARXIV_NEW.sub(" ", clean)
    clean = _ARXIV_OLD.sub(" ", clean)
    clean = re.sub(r"\bdoi:\s*", " ", clean, flags=re.IGNORECASE)
    ym = _YEAR.search(clean)
    if ym:
        out["year"] = int(ym.group(1))

    # physics style: "A. Author and B. Author, Journal 87, 189 (2012)."
    pm = re.search(r"([A-Z][A-Za-z&.\s:]*?)\s*(\d+)\s*,\s*([A-Z]?\d+)\s*\((?:19|20)\d\d\)", clean)

    # authors: the segment before the year (AAS style) or before the journal (physics style)
    cut = ym.start() if ym else min(len(clean), 200)
    if pm and pm.start() < cut:
        cut = pm.start()
    seg = clean[:cut]
    seg = re.sub(r"\(\s*$", "", seg).strip().rstrip(",;:(").strip()
    seg = re.sub(r"\bet\s+al\.?", "", seg).strip().rstrip(",").strip()
    authors: list[str] = []
    if seg:
        # "Surname, I. J., Surname2, K., & Surname3, L."
        pairs = re.findall(r"([^,&;]+?),\s*((?:[A-Z][a-z]?\.\s*-?\s*)+)(?:,|$|\s*&|\s*and\b)", seg)
        if pairs:
            authors = [f"{a.strip().lstrip('&').strip()}, {i.strip()}" for a, i in pairs]
        else:
            for part in re.split(r",\s*(?:and\s+|&\s*)?|\s+and\s+|\s*&\s*", seg):
                part = part.strip()
                if not part:
                    continue
                if re.search(r"\d", part) or len(part.split()) > 5:
                    break                      # reached the title / journal
                authors.append(part)
            authors = authors[:50]
    if authors:
        out["authors"] = authors
        fa = _surname_from_name(authors[0])
        if fa:
            out["first_author"] = fa

    # journal / volume / page
    after = clean[ym.end():] if ym else ""
    jm = re.match(r"\s*[a-z]?\)?\s*[,.]?\s*([A-Za-z][A-Za-z&.\s:\-]*?)\s*,\s*(\d+)(?:\s*,\s*([A-Za-z]?\d+))?",
                  after)
    if pm and (not jm or pm.start() < (ym.start() if ym else 0)):
        out["journal"], out["volume"], out["page"] = pm.group(1).strip(), pm.group(2), pm.group(3)
    elif jm:
        out["journal"], out["volume"] = jm.group(1).strip(), jm.group(2)
        if jm.group(3):
            out["page"] = jm.group(3)
    else:
        jm2 = re.match(r"\s*[a-z]?\)?\s*[,.]?\s*([A-Za-z][A-Za-z&.\s\-]*?)\s*(?:[,.]|$)", after)
        if jm2 and journal_bibstem(jm2.group(1)):
            out["journal"] = jm2.group(1).strip()
    if "journal" not in out and out.get("arxiv_id"):
        out["journal"] = "arXiv"
    return out


def make_bib_entry(key: str, raw_text: str, hrefs: list[str] | None = None,
                   title: str | None = None) -> BibEntry:
    f = parse_bib_text(raw_text, hrefs)
    return BibEntry(
        key=key, raw_text=raw_text, authors=f.get("authors", []), first_author=f.get("first_author"),
        year=f.get("year"), title=title or f.get("title"), journal=f.get("journal"),
        arxiv_id=f.get("arxiv_id"), doi=f.get("doi"),
    )


def bib_volume_page(raw_text: str) -> tuple[str | None, str | None]:
    f = parse_bib_text(raw_text)
    return f.get("volume"), f.get("page")
