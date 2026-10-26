"""arXiv RSS listings (``https://rss.arxiv.org/rss/<category>``): parsing, the corpus
topic rule, and :class:`PaperMeta` records for papers ADS does not know yet.

The feed holds one day's announcements (new submissions, cross-lists and replacements)
and is rebuilt around 04:00 UTC on weekdays, so it has to be polled daily; anything
missed is picked up later by ``galrag update`` through ADS.  Entries carry the arXiv
id, title, authors, abstract and categories.  ADS usually indexes a listing a day or
two after it appears here, so a paper taken from the feed is keyed ``arXiv:<id>``
until the ADS record arrives, when :meth:`galrag.db.Database.upsert_papers` re-keys it
to the bibcode.

For a date range (backfill) :func:`fetch_listing` walks the arXiv API
(``https://export.arxiv.org/api/query``, Atom) over ``cat:<category> AND submittedDate:[..]``
and returns the same :class:`FeedEntry` records; new submissions in range only, so there
are no replacements.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import feedparser
import httpx

from galrag.config import TOPIC_KEYWORDS, get_settings
from galrag.db import compute_tier_and_priority, normalise_arxiv_id
from galrag.models import PaperMeta

log = logging.getLogger(__name__)

FEED_URL = "https://rss.arxiv.org/rss/{category}"
API_URL = "https://export.arxiv.org/api/query"
API_PAGE = 500                  # results per API call (arXiv asks for <= 2000 and a 3 s pause)
API_DELAY_S = 3.0
ANNOUNCE_TYPES = ("new", "cross", "replace", "replace-cross")

_SUMMARY = re.compile(r"^\s*arXiv:(?P<id>\S+?)\s+Announce Type:\s*(?P<type>[\w-]+)\s*"
                      r"(?:Abstract:)?\s*(?P<abstract>.*)$", re.DOTALL)
_VERSION = re.compile(r"v(\d+)$")
_PAREN = re.compile(r"\s*\([^)]*\)")
_PARTICLES = {"van", "der", "den", "de", "da", "di", "du", "do", "dos", "das", "del", "della",
              "von", "la", "le", "ten", "ter", "van't", "'t", "y", "e", "af", "zu", "al", "el", "bin"}
_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"}


@dataclass
class FeedEntry:
    arxiv_id: str                       # bare id, no version
    version: int | None
    title: str
    authors: list[str]                  # "Surname, Given" (ADS style)
    abstract: str
    categories: list[str]
    announce_type: str                  # new | cross | replace | replace-cross
    link: str = ""
    published: str | None = None        # YYYY-MM-DD
    raw: dict = field(default_factory=dict)

    @property
    def is_replacement(self) -> bool:
        return self.announce_type.startswith("replace")

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.abstract}".strip()


# ------------------------------------------------------------------ parsing
def parse_feed(text: str | bytes) -> tuple[dict, list[FeedEntry]]:
    """Parse an arXiv RSS document.  Returns (feed info, entries); malformed items are
    skipped with a warning."""
    f = feedparser.parse(text)
    info = {"title": f.feed.get("title"), "updated": f.feed.get("updated") or f.feed.get("published"),
            "link": f.feed.get("link"), "n_items": len(f.entries)}
    if f.get("bozo") and not f.entries:
        raise ValueError(f"not an RSS feed: {f.get('bozo_exception')}")
    out: list[FeedEntry] = []
    for e in f.entries:
        try:
            out.append(_entry(e))
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping feed item %s: %s", e.get("id") or e.get("link"), exc)
    return info, out


def _entry(e) -> FeedEntry:
    m = _SUMMARY.match(e.get("summary") or "")
    full_id = (m.group("id") if m else "") or e.get("id", "").rsplit(":", 1)[-1] or e.get("link", "")
    arxiv_id = normalise_arxiv_id(full_id)
    if not arxiv_id:
        raise ValueError(f"no arXiv id in {full_id!r}")
    vm = _VERSION.search(full_id)
    announce = (m.group("type") if m else "") or e.get("arxiv_announce_type") or "new"
    abstract = " ".join((m.group("abstract") if m else e.get("summary", "")).split())
    pub = None
    for key in ("published", "updated"):
        if e.get(key):
            try:
                pub = parsedate_to_datetime(e[key]).date().isoformat()
                break
            except (TypeError, ValueError):
                pass
    return FeedEntry(
        arxiv_id=arxiv_id, version=int(vm.group(1)) if vm else None,
        title=" ".join((e.get("title") or "").split()),
        authors=split_authors(e.get("author") or ""),
        abstract=abstract,
        categories=[t.get("term") for t in e.get("tags", []) if t.get("term")],
        announce_type=announce.lower(), link=e.get("link", ""), published=pub,
        raw={"id": e.get("id"), "rights": e.get("rights")},
    )


def split_authors(creator: str) -> list[str]:
    """``"Jane Doe, Arjen van der Wel (MPIA)"`` -> ``["Doe, Jane", "van der Wel, Arjen"]``."""
    out = []
    for name in _PAREN.sub("", creator).split(","):
        name = " ".join(name.split())
        if name:
            out.append(ads_name(name))
    return out


def ads_name(name: str) -> str:
    """Natural order -> ``Surname, Given``; keeps lower-case particles with the surname."""
    if "," in name:
        return name
    parts = name.split()
    if len(parts) < 2:
        return name
    suffix = ""
    if parts[-1].lower() in _SUFFIXES and len(parts) > 2:
        suffix, parts = parts[-1], parts[:-1]
    i = len(parts) - 1
    while i > 1 and parts[i - 1].lower() in _PARTICLES:
        i -= 1
    surname, given = " ".join(parts[i:]), " ".join(parts[:i])
    return f"{surname}, {given}" + (f", {suffix}" if suffix else "")


# ------------------------------------------------------------------ fetching
def fetch_feed(category: str | None = None, *, url: str | None = None,
               path: str | Path | None = None, timeout: float | None = None) -> tuple[dict, list[FeedEntry]]:
    """Download (or read from ``path``) and parse one feed.  ``category`` is e.g.
    ``astro-ph.GA`` (default: settings.feed_category)."""
    s = get_settings()
    if path is not None:
        info, entries = parse_feed(Path(path).read_bytes())
        info["source"] = str(path)
        return info, entries
    url = url or FEED_URL.format(category=category or s.feed_category)
    r = httpx.get(url, headers={"User-Agent": s.user_agent}, timeout=timeout or s.http_timeout_s,
                  follow_redirects=True)
    r.raise_for_status()
    info, entries = parse_feed(r.content)
    info["source"] = url
    return info, entries


def _api_entry(e, category: str) -> FeedEntry:
    """One Atom entry of the arXiv API -> :class:`FeedEntry` (announce type ``new`` when the
    primary category is the one queried, else ``cross``)."""
    full_id = (e.get("id") or "").rsplit("/abs/", 1)[-1]
    arxiv_id = normalise_arxiv_id(full_id)
    if not arxiv_id:
        raise ValueError(f"no arXiv id in {full_id!r}")
    vm = _VERSION.search(full_id)
    primary = (e.get("arxiv_primary_category") or {}).get("term") or ""
    cats = [t.get("term") for t in e.get("tags", []) if t.get("term")]
    if primary and primary not in cats:
        cats.insert(0, primary)
    pub = (e.get("published") or e.get("updated") or "")[:10] or None
    return FeedEntry(
        arxiv_id=arxiv_id, version=int(vm.group(1)) if vm else None,
        title=" ".join((e.get("title") or "").split()),
        authors=[ads_name(" ".join(a.get("name", "").split())) for a in e.get("authors", []) if a.get("name")],
        abstract=" ".join((e.get("summary") or "").split()),
        categories=cats,
        announce_type="new" if (not primary or primary == category) else "cross",
        link=e.get("link", ""), published=pub,
        raw={"id": e.get("id"), "doi": e.get("arxiv_doi"), "comment": e.get("arxiv_comment"),
             "updated": e.get("updated")},
    )


def fetch_listing(category: str | None = None, *, since: date | str, until: date | str | None = None,
                  page: int = API_PAGE, delay_s: float = API_DELAY_S, timeout: float | None = None,
                  max_entries: int | None = None) -> tuple[dict, list[FeedEntry]]:
    """All papers submitted to ``category`` (primary or cross-listed) between ``since`` and
    ``until`` (inclusive, default today) from the arXiv API, oldest first.  Returns
    (info, entries) like :func:`fetch_feed`."""
    s = get_settings()
    category = category or s.feed_category
    since = date.fromisoformat(since) if isinstance(since, str) else since
    until = date.fromisoformat(until) if isinstance(until, str) else (until or date.today())
    if until < since:
        raise ValueError(f"until {until} is before since {since}")
    q = f"cat:{category} AND submittedDate:[{since:%Y%m%d}0000 TO {until:%Y%m%d}2359]"
    source = f"{API_URL}?search_query={q}"
    info = {"title": f"arXiv API: {category} {since} to {until}", "updated": None, "link": source,
            "source": source, "n_items": 0, "since": since.isoformat(), "until": until.isoformat()}
    entries: list[FeedEntry] = []
    start, total = 0, None
    with httpx.Client(headers={"User-Agent": s.user_agent}, timeout=timeout or s.http_timeout_s,
                      follow_redirects=True) as client:
        while True:
            params = {"search_query": q, "start": start, "max_results": page,
                      "sortBy": "submittedDate", "sortOrder": "ascending"}
            for attempt in range(4):
                r = client.get(API_URL, params=params)
                f = feedparser.parse(r.content)
                # the API answers an empty feed (HTTP 200) while a query is still being prepared
                if r.status_code == 200 and (f.entries or f.feed.get("opensearch_totalresults") == "0"):
                    break
                log.warning("arXiv API start=%d: HTTP %d, %d entries; retrying", start, r.status_code,
                            len(f.entries))
                time.sleep(delay_s * (attempt + 2))
            else:
                r.raise_for_status()
                raise RuntimeError(f"arXiv API returned no entries for start={start} (total {total})")
            if total is None:
                total = int(f.feed.get("opensearch_totalresults") or 0)
                info["total_results"] = total
                log.info("arXiv API: %d papers in %s between %s and %s", total, category, since, until)
            for e in f.entries:
                try:
                    entries.append(_api_entry(e, category))
                except Exception as exc:  # noqa: BLE001
                    log.warning("skipping API entry %s: %s", e.get("id"), exc)
            start += len(f.entries)
            if start >= total or not f.entries or (max_entries is not None and len(entries) >= max_entries):
                break
            time.sleep(delay_s)
    if max_entries is not None:
        entries = entries[:max_entries]
    info["n_items"] = len(entries)
    info["updated"] = datetime.now().isoformat(timespec="seconds")
    return info, entries


# ------------------------------------------------------------------ topic rule
def _norm(text: str) -> str:
    t = text.lower().replace("-", " ").replace("–", " ")
    t = re.sub(r"isation\b", "ization", t)
    return " ".join(t.split())


def _compile_keywords(keywords: tuple[str, ...] | list[str]) -> list[tuple[str, re.Pattern]]:
    seen: dict[str, re.Pattern] = {}
    for k in keywords:
        n = _norm(k)
        if n not in seen:
            seen[n] = re.compile(r"\b" + re.escape(n) + r"\b")
    return list(seen.items())


_GALAXY = re.compile(r"\bgalax(?:y|ies)\b")
_KEYWORDS = _compile_keywords(TOPIC_KEYWORDS)


def topic_keywords(title: str, abstract: str, keywords: tuple[str, ...] | None = None) -> list[str]:
    """The corpus topic rule applied to a title+abstract, mirroring the ADS harvest query
    ``abs:(galaxy OR galaxies) AND abs:(<topic terms>)``.  Returns the matched topic
    terms (normalised), empty when the paper is off-topic.  Case-, hyphen- and
    British-spelling-insensitive."""
    text = _norm(f"{title} {abstract}")
    if not _GALAXY.search(text):
        return []
    pats = _compile_keywords(keywords) if keywords is not None else _KEYWORDS
    return [k for k, p in pats if p.search(text)]


# ------------------------------------------------------------------ metadata
def meta_from_feed(e: FeedEntry) -> PaperMeta:
    """A :class:`PaperMeta` keyed ``arXiv:<id>`` for a paper ADS has not indexed yet."""
    year = None
    if e.published:
        year = int(e.published[:4])
    elif re.match(r"\d{4}\.", e.arxiv_id):
        year = 2000 + int(e.arxiv_id[:2])
    m = PaperMeta(
        paper_key=f"arXiv:{e.arxiv_id}", arxiv_id=e.arxiv_id, title=e.title, authors=e.authors,
        first_author=e.authors[0] if e.authors else "", year=year, pubdate=e.published,
        entdate=e.published or datetime.now().date().isoformat(), abstract=e.abstract,
        arxiv_class=e.categories, bibstem="arXiv", pub="arXiv e-prints", doctype="eprint",
        refereed=False, raw={"source": "arxiv_rss", "announce_type": e.announce_type,
                             "version": e.version, "link": e.link, **e.raw},
    )
    m.tier, m.priority = compute_tier_and_priority(m)
    return m
