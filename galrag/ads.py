"""NASA ADS / SciX API client: search, harvest, BibTeX export, citation resolution."""

from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime, timedelta
from typing import Iterator

import httpx
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from galrag.config import ADS_FIELDS, get_settings
from galrag.db import Database, meta_from_ads, normalise_arxiv_id

log = logging.getLogger(__name__)

BASE_URL = "https://api.adsabs.harvard.edu/v1"
DEFAULT_DOCTYPE_FQ = "doctype:(article OR eprint OR inproceedings)"
RESOLVE_FIELDS = ["bibcode", "title", "first_author", "author", "year", "pub", "citation_count", "doi"]


class ADSError(RuntimeError):
    pass


class ADSRateLimitError(ADSError):
    pass


class _Retryable(ADSError):
    def __init__(self, msg: str, wait: float | None = None):
        super().__init__(msg)
        self.wait = wait


class ADSClient:
    def __init__(self, token: str | None = None, *, timeout: float | None = None):
        s = get_settings()
        token = token or s.ads_token
        if not token:
            raise ADSError("No ADS token: set GALRAG_ADS_TOKEN in .env")
        self.http = httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {token}", "User-Agent": s.user_agent},
            timeout=timeout or s.http_timeout_s,
        )
        self.rows_per_page = min(s.ads_rows_per_page, 2000)
        self.remaining: int | None = None
        self.limit: int | None = None
        self.reset: float | None = None   # epoch seconds

    def close(self) -> None:
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ------------------------------------------------------------ transport
    def _track(self, r: httpx.Response) -> None:
        h = r.headers
        try:
            if "x-ratelimit-remaining" in h:
                self.remaining = int(h["x-ratelimit-remaining"])
            if "x-ratelimit-limit" in h:
                self.limit = int(h["x-ratelimit-limit"])
            if "x-ratelimit-reset" in h:
                self.reset = float(h["x-ratelimit-reset"])
        except ValueError:
            pass

    def _check_budget(self) -> None:
        if self.remaining == 0 and self.reset:
            wait = self.reset - time.time()
            if wait > 120:
                raise ADSRateLimitError(
                    f"ADS daily rate limit exhausted; resets in {wait / 3600:.1f} h")
            if wait > 0:
                log.warning("ADS rate limit hit; sleeping %.0fs", wait + 1)
                time.sleep(wait + 1)

    @retry(retry=retry_if_exception_type((_Retryable, httpx.TransportError)),
           wait=wait_exponential(multiplier=2, max=60), stop=stop_after_attempt(6), reraise=True)
    def _request(self, method: str, path: str, **kw) -> httpx.Response:
        self._check_budget()
        r = self.http.request(method, path, **kw)
        self._track(r)
        if r.status_code == 429:
            ra = r.headers.get("retry-after")
            wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else None
            if self.remaining == 0 and self.reset and self.reset - time.time() > 120:
                raise ADSRateLimitError("ADS rate limit exhausted (429); resets in "
                                        f"{(self.reset - time.time()) / 3600:.1f} h")
            if wait:
                time.sleep(min(wait, 300))
            raise _Retryable("429 Too Many Requests", wait)
        if r.status_code >= 500:
            raise _Retryable(f"ADS server error {r.status_code}")
        if r.status_code >= 400:
            raise ADSError(f"ADS {r.status_code}: {r.text[:300]}")
        return r

    # --------------------------------------------------------------- search
    def _page(self, q: str, fl: list[str], rows: int, start: int, sort: str,
              fq: list[str] | None) -> dict:
        params: dict = {"q": q, "fl": ",".join(fl), "rows": rows, "start": start, "sort": sort}
        if fq:
            params["fq"] = fq
        return self._request("GET", "/search/query", params=params).json()["response"]

    def count(self, q: str, fq: list[str] | str | None = None) -> int:
        if isinstance(fq, str):
            fq = [fq]
        return int(self._page(q, ["bibcode"], 1, 0, "date desc", fq)["numFound"])

    def search(self, q: str, fl: list[str] | None = None, rows: int = 2000,
               sort: str = "citation_count desc", max_records: int | None = None,
               fq: list[str] | str | None = None) -> Iterator[dict]:
        """Yield docs, paging with start/rows; split by year if deep paging is refused."""
        fl = fl or ADS_FIELDS
        if isinstance(fq, str):
            fq = [fq]
        rows = max(1, min(rows, 2000, self.rows_per_page if rows >= 2000 else rows))
        n = 0
        try:
            for doc in self._paged(q, fl, rows, sort, fq, max_records):
                n += 1
                yield doc
        except ADSError as e:
            if isinstance(e, ADSRateLimitError) or "start" not in str(e).lower() or n:
                raise
            log.warning("Deep paging refused (%s); splitting by year", str(e)[:120])
            yield from self._by_year(q, fl, rows, sort, fq, max_records)

    def _paged(self, q, fl, rows, sort, fq, max_records) -> Iterator[dict]:
        start, got = 0, 0
        while True:
            want = rows if max_records is None else min(rows, max_records - got)
            if want <= 0:
                return
            resp = self._page(q, fl, want, start, sort, fq)
            docs = resp["docs"]
            total = resp["numFound"]
            if start == 0:
                log.info("ADS query matched %d records", total)
            for d in docs:
                yield d
            got += len(docs)
            start += len(docs)
            log.info("ADS fetched %d / %d (rate-limit remaining %s)", got, total, self.remaining)
            if not docs or start >= total:
                return

    def _by_year(self, q, fl, rows, sort, fq, max_records) -> Iterator[dict]:
        # Facet-free: just iterate years descending, bounded by this year + 1.
        got = 0
        for y in range(date.today().year + 1, 1800, -1):
            fq_y = (fq or []) + [f"year:{y}"]
            if self.count(q, fq_y) == 0:
                if y < 1990:
                    break
                continue
            for d in self._paged(q, fl, rows, sort, fq_y,
                                 None if max_records is None else max_records - got):
                got += 1
                yield d
            if max_records is not None and got >= max_records:
                return

    # --------------------------------------------------------------- bibtex
    def export_bibtex(self, bibcodes: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for i in range(0, len(bibcodes), 500):
            batch = bibcodes[i:i + 500]
            r = self._request("POST", "/export/bibtex", json={"bibcode": batch})
            out.update(split_bibtex(r.json().get("export", "")))
        return out

    # ------------------------------------------------------------- citations
    def citations_of(self, bibcode: str, rows: int = 200) -> list[dict]:
        return list(self.search(f"citations(bibcode:{bibcode})", fl=RESOLVE_FIELDS, rows=min(rows, 2000),
                                max_records=rows))

    def references_of_live(self, bibcode: str, rows: int = 2000) -> list[dict]:
        return list(self.search(f"references(bibcode:{bibcode})", fl=RESOLVE_FIELDS,
                                rows=min(rows, 2000), max_records=rows))

    def resolve_citation(self, text: str, max_results: int = 5) -> list[dict]:
        q, sort = build_citation_query(text)
        if q is None:
            return []
        docs = list(self.search(q, fl=RESOLVE_FIELDS, rows=max_results, sort=sort, max_records=max_results))
        return [_slim(d) for d in docs]

    # --------------------------------------------------------------- harvest
    def harvest(self, db: Database, query: str | None = None, *, max_records: int | None = None,
                since_entdate: str | None = None, extra_fq: list[str] | None = None,
                kind: str = "harvest") -> tuple[int, int]:
        s = get_settings()
        q = query or s.topic_query
        fq = [f"year:{s.year_min}-2100", DEFAULT_DOCTYPE_FQ]
        if since_entdate:
            fq.append(f'entdate:["{since_entdate[:10]}" TO *]')
        fq += extra_fq or []
        run = db.start_run(kind, f"{q} | fq={fq}")
        n_new = n_upd = 0
        batch: list = []
        try:
            for doc in self.search(q, fq=fq, max_records=max_records, sort="date desc"):
                try:
                    batch.append(meta_from_ads(doc))
                except Exception as e:  # malformed record: skip, keep going
                    log.warning("Skipping %s: %s", doc.get("bibcode"), e)
                if len(batch) >= 2000:
                    a, b = db.upsert_papers(batch)
                    n_new += a; n_upd += b; batch = []
            if batch:
                a, b = db.upsert_papers(batch)
                n_new += a; n_upd += b
        except BaseException as e:
            if batch:
                a, b = db.upsert_papers(batch)
                n_new += a; n_upd += b
            db.finish_run(run, n_new=n_new, n_updated=n_upd, note=f"aborted: {e!r}"[:300])
            raise
        db.finish_run(run, n_new=n_new, n_updated=n_upd)
        log.info("Harvest done: %d new, %d updated", n_new, n_upd)
        return n_new, n_upd

    def update(self, db: Database, **kw) -> tuple[int, int]:
        md = db.max_entdate()
        since = None
        if md:
            since = (datetime.fromisoformat(md[:10]) - timedelta(days=2)).date().isoformat()
        return self.harvest(db, since_entdate=since, kind="update", **kw)


# ------------------------------------------------------------------ helpers
_ENTRY = re.compile(r"^@\w+\s*\{\s*([^,\s]+)\s*,", re.M)


def split_bibtex(export: str) -> dict[str, str]:
    """Split an ADS bibtex export string into {bibcode: entry}."""
    out: dict[str, str] = {}
    ms = list(_ENTRY.finditer(export))
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(export)
        out[m.group(1)] = export[m.start():end].strip()
    return out


def get_bibtex_cached(db: Database, bibcodes: list[str], client: ADSClient | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    missing = []
    for b in dict.fromkeys(bibcodes):
        t = db.get_bibtex(b)
        if t:
            out[b] = t
        else:
            missing.append(b)
    if missing:
        own = client is None
        client = client or ADSClient()
        try:
            new = client.export_bibtex(missing)
        finally:
            if own:
                client.close()
        if new:
            db.put_bibtex(new)
            out.update(new)
    return out


_ARXIV_RE = re.compile(r"(?:arxiv:\s*)?\b(\d{4}\.\d{4,5})(?:v\d+)?\b|\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})\b", re.I)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.I)
_BIBCODE_RE = re.compile(r"\b(\d{4}[A-Za-z&.]{2,9}[.\d]{4,5}[A-Za-z0-9.]{4}[A-Z.])\b")
_STOP = {"et", "al", "al.", "and", "the", "of", "in", "a", "on", "for", "with", "to", "by"}
_JOURNALS = {"apj", "apjl", "apjs", "aj", "mnras", "a&a", "aap", "nature", "science", "natastron",
             "arxiv", "prd", "jcap", "pasj", "pasp", "araa", "a&ar", "ssrv", "phr", "newar"}


def _slim(d: dict) -> dict:
    return {
        "bibcode": d.get("bibcode"),
        "title": (d.get("title") or [""])[0],
        "first_author": d.get("first_author", ""),
        "year": int(d["year"]) if str(d.get("year", "")).isdigit() else None,
        "pub": d.get("pub"),
        "citation_count": int(d.get("citation_count") or 0),
    }


def _esc(s: str) -> str:
    return s.replace('"', "")


def build_citation_query(text: str) -> tuple[str | None, str]:
    """Turn a loose citation string into (ADS query, sort). Returns (None, ..) if unparseable."""
    t = text.strip()
    default_sort = "citation_count desc"
    if not t:
        return None, default_sort
    m = _DOI_RE.search(t)
    if m:
        return f'identifier:"{m.group(1).rstrip(".,;)")}"', "date desc"
    m = _BIBCODE_RE.search(t)
    if m and len(m.group(1)) == 19:
        return f'identifier:"{m.group(1)}"', "date desc"
    m = _ARXIV_RE.search(t)
    if m:
        aid = normalise_arxiv_id(m.group(1) or m.group(2))
        if aid:
            return f'identifier:"arXiv:{aid}"', "date desc"

    # author + year
    year = None
    ym = re.search(r"\+\s*'?(\d{2})\b", t)
    if ym:
        yy = int(ym.group(1))
        year = 2000 + yy if yy < 70 else 1900 + yy
        t = t[:ym.start()] + " " + t[ym.end():]
    else:
        ym = re.search(r"\b((?:19|20)\d{2})[a-z]?\b", t)
        if ym:
            year = int(ym.group(1))
            t = t[:ym.start()] + " " + t[ym.end():]
    t = re.sub(r"\bet\.?\s*al\.?", " ", t, flags=re.I)
    t = re.sub(r"[(),;]", " ", t)
    tokens = [w for w in t.split() if w]
    tokens = [w for w in tokens if w.lower().strip(".") not in _JOURNALS]
    # first author = first capitalised token (handle "Surname, I." leading comma already stripped)
    surname = None
    rest: list[str] = []
    for w in tokens:
        if surname is None and w[:1].isupper() and not re.fullmatch(r"[A-Z]\.?", w):
            surname = w.strip(".&")
            continue
        rest.append(w)
    if surname is None:
        return None, default_sort
    parts = [f'first_author:"{_esc(surname)}"']
    if year:
        parts.append(f"year:{year}")
    # title-ish words: lower-case words of length>3 after the author block, only if >=2 of them
    # (co-author surnames in "Bunker, Saxena, Cameron" are capitalised and skipped)
    words = [w for w in rest if w.lower() not in _STOP and len(w) > 3 and w[:1].islower()]
    if len(words) >= 2:
        parts.append(f'title:({" ".join(_esc(w) for w in words)})')
    return " ".join(parts), default_sort
