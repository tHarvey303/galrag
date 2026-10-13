"""Polite, resumable full-text acquisition.

Order of preference (see docs/PLAN.md §2):

1. ``https://arxiv.org/html/<id>``           native LaTeXML HTML  -> ``arxiv_html``
2. ``https://ar5iv.labs.arxiv.org/html/<id>`` LaTeXML HTML (older) -> ``ar5iv``
3. ``https://arxiv.org/pdf/<id>``             PDF                  -> ``arxiv_pdf``
4. ADS link gateway ``.../link_gateway/<bibcode>/PUB_PDF``          -> ``pub_pdf``
   (only for open-access records, ``PUB_PDF`` in ``esources``)

Raw bodies are cached on disk::

    raw/html/<id>.html.gz         arxiv.org/html
    raw/html/<id>.ar5iv.html.gz   ar5iv
    raw/pdf/<id>.pdf              arXiv PDF
    raw/pdf/<bibcode>.pub.pdf     publisher PDF
    raw/misses/<id>.<source>      negative cache (source known to be unavailable)

with ``/`` in old-style arXiv ids replaced by ``_`` (:func:`fs_safe`).  An
existing raw file is returned without touching the network, so fetching is
idempotent and resumable.  Network etiquette: one :class:`RateLimiter` per
process enforces a minimum spacing per host, sends the configured
User-Agent and backs off exponentially on 429/5xx (honouring ``Retry-After``).
"""

from __future__ import annotations

import gzip
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from galrag.config import Settings, get_settings
from galrag.models import FullTextSource, PaperMeta

log = logging.getLogger(__name__)

ALL_SOURCES: tuple[FullTextSource, ...] = ("arxiv_html", "ar5iv", "arxiv_pdf", "pub_pdf")
#: A source recorded as unavailable is retried after this many days (arXiv adds HTML over time).
MISS_TTL_DAYS = 30
_RETRY_STATUS = {429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    """No full-text source could be retrieved."""


@dataclass
class FetchResult:
    source: FullTextSource
    path: Path
    url: str
    fetched_at: datetime
    from_cache: bool = False


def fs_safe(key: str) -> str:
    """Filesystem-safe version of a paper key / arXiv id / bibcode.

    ``astro-ph/0601001`` -> ``astro-ph_0601001``; ``arXiv:2405.03781`` ->
    ``arXiv_2405.03781``.  Bibcodes keep their dots and ``&``.
    """
    s = key.replace("/", "_").replace(":", "_").replace("\\", "_")
    s = re.sub(r"[\x00-\x1f<>\"|?*]", "_", s)
    return s.strip() or "_"


# ---------------------------------------------------------------- rate limiting
class RateLimiter:
    """Per-host minimum spacing between requests plus retry/backoff.

    Process-local and thread-safe.  ``delays`` maps host -> seconds; hosts not
    listed use ``default_delay``.
    """

    def __init__(self, delays: dict[str, float] | None = None, default_delay: float = 5.0,
                 *, max_retries: int = 5, backoff_base: float = 5.0, max_backoff: float = 300.0,
                 sleep=time.sleep, clock=time.monotonic):
        self.delays = dict(delays or {})
        self.default_delay = default_delay
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff
        self._sleep = sleep
        self._clock = clock
        self._last: dict[str, float] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._glock = threading.Lock()

    @classmethod
    def from_settings(cls, s: Settings | None = None) -> RateLimiter:
        s = s or get_settings()
        return cls({"arxiv.org": s.arxiv_delay_s, "export.arxiv.org": s.arxiv_delay_s,
                    "ar5iv.labs.arxiv.org": s.ar5iv_delay_s, "ar5iv.org": s.ar5iv_delay_s},
                   default_delay=s.pub_delay_s)

    def delay_for(self, host: str) -> float:
        host = host.lower()
        if host in self.delays:
            return self.delays[host]
        if host.endswith(".arxiv.org") and "ar5iv" not in host:
            return self.delays.get("arxiv.org", self.default_delay)
        return self.default_delay

    def _lock(self, host: str) -> threading.Lock:
        with self._glock:
            return self._locks.setdefault(host, threading.Lock())

    def wait(self, host: str) -> None:
        """Block until a request to ``host`` is allowed, then mark it as made."""
        with self._lock(host):
            now = self._clock()
            last = self._last.get(host)
            if last is not None:
                remaining = self.delay_for(host) - (now - last)
                if remaining > 0:
                    self._sleep(remaining)
            self._last[host] = self._clock()

    @staticmethod
    def retry_after(resp: httpx.Response) -> float | None:
        v = resp.headers.get("retry-after")
        if not v:
            return None
        v = v.strip()
        if v.isdigit():
            return float(v)
        try:
            dt = parsedate_to_datetime(v)
            return max(0.0, (dt - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError):
            return None

    def request(self, client: httpx.Client, method: str, url: str, **kw) -> httpx.Response:
        """Rate-limited request with exponential backoff on 429/5xx and transport errors."""
        host = urlparse(url).hostname or ""
        attempt = 0
        while True:
            self.wait(host)
            try:
                resp = client.request(method, url, **kw)
            except httpx.TransportError as e:
                if attempt >= self.max_retries:
                    raise
                pause = min(self.max_backoff, self.backoff_base * 2 ** attempt)
                log.warning("%s %s failed (%s); retrying in %.0fs", method, url, e, pause)
                self._sleep(pause)
                attempt += 1
                continue
            if resp.status_code in _RETRY_STATUS and attempt < self.max_retries:
                ra = self.retry_after(resp)
                pause = min(self.max_backoff, ra if ra is not None else self.backoff_base * 2 ** attempt)
                log.warning("%s %s -> %s; backing off %.0fs", method, url, resp.status_code, pause)
                self._sleep(pause)
                attempt += 1
                continue
            return resp


_client: httpx.Client | None = None
_limiter: RateLimiter | None = None
_init_lock = threading.Lock()


def get_client() -> httpx.Client:
    global _client
    with _init_lock:
        if _client is None:
            s = get_settings()
            _client = httpx.Client(follow_redirects=True, timeout=s.http_timeout_s,
                                   headers={"User-Agent": s.user_agent})
        return _client


def get_limiter() -> RateLimiter:
    global _limiter
    with _init_lock:
        if _limiter is None:
            _limiter = RateLimiter.from_settings()
        return _limiter


def fetch_url_raw(url: str, *, method: str = "GET", client: httpx.Client | None = None,
                  limiter: RateLimiter | None = None, **kw) -> httpx.Response:
    """Rate-limited, retried request; returns the (fully read) response without raising on status."""
    return (limiter or get_limiter()).request(client or get_client(), method, url, **kw)


# ------------------------------------------------------------------ raw paths
def raw_path(source: FullTextSource, ident: str, s: Settings | None = None) -> Path:
    s = s or get_settings()
    k = fs_safe(ident)
    if source == "arxiv_html":
        return s.raw_dir / "html" / f"{k}.html.gz"
    if source == "ar5iv":
        return s.raw_dir / "html" / f"{k}.ar5iv.html.gz"
    if source == "arxiv_pdf":
        return s.raw_dir / "pdf" / f"{k}.pdf"
    if source == "pub_pdf":
        return s.raw_dir / "pdf" / f"{k}.pub.pdf"
    raise ValueError(source)


def source_url(source: FullTextSource, ident: str) -> str:
    if source == "arxiv_html":
        return f"https://arxiv.org/html/{ident}"
    if source == "ar5iv":
        return f"https://ar5iv.labs.arxiv.org/html/{ident}"
    if source == "arxiv_pdf":
        return f"https://arxiv.org/pdf/{ident}"
    if source == "pub_pdf":
        return f"https://ui.adsabs.harvard.edu/link_gateway/{ident}/PUB_PDF"
    raise ValueError(source)


def read_raw(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


def read_raw_text(path: Path) -> str:
    return read_raw(path).decode("utf-8", errors="replace")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _miss_path(source: FullTextSource, ident: str, s: Settings) -> Path:
    return s.raw_dir / "misses" / f"{fs_safe(ident)}.{source}"


def _recent_miss(source: FullTextSource, ident: str, s: Settings) -> bool:
    p = _miss_path(source, ident, s)
    return p.exists() and (time.time() - p.stat().st_mtime) < MISS_TTL_DAYS * 86400


def _record_miss(source: FullTextSource, ident: str, s: Settings, reason: str) -> None:
    p = _miss_path(source, ident, s)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(reason + "\n")


# ------------------------------------------------------------- availability
_LTX_PARA = re.compile(r'class="[^"]*\bltx_para\b')


def usable_latexml(html: str) -> bool:
    """Real LaTeXML article (not "No HTML for this paper" / conversion-failed page)."""
    if "ltx_document" not in html:
        return False
    if "ltx_page_main" not in html and "<article" not in html:
        return False
    return len(_LTX_PARA.findall(html)) >= 3


def _try_source(source: FullTextSource, ident: str, s: Settings) -> tuple[bytes, str] | str:
    """Fetch one source; return (body, final_url) or a reason string if unavailable."""
    url = source_url(source, ident)
    if source in ("arxiv_html", "ar5iv"):
        resp = fetch_url_raw(url)
        if resp.status_code != 200:
            return f"HTTP {resp.status_code}"
        final = str(resp.url)
        if "/html/" not in final:
            return f"redirected to {final}"
        html = resp.text
        if not usable_latexml(html):
            return "placeholder / conversion failed"
        return html.encode("utf-8"), final
    # PDFs: stream so oversize bodies are not downloaded
    limit = s.max_pdf_mb * 1024 * 1024
    client, limiter = get_client(), get_limiter()
    host = urlparse(url).hostname or ""
    attempt = 0
    while True:
        limiter.wait(host)
        with client.stream("GET", url) as resp:
            if resp.status_code in _RETRY_STATUS and attempt < limiter.max_retries:
                ra = limiter.retry_after(resp)
                pause = min(limiter.max_backoff, ra if ra is not None else limiter.backoff_base * 2 ** attempt)
                limiter._sleep(pause)
                attempt += 1
                continue
            if resp.status_code != 200:
                return f"HTTP {resp.status_code}"
            ct = resp.headers.get("content-type", "").lower()
            if "pdf" not in ct:
                return f"content-type {ct or '?'}"
            cl = resp.headers.get("content-length")
            if cl and cl.isdigit() and int(cl) > limit:
                return f"too large ({int(cl) / 1e6:.1f} MB)"
            buf = bytearray()
            for chunk in resp.iter_bytes():
                buf.extend(chunk)
                if len(buf) > limit:
                    return "too large"
            if bytes(buf[:5]) != b"%PDF-":
                return "not a PDF body"
            return bytes(buf), str(resp.url)


def fetch_fulltext(meta: PaperMeta, *, prefer: tuple[FullTextSource, ...] = ALL_SOURCES,
                   allow_pub_pdf: bool | None = None, force: bool = False,
                   settings: Settings | None = None) -> FetchResult:
    """Fetch (or reuse from disk) the best available full text for ``meta``.

    Raises :class:`FetchError` if nothing works.  ``force=True`` ignores the
    raw cache and the negative cache.
    """
    s = settings or get_settings()
    if allow_pub_pdf is None:
        allow_pub_pdf = s.fetch_pub_pdf
    plan: list[tuple[FullTextSource, str]] = []
    for src in prefer:
        if src in ("arxiv_html", "ar5iv", "arxiv_pdf") and meta.arxiv_id:
            plan.append((src, meta.arxiv_id))
        elif src == "pub_pdf" and allow_pub_pdf and meta.bibcode and "PUB_PDF" in (meta.esources or []):
            plan.append((src, meta.bibcode))
    if not plan:
        raise FetchError(f"{meta.paper_key}: no fetchable source (arxiv_id={meta.arxiv_id}, "
                         f"esources={meta.esources})")

    # 1) anything already on disk?
    if not force:
        for src, ident in plan:
            p = raw_path(src, ident, s)
            if p.exists() and p.stat().st_size > 0:
                return FetchResult(src, p, source_url(src, ident),
                                   datetime.fromtimestamp(p.stat().st_mtime, UTC), True)

    # 2) network, in order
    reasons = []
    for src, ident in plan:
        if not force and _recent_miss(src, ident, s):
            reasons.append(f"{src}: cached miss")
            continue
        try:
            got = _try_source(src, ident, s)
        except httpx.HTTPError as e:
            reasons.append(f"{src}: {type(e).__name__}: {e}")
            log.warning("%s %s: %s", meta.paper_key, src, e)
            continue
        if isinstance(got, str):
            reasons.append(f"{src}: {got}")
            # only definitive negatives go into the miss cache
            if not got.startswith("HTTP 5") and not got.startswith("HTTP 429"):
                _record_miss(src, ident, s, got)
            continue
        body, final_url = got
        p = raw_path(src, ident, s)
        _atomic_write(p, gzip.compress(body, 6) if p.suffix == ".gz" else body)
        return FetchResult(src, p, final_url, datetime.now(UTC), False)
    raise FetchError(f"{meta.paper_key}: " + "; ".join(reasons))
