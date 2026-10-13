"""Locate quotes and citation contexts inside parsed documents.

These are the primitives behind the MCP tools ``get_quote``,
``citation_contexts`` and ``verify_claim``: given a Document, find where a
(possibly paraphrased) string occurs and return the *exact* span from
``full_text`` with section / paragraph / page provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from galrag.models import Document, Paragraph, QuoteMatch, Section
from galrag.parse.common import sentence_spans

_WS = re.compile(r"\s+")
def _surname(name: str) -> str:
    """'Bouwens, R. J.' / 'Bouwens RJ' / 'Le Fevre O' -> 'Bouwens' / 'Le Fevre'."""
    base = name.split(",")[0].strip()
    return re.sub(r"(\s+[A-Z]{1,3}\.?)+$", "", base).strip() or base


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def locate_paragraph(doc: Document, char_pos: int) -> tuple[Section, Paragraph] | None:
    for s, p in doc.iter_paragraphs():
        if p.char_start <= char_pos < p.char_end:
            return s, p
    return None


def split_sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets of sentences inside ``text`` (exact offsets)."""
    return sentence_spans(text)


def find_quote(doc: Document, query: str, *, min_similarity: float = 60.0) -> QuoteMatch:
    """Find ``query`` in ``doc``.

    1. exact substring match (whitespace-insensitive);
    2. otherwise the best-matching sentence window by rapidfuzz partial ratio.
    The returned ``quote`` is always an exact slice of ``doc.full_text``.  When the
    best candidate scores below ``min_similarity`` it is still returned with
    ``found=False`` so the caller can see the nearest passage.
    """
    ft = doc.full_text
    q = query.strip()
    miss = QuoteMatch(paper_key=doc.paper_key, found=False, exact=False, similarity=0.0, quote="",
                      char_start=-1, char_end=-1, source=doc.source)
    if not q or not ft:
        return miss

    # 1. exact, whitespace-insensitive: build a regex with \s+ between tokens
    pattern = r"\s+".join(re.escape(tok) for tok in q.split())
    m = re.search(pattern, ft, flags=re.IGNORECASE)
    if m:
        return _match(doc, m.start(), m.end(), exact=True, similarity=100.0)

    # 2. fuzzy: score each paragraph quickly, then sentence windows in the best few
    scored = []
    for s, p in doc.iter_paragraphs():
        if p.kind in ("title",):
            continue
        sc = max(fuzz.partial_ratio(_norm(q), _norm(p.text)),
                 fuzz.partial_token_set_ratio(_norm(q), _norm(p.text)) - 10)
        if sc >= min_similarity - 15:
            scored.append((sc, s, p))
    scored.sort(key=lambda t: -t[0])
    best: tuple[float, int, int] | None = None
    n_q_sent = max(1, len(split_sentences(q)))
    for sc, s, p in scored[:8]:
        sents = split_sentences(p.text)
        # windows of 1..n_q_sent+1 sentences
        for w in range(1, min(len(sents), n_q_sent + 1) + 1):
            for i in range(len(sents) - w + 1):
                a, b = sents[i][0], sents[i + w - 1][1]
                cand = p.text[a:b]
                nq, nc = _norm(q), _norm(cand)
                r = max(fuzz.ratio(nq, nc), 0.5 * (fuzz.token_set_ratio(nq, nc) + fuzz.ratio(nq, nc)))
                if best is None or r > best[0]:
                    best = (r, p.char_start + a, p.char_start + b)
    if best:
        m = _match(doc, best[1], best[2], exact=False, similarity=float(best[0]))
        m.found = best[0] >= min_similarity
        return m
    if scored:
        sc, s, p = scored[0]
        m = _match(doc, p.char_start, p.char_end, exact=False, similarity=float(sc))
        m.found = sc >= min_similarity
        return m
    return miss


def _match(doc: Document, a: int, b: int, *, exact: bool, similarity: float) -> QuoteMatch:
    loc = locate_paragraph(doc, a)
    s, p = loc if loc else (None, None)
    return QuoteMatch(
        paper_key=doc.paper_key, found=True, exact=exact, similarity=similarity,
        quote=doc.full_text[a:b], char_start=a, char_end=b,
        section_id=s.id if s else None, section_title=s.title if s else None,
        para_id=p.id if p else None, page=p.page if p else None, source=doc.source,
    )


@dataclass
class CitationContext:
    para_id: str
    section_id: str
    section_title: str
    page: int | None
    sentence: str          # the sentence containing the citation marker (exact slice)
    char_start: int
    char_end: int
    marker: str


def citation_contexts(doc: Document, bib_keys: set[str] | None = None, *, bibcode: str | None = None,
                      first_author: str | None = None, year: int | None = None) -> list[CitationContext]:
    """Sentences in ``doc`` that cite the bibliography entries selected by ``bib_keys``,
    or by ``bibcode`` / (``first_author``, ``year``) matched against the bibliography.

    Falls back to a textual "Surname et al. YEAR" / "Surname YEAR" search when the
    document has no resolved citation markers (e.g. PDF-derived text).
    """
    keys: set[str] = set(bib_keys or ())
    if bibcode:
        keys |= {b.key for b in doc.bibliography if b.bibcode == bibcode}
    if first_author and year and not keys:
        fa = first_author.split(",")[0].strip().lower()
        keys |= {b.key for b in doc.bibliography
                 if b.year == year and _surname(b.first_author or "").lower() == _surname(first_author).lower()}
    # Textual fallback (PDF-derived text, or HTML without \cite links): look for
    # "Surname et al. YEAR" / "Surname YEAR" when no marker in the document points at ``keys``.
    if not first_author or not year:
        for b in doc.bibliography:
            if b.key in keys and b.first_author and b.year:
                first_author, year = b.first_author, b.year
                break
    has_markers = any(keys & set(c.bib_keys) for _, p in doc.iter_paragraphs() for c in p.cites)
    pat = None
    if not has_markers and first_author and year:
        fa = re.escape(_surname(first_author))
        pat = re.compile(
            rf"\b{fa}\b(?:\s*,?\s*(?:et\s+al\.?|(?:&|and)\s+[A-Z][\w'-]+))?\s*,?\s*\(?{year}[a-z]?\)?",
            re.IGNORECASE)
    out: list[CitationContext] = []
    seen: set[tuple[int, int]] = set()
    for s, p in doc.iter_paragraphs():
        hits = [c for c in p.cites if keys & set(c.bib_keys)]
        if not hits and pat is not None:
            hits = [type("M", (), {"char_start": p.char_start + m.start(), "char_end": p.char_start + m.end(),
                                   "text": m.group(0)})() for m in pat.finditer(p.text)]
        if not hits:
            continue
        sents = split_sentences(p.text)
        for c in hits:
            rel = c.char_start - p.char_start
            for a, b in sents:
                if a <= rel < b:
                    key = (p.char_start + a, p.char_start + b)
                    if key in seen:
                        break
                    seen.add(key)
                    out.append(CitationContext(
                        para_id=p.id, section_id=s.id, section_title=s.title, page=p.page,
                        sentence=doc.full_text[key[0]:key[1]], char_start=key[0], char_end=key[1],
                        marker=c.text,
                    ))
                    break
    return out
