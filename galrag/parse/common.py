"""Text normalisation, sentence splitting and full-text assembly.

The single most important contract in the parsing layer lives here:
:func:`build_full_text` concatenates every paragraph of a document with
``"\\n\\n"`` and assigns ``char_start``/``char_end`` on each
:class:`~galrag.models.Paragraph` and :class:`~galrag.models.CiteMarker` so
that ``doc.full_text[p.char_start:p.char_end] == p.text`` (and likewise for
``cite.text``) exactly.

Parsers construct paragraphs with ``char_start=0, char_end=len(text)`` and
cite offsets *relative to the paragraph text*; :func:`build_full_text` then
shifts everything.  It interprets existing cite offsets as relative to the
paragraph's current ``char_start``, so calling it twice is harmless.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from galrag.models import Document, Section

#: Sentinel a parser can put into raw text to force a hard line break that
#: survives whitespace collapsing (e.g. between list items).
HARD_BREAK = "\x00"

PARA_SEP = "\n\n"

_ZERO_WIDTH = {"­", "\u200b", "‌", "‍", "⁠", "﻿"}


def _char_fold(ch: str) -> str:
    """Selective compatibility folding of a single character.

    Unlike blanket NFKC this keeps Greek letters, math operators, ``µ``/``Å``,
    super/subscripts and typographic quotes, but folds the things that only
    hurt search and quoting: ligatures (``ﬁ``), Mathematical Alphanumeric
    Symbols (``𝑧`` -> ``z``), full-width forms and invisible characters.
    """
    if ch in _ZERO_WIDTH:
        return ""
    o = ord(ch)
    if (0xFB00 <= o <= 0xFB06) or (0x1D400 <= o <= 0x1D7FF) or (0xFF01 <= o <= 0xFF5E) or o == 0x210E:
        return unicodedata.normalize("NFKC", ch)
    if ch == "−":          # minus sign stays (it is meaningful); nothing to do
        return ch
    return ch


def normalize_with_map(raw: str, *, hard_break: str = HARD_BREAK) -> tuple[str, list[int]]:
    """Normalise ``raw`` and return ``(text, offset_map)``.

    * every run of whitespace collapses to one space, or to ``"\\n"`` if the run
      contains a :data:`HARD_BREAK`; leading/trailing whitespace is dropped;
    * soft hyphens / zero-width characters are removed;
    * selected compatibility characters are folded (see :func:`_char_fold`).

    ``offset_map[i]`` is the output offset corresponding to raw offset ``i``
    (``len(offset_map) == len(raw) + 1``); use :func:`map_span` to translate
    spans because a mapped start may land on a collapsed space.
    """
    out: list[str] = []
    n = 0                      # current output length
    m = [0] * (len(raw) + 1)
    pending: str | None = None  # " " or "\n" waiting to be emitted
    for i, ch in enumerate(raw):
        m[i] = n
        if ch == hard_break:
            pending = "\n"
            continue
        if ch.isspace():
            if pending is None:
                pending = " "
            continue
        rep = _char_fold(ch)
        if not rep:
            continue
        if pending is not None:
            if n:
                out.append(pending)
                n += 1
                m[i] = n
            pending = None
        out.append(rep)
        n += len(rep)
    m[len(raw)] = n
    return "".join(out), m


def map_span(text: str, offset_map: list[int], start: int, end: int) -> tuple[int, int]:
    """Translate a raw span through ``offset_map`` and trim whitespace at its edges."""
    a = min(offset_map[start], len(text))
    b = min(offset_map[end], len(text))
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def normalize_text(raw: str) -> str:
    """Whitespace/unicode normalisation of one paragraph (no offset map)."""
    return normalize_with_map(raw)[0]


def normalize_block(raw: str) -> str:
    """Normalise multi-paragraph text: blank lines are kept as paragraph breaks."""
    paras = re.split(r"\n\s*\n", raw)
    return PARA_SEP.join(t for t in (normalize_text(p) for p in paras) if t)


# --------------------------------------------------------------------- PDF helpers
_HYPHEN_BREAK = re.compile(r"(\w)[-‐­]\n(\w)")


def dehyphenate(text: str) -> str:
    """Join words broken across lines: ``"gal-\\naxies"`` -> ``"galaxies"``.

    Keeps the hyphen when the second part starts with an upper-case letter or
    digit (``"Lyman-\\nAlpha"``, ``"z-\\n6"``), which is almost always a real
    compound.  Other line breaks are left for the caller to collapse.
    """

    def repl(mt: re.Match) -> str:
        a, b = mt.group(1), mt.group(2)
        if b.isupper() or b.isdigit() or a.isdigit():
            return f"{a}-{b}"
        return a + b

    return _HYPHEN_BREAK.sub(repl, text)


# ------------------------------------------------------------- sentence splitting
_ABBREV = {
    "al", "fig", "figs", "eq", "eqs", "e.g", "i.e", "cf", "vs", "sect", "sec", "secs", "tab",
    "ref", "refs", "approx", "resp", "no", "nos", "dr", "prof", "etc", "viz", "ca", "est",
    "app", "apps", "chap", "ch", "vol", "pp", "p", "ed", "eds", "mr", "mrs", "ms", "jr", "sr",
    "st", "inc", "co", "ltd", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec", "e.g.,", "i.e.,", "phys", "astrophys", "astron", "mon", "not",
    "sci", "res", "lett", "rev", "suppl", "j",
}
# sentence end: [.!?] + optional closers, whitespace, then something that can start a sentence
_SENT_END = re.compile(r"[.!?][\"'”’)\]]*(?=\s+[\"'“‘(\[$]?[A-Z0-9Α-Ω])")
_WORD_BEFORE = re.compile(r"([A-Za-z][A-Za-z.]*)$")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` offsets of sentences in ``text``.

    A simple regex splitter that does not break after common scholarly
    abbreviations (``et al.``, ``Fig.``, ``e.g.``, ``Eq.``, ``Sect.``...), after
    single-letter initials (``J. Smith``) or inside decimals (``z = 6.3``).
    Offsets are exact so that sentences can be quoted from ``full_text``.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    # skip leading whitespace
    while start < len(text) and text[start].isspace():
        start += 1
    for mt in _SENT_END.finditer(text):
        dot = mt.start()
        if text[dot] == ".":
            wm = _WORD_BEFORE.search(text[max(0, dot - 20):dot])
            if wm:
                word = wm.group(1).lower().rstrip(".")
                last = word.split(".")[-1] if "." in word and word not in _ABBREV else word
                if word in _ABBREV or last in _ABBREV:
                    continue
                if len(wm.group(1)) == 1 and wm.group(1).isupper():   # initial "J."
                    continue
        end = mt.end()
        if end > start:
            spans.append((start, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    if start < len(text):
        end = len(text)
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans


def split_sentences(text: str) -> list[str]:
    return [text[a:b] for a, b in sentence_spans(text)]


# --------------------------------------------------------------------- assembly
def build_full_text(sections: Iterable[Section]) -> str:
    """Concatenate all paragraphs and assign absolute offsets (in place).

    Paragraph texts are joined with ``"\\n\\n"`` in section order.  Cite offsets
    are interpreted relative to the paragraph's *current* ``char_start``
    (parsers create paragraphs with ``char_start=0``), validated against the
    cite's ``text`` and rewritten as absolute offsets into the returned string.
    """
    parts: list[str] = []
    pos = 0
    for sec in sections:
        for p in sec.paragraphs:
            if parts:
                parts.append(PARA_SEP)
                pos += len(PARA_SEP)
            old_start = p.char_start
            p.char_start = pos
            p.char_end = pos + len(p.text)
            for c in p.cites:
                ls, le = c.char_start - old_start, c.char_end - old_start
                if not (0 <= ls <= le <= len(p.text)) or p.text[ls:le] != c.text:
                    raise ValueError(
                        f"cite marker {c.text!r} does not match paragraph {p.id} at [{ls}:{le}]"
                    )
                c.char_start, c.char_end = pos + ls, pos + le
            parts.append(p.text)
            pos = p.char_end
    return "".join(parts)


def finalize_document(doc: Document) -> Document:
    """Drop empty paragraphs and untitled empty sections, build ``full_text``, verify offsets.

    Titled sections without paragraphs (a ``\\section`` that only contains
    subsections) are kept so that section paths can be reconstructed.
    """
    for s in doc.sections:
        s.paragraphs = [p for p in s.paragraphs if p.text]
    doc.sections = [s for s in doc.sections if s.paragraphs or s.title]
    doc.full_text = build_full_text(doc.sections)
    check_offsets(doc)
    return doc


def check_offsets(doc: Document) -> None:
    """Raise ``AssertionError`` if any paragraph or cite offset is wrong."""
    ft = doc.full_text
    for _, p in doc.iter_paragraphs():
        assert ft[p.char_start:p.char_end] == p.text, f"paragraph {p.id} offsets broken"
        for c in p.cites:
            assert p.char_start <= c.char_start <= c.char_end <= p.char_end, f"cite outside {p.id}"
            assert ft[c.char_start:c.char_end] == c.text, f"cite {c.text!r} in {p.id} broken"


def unique_id(base: str, seen: set[str]) -> str:
    """Return ``base`` or ``base~2``, ``base~3``... so paragraph ids stay unique."""
    if base not in seen:
        seen.add(base)
        return base
    k = 2
    while f"{base}~{k}" in seen:
        k += 1
    seen.add(f"{base}~{k}")
    return f"{base}~{k}"
