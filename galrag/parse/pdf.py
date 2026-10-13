"""PDF parser (arXiv / publisher PDFs) built on raw PyMuPDF text dictionaries.

Why not ``pymupdf4llm``?  On the two-column REVTeX/AASTeX fixture it produced
correct reading order but dropped superscript information into ``<sup>`` tags,
dehyphenated compound words blindly (``inelastic-dissipative`` ->
``inelasticdissipative``) and scrambled the reference list.  The PDF content
stream of LaTeX output is already in reading order, so we work with PyMuPDF
blocks/lines/spans directly, which also gives fonts, sizes, superscript flags
and positions for the heuristics below.

Pipeline
--------
1. lines per page with font statistics; same-baseline fragments are merged;
   rotated text (arXiv side stamp), running headers/footers and page numbers
   are dropped; text inside figure regions (``cluster_drawings`` / images) is
   dropped unless it is a caption.
2. blocks -> paragraphs (indentation splits), classified as heading / caption /
   footnote / body / non-prose (equations, axis labels: dropped).
3. title (largest font on page 1), abstract ("Abstract" heading/prefix, or the
   longest prose block before the first heading), numbered or styled headings
   -> sections (``S1``, ``S1.SS2``, ``A1``).
4. paragraphs are continued across columns/pages when the previous one does
   not end a sentence and the next starts in lower case.
5. "References" (heading or a numbered list at the end) -> :class:`BibEntry`
   split on numbering or hanging indentation; never body paragraphs.
6. line-end hyphenation is resolved with a document vocabulary; numeric
   superscript citations become ``[n]`` with :class:`CiteMarker` links when the
   reference list is numbered.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from galrag.models import (
    BibEntry,
    CiteMarker,
    Document,
    FullTextSource,
    Paragraph,
    ParagraphKind,
    Section,
)
from galrag.parse.bib import make_bib_entry
from galrag.parse.common import finalize_document, map_span, normalize_with_map, unique_id

PARSER_VERSION = "pdf-1"

SUP_OPEN, SUP_CLOSE = "\x01", "\x02"      # superscript markers inside raw line text
_MATH_FONT = re.compile(r"CMMI|CMSY|CMEX|MSAM|MSBM|Symbol|Math|cmmi|cmsy|cmex|rsfs|stmary|esint", re.IGNORECASE)
_BOLD_FONT = re.compile(r"Bold|BX|Black|Heavy|Semibold|CMB|SSBX|\.B$|-B$", re.IGNORECASE)
_DIACRITIC = {"¨": "̈", "´": "́", "ˆ": "̂", "˜": "̃", "¸": "̧", "ˇ": "̌",
              "˚": "̊", "¯": "̄", "˘": "̆", "˙": "̇"}
_DIACRITIC_RE = re.compile("([" + "".join(_DIACRITIC) + "])([A-Za-zı])")
_CAPTION_RE = re.compile(r"^(?:FIG(?:URE)?\.?|Fig(?:ure)?\.?|TABLE|Table|Tab\.)\s*[A-Z]?\d+[a-z]?\s*[.:|\-—]",
                         re.UNICODE)
_ROMAN = r"(?:X{0,3})(?:IX|IV|V?I{0,3})"
_HEAD_NUM_RE = re.compile(
    rf"^(?P<num>(?:{_ROMAN}|\d{{1,2}}(?:\.\d{{1,2}}){{0,3}}|[A-Z](?:\.\d{{1,2}}){{0,2}}))(?:\.|\s)\s*"
    r"(?P<title>[A-ZΑ-Ω][^\n]{1,140})$")
_UNNUMBERED_HEADS = re.compile(
    r"^(abstract|acknowledge?ments?|references|bibliography|literature cited|appendix(?:\s+[A-Z0-9]+)?"
    r"(?:[:.\s].*)?|introduction|conclusions?|summary|discussion|data availability)\.?$", re.IGNORECASE)
_REF_HEAD = re.compile(r"^(?:\d+\.?\s*)?(references|bibliography|literature cited|references and notes)\.?$",
                       re.IGNORECASE)
_ARXIV_STAMP = re.compile(r"^arXiv:\s*\d{4}\.\d{4,5}v\d+|^arXiv:[a-z\-]+/\d{7}v\d+", re.IGNORECASE)
_NUM_REF_START = re.compile(r"^\s*(?:\x01(\d{1,3})\x02|\[(\d{1,3})\]|(\d{1,3})\.)\s*(?=\S)")
_CITE_SUP = re.compile(r"^\d{1,3}(?:\s*[,–\-]\s*\d{1,3})*$")
_BRACKET_CITE = re.compile(r"\[(\d{1,3}(?:\s*[,–\-]\s*\d{1,3})*)\]")


@dataclass
class _Line:
    text: str                 # raw, with SUP_OPEN/SUP_CLOSE markers around superscripts
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool
    math_frac: float
    page: int

    @property
    def plain(self) -> str:
        return self.text.replace(SUP_OPEN, "").replace(SUP_CLOSE, "")


@dataclass
class _Block:
    lines: list[_Line]
    page: int
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def plain(self) -> str:
        return " ".join(ln.plain for ln in self.lines).strip()


@dataclass
class _Item:
    kind: str                 # heading | body | caption | footnote | ref
    lines: list[_Line]
    page: int
    level: int = 1
    num: str | None = None
    title: str = ""
    x_left: float = 0.0
    extra: dict = field(default_factory=dict)


def _fix_diacritics(s: str) -> str:
    def repl(m: re.Match) -> str:
        letter = "i" if m.group(2) == "ı" else m.group(2)
        return unicodedata.normalize("NFC", letter + _DIACRITIC[m.group(1)])

    return _DIACRITIC_RE.sub(repl, s)


# ---------------------------------------------------------------- extraction
def _figure_rects(page) -> list:
    """Regions occupied by vector drawings or images (text inside them is figure labels)."""
    import pymupdf

    rects = []
    cluster = getattr(page, "cluster_drawings", None)       # PyMuPDF >= 1.24
    if cluster is not None:
        try:
            rects.extend(r for r in cluster() if r.width > 40 and r.height > 40)
        except (ValueError, RuntimeError):
            pass
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"])
        if r.width > 40 and r.height > 40:
            rects.append(r)
    return rects


def _extract(pdf) -> tuple[list[_Block], dict]:
    import pymupdf

    blocks: list[_Block] = []
    stats = {"pages": len(pdf), "dropped_rotated": 0, "dropped_figure_text": 0}
    for page in pdf:
        pno = page.number + 1
        d = page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)
        frects = _figure_rects(page)
        for b in d["blocks"]:
            if b.get("type") != 0:
                continue
            lines: list[_Line] = []
            for ln in b["lines"]:
                dx, dy = ln.get("dir", (1, 0))
                if abs(dy) > 0.1 or dx < 0.9:
                    stats["dropped_rotated"] += 1
                    continue
                spans = [s for s in ln["spans"] if s["text"]]
                if not spans:
                    continue
                main = max(spans, key=lambda z: (z["size"], len(z["text"].strip())))
                main_size, base_y = main["size"], main["origin"][1]
                parts, nchar, nmath, nbold = [], 0, 0, 0
                sizes: Counter = Counter()
                for s in spans:
                    t = s["text"]
                    n = len(t.strip())
                    nchar += n
                    if _MATH_FONT.search(s["font"]):
                        nmath += n
                    if (s["flags"] & 16) or _BOLD_FONT.search(s["font"]):
                        nbold += n
                    raised = s["origin"][1] < base_y - 0.15 * main_size
                    sup = s["size"] < main_size - 0.5 and (bool(s["flags"] & 1) or raised)
                    if sup and t.strip():
                        parts.append(SUP_OPEN + t.strip() + SUP_CLOSE)
                    else:
                        parts.append(t)
                        sizes[round(s["size"] * 2) / 2] += n
                text = _fix_diacritics("".join(parts))
                # "26" "," "44" as separate superscript spans -> one marker
                text = re.sub("\x02(\\s*[,–\\-]?\\s*)\x01", r"\1", text)
                if not text.replace(SUP_OPEN, "").replace(SUP_CLOSE, "").strip():
                    continue
                size = sizes.most_common(1)[0][0] if sizes and sizes.most_common(1)[0][1] else main_size
                x0, y0, x1, y1 = ln["bbox"]
                lines.append(_Line(text, x0, y0, x1, y1, size, nbold > 0.6 * max(nchar, 1),
                                   nmath / max(nchar, 1), pno))
            if not lines:
                continue
            lines = _merge_same_baseline(lines)
            x0, y0, x1, y1 = b["bbox"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            first = lines[0].plain.strip()
            if frects and not _CAPTION_RE.match(first) and any(
                    r.x0 - 2 <= cx <= r.x1 + 2 and r.y0 - 2 <= cy <= r.y1 + 2 for r in frects):
                stats["dropped_figure_text"] += 1
                continue
            blocks.append(_Block(lines, pno, x0, y0, x1, y1))
    return blocks, stats


def _merge_same_baseline(lines: list[_Line]) -> list[_Line]:
    out: list[_Line] = []
    for ln in lines:
        if out:
            p = out[-1]
            overlap = min(p.y1, ln.y1) - max(p.y0, ln.y0)
            h = min(p.y1 - p.y0, ln.y1 - ln.y0) or 1
            if overlap > 0.5 * h and ln.x0 >= p.x1 - 1.5:
                gap = " " if ln.x0 - p.x1 > 0.8 or not p.text.endswith(("-",)) else ""
                n_p, n_l = len(p.plain), len(ln.plain)
                p.text = p.text + gap + ln.text
                p.x1, p.y0, p.y1 = ln.x1, min(p.y0, ln.y0), max(p.y1, ln.y1)
                p.math_frac = (p.math_frac * n_p + ln.math_frac * n_l) / max(n_p + n_l, 1)
                p.bold = p.bold and ln.bold
                continue
        out.append(ln)
    return out


def _drop_running_heads(blocks: list[_Block], page_h: dict[int, float], n_pages: int) -> tuple[list[_Block], int]:
    """Remove page numbers and lines repeating in the top/bottom margin across pages."""
    def key(t: str) -> str:
        return re.sub(r"\d+", "#", t.strip().lower())

    margin_counts: Counter = Counter()
    for b in blocks:
        h = page_h.get(b.page, 792.0)
        for ln in b.lines:
            if ln.y1 < 0.08 * h or ln.y0 > 0.92 * h:
                margin_counts[key(ln.plain)] += 1
    thresh = max(2, n_pages // 3)
    dropped = 0
    out = []
    for b in blocks:
        h = page_h.get(b.page, 792.0)
        keep = []
        for ln in b.lines:
            t = ln.plain.strip()
            in_margin = ln.y1 < 0.08 * h or ln.y0 > 0.92 * h
            if _ARXIV_STAMP.match(t) or (in_margin and (re.fullmatch(r"[\d\s\-–—]+|page \d+", t, re.IGNORECASE)
                                                        or margin_counts[key(t)] >= thresh)):
                dropped += 1
                continue
            keep.append(ln)
        if keep:
            b.lines = keep
            out.append(b)
    return out, dropped


# ------------------------------------------------------------- classification
def _is_prose(text: str, math_frac: float) -> bool:
    words = re.findall(r"[A-Za-z]{3,}", text)
    toks = text.split()
    if math_frac > 0.6:
        return False
    return len(words) >= 4 and len(words) >= 0.35 * max(len(toks), 1)


def _heading(block: _Block, body_size: float) -> tuple[int, str | None, str] | None:
    """Return (level, number, title) if ``block`` is a section heading."""
    if len(block.lines) > 3:
        return None
    text = " ".join(ln.plain.strip() for ln in block.lines)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or len(text) > 160:
        return None
    bold = all(ln.bold for ln in block.lines)
    size = max(ln.size for ln in block.lines)
    styled = bold or size > body_size + 0.4 or (text.isupper() and len(text) > 3)
    small = size < body_size - 1.6
    if text.endswith((",", ";")) or small:
        return None
    m = _HEAD_NUM_RE.match(text)
    if m and styled:
        num, title = m.group("num"), m.group("title").strip()
        if title.endswith(".") and len(title.split()) > 12:
            return None
        if re.fullmatch(r"[A-Z]", num):
            level = 2
            if title.isupper():       # "A. APPENDIX TITLE" style
                level = 1
        elif re.fullmatch(_ROMAN, num) and num:
            level = 1
        else:
            level = num.count(".") + 1
        return level, num, title
    if _UNNUMBERED_HEADS.match(text) and (styled or text.isupper() or len(text) < 30):
        return 1, None, text
    if styled and bold and len(text.split()) <= 10 and not text.endswith(".") and text[:1].isupper() \
            and size >= body_size - 0.5:
        return 1, None, text
    return None


def _split_paragraph_lines(block: _Block, body_size: float) -> list[list[_Line]]:
    """Split a block on first-line indentation / large vertical gaps."""
    if len(block.lines) <= 1:
        return [block.lines]
    left = min(ln.x0 for ln in block.lines)
    gaps = [b.y0 - a.y1 for a, b in zip(block.lines, block.lines[1:]) if b.y0 > a.y0]
    typical = statistics.median(gaps) if gaps else 2.0
    groups: list[list[_Line]] = [[block.lines[0]]]
    for prev, ln in zip(block.lines, block.lines[1:]):
        indented = ln.x0 > left + 5 and ln.x0 < left + 40
        ends = prev.plain.rstrip().endswith((".", "!", "?", ":", ".)", '."', ".”"))
        big_gap = (ln.y0 - prev.y1) > max(typical * 2.5, 4.0) and ln.y0 > prev.y0
        if (indented and ends) or (big_gap and ends):
            groups.append([ln])
        else:
            groups[-1].append(ln)
    return groups


# ------------------------------------------------------------------- joining
class _Vocab:
    def __init__(self, blocks: list[_Block]):
        self.words: Counter = Counter()
        for b in blocks:
            for ln in b.lines:
                for w in re.findall(r"[A-Za-z][A-Za-z\-]+[A-Za-z]", ln.plain):
                    self.words[w.lower()] += 1

    def join(self, left: str, right: str) -> str:
        """Join ``left`` (ending with '-') and ``right`` (next line)."""
        m1 = re.search(r"([A-Za-z]+)-$", left)
        m2 = re.match(r"([A-Za-z]+)", right)
        if not m1 or not m2:
            return left + right
        a, b = m1.group(1), m2.group(1)
        if b[:1].isupper():
            return left + right                       # Lyman-Alpha, X-Ray: keep hyphen
        joined, hyph = (a + b).lower(), f"{a}-{b}".lower()
        nj, nh = self.words.get(joined, 0), self.words.get(hyph, 0)
        if nh > nj:
            return left + right
        if nj:
            return left[:-1] + right
        # compound of two real words ("temperature-dependent", "tight-binding")
        if len(a) >= 4 and len(b) >= 4 and self.words.get(a.lower()) and self.words.get(b.lower()):
            return left + right
        return left[:-1] + right


def _join_lines(lines: list[_Line], vocab: _Vocab) -> str:
    out = ""
    for ln in lines:
        t = ln.text.strip()
        if not out:
            out = t
        elif out.endswith("-") and not out.endswith(" -") and t[:1].isalpha():
            out = vocab.join(out, t)
        elif out.endswith("­"):
            out = out[:-1] + t
        else:
            out = out + " " + t
    return out


# ---------------------------------------------------------------- references
def _ref_number(line: _Line) -> int | None:
    m = _NUM_REF_START.match(line.text)
    if not m:
        return None
    return int(next(g for g in m.groups() if g))


def _looks_like_numbered_refs(blocks: list[_Block], start: int) -> bool:
    nums = []
    for b in blocks[start:start + 6]:
        for ln in b.lines:
            n = _ref_number(ln)
            if n is not None:
                nums.append(n)
    return len(nums) >= 3 and nums[0] == 1 and nums[1:3] == [2, 3]


def _split_references(lines: list[tuple[_Line, float]], vocab: _Vocab) -> tuple[list[str], bool]:
    """Split reference lines into entry texts.  ``lines`` = (line, block_left)."""
    numbered = sum(1 for ln, _ in lines if _ref_number(ln) is not None) >= max(3, 0.25 * len(lines)) \
        if lines else False
    entries: list[list[_Line]] = []
    expect = None
    for ln, left in lines:
        new = False
        if numbered:
            n = _ref_number(ln)
            if n is not None and (expect is None or n == expect or (n > expect and n - expect <= 2)):
                new, expect = True, n + 1
        else:
            new = ln.x0 <= left + 2.5
        if new or not entries:
            entries.append([ln])
        else:
            entries[-1].append(ln)
    texts = []
    for ent in entries:
        t = _join_lines(ent, vocab)
        if numbered:
            t = _NUM_REF_START.sub("", t, count=1)
        t = t.replace(SUP_OPEN, "").replace(SUP_CLOSE, "")
        t = normalize_with_map(t)[0]
        if len(t) >= 8:
            texts.append(t)
    return texts, numbered


# --------------------------------------------------------------------- main
def _expand_nums(s: str) -> list[int]:
    out: list[int] = []
    for part in re.split(r"\s*,\s*", s):
        m = re.fullmatch(r"(\d+)\s*[–\-]\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if 0 < b - a < 60:
                out.extend(range(a, b + 1))
                continue
        if part.strip().isdigit():
            out.append(int(part))
    return out


def _render_paragraph(raw: str, n_refs: int, numeric: bool) -> tuple[str, list[CiteMarker]]:
    """Turn raw text with superscript markers into text + cite markers."""
    buf: list[str] = []
    spans: list[tuple[int, int, list[str]]] = []
    pos = 0
    i = 0
    while i < len(raw):
        j = raw.find(SUP_OPEN, i)
        if j < 0:
            buf.append(raw[i:])
            pos += len(raw) - i
            break
        buf.append(raw[i:j])
        pos += j - i
        k = raw.find(SUP_CLOSE, j)
        if k < 0:
            k = len(raw)
        sup = raw[j + 1:k].strip()
        before = "".join(buf)[-3:]
        # a citation follows a word (>= 2 letters) or punctuation, not a digit or a lone symbol (e^2, 10^9)
        word_ok = bool(re.search(r"([A-Za-z\u00C0-\u024F]{2}|[.,;:)\]’'\"”])\s*$", before))
        nums = _expand_nums(sup) if numeric and word_ok and _CITE_SUP.match(sup) else []
        if nums and all(1 <= n <= n_refs for n in nums):
            t = f"[{sup}]"
            spans.append((pos, pos + len(t), [f"bib.bib{n}" for n in nums]))
        else:
            t = sup
        buf.append(t)
        pos += len(t)
        i = k + 1
    text0 = "".join(buf)
    if numeric:
        for m in _BRACKET_CITE.finditer(text0):
            if any(a <= m.start() < b for a, b, _ in spans):
                continue
            nums = _expand_nums(m.group(1))
            if nums and all(1 <= n <= n_refs for n in nums):
                spans.append((m.start(), m.end(), [f"bib.bib{n}" for n in nums]))
    text, mp = normalize_with_map(text0)
    cites = []
    for a, b, keys in sorted(spans):
        s, e = map_span(text, mp, a, b)
        if e > s:
            cites.append(CiteMarker(bib_keys=keys, char_start=s, char_end=e, text=text[s:e]))
    return text, cites


def parse_pdf(path: str | Path, paper_key: str, source: FullTextSource = "arxiv_pdf",
              url: str | None = None) -> Document:
    import pymupdf

    pdf = pymupdf.open(str(path))
    page_h = {p.number + 1: p.rect.height for p in pdf}
    page_w = {p.number + 1: p.rect.width for p in pdf}
    blocks, _stats = _extract(pdf)
    blocks, _n_dropped = _drop_running_heads(blocks, page_h, len(pdf))
    meta_title = (pdf.metadata or {}).get("title") or ""
    n_pages = len(pdf)
    pdf.close()
    flags: list[str] = []

    all_lines = [ln for b in blocks for ln in b.lines]
    if not all_lines:
        doc = Document(paper_key=paper_key, source=source, source_url=url, parser_version=PARSER_VERSION,
                       n_pages=n_pages, quality_flags=["no_text", "ocr_suspect"])
        return finalize_document(doc)
    size_hist: Counter = Counter()
    for ln in all_lines:
        size_hist[ln.size] += len(ln.plain)
    body_size = size_hist.most_common(1)[0][0]
    vocab = _Vocab(blocks)

    # two-column suspicion: most body lines narrower than ~half the page
    widths = [ln.x1 - ln.x0 for ln in all_lines if abs(ln.size - body_size) < 0.6 and len(ln.plain) > 30]
    if widths and statistics.median(widths) < 0.45 * statistics.median(page_w.values()):
        flags.append("two_column_suspect")

    # ---- title: largest text in the top 40% of page 1
    title = ""
    p1 = [b for b in blocks if b.page == 1]
    if p1:
        big = max(max(ln.size for ln in b.lines) for b in p1 if b.y0 < 0.4 * page_h[1]) if any(
            b.y0 < 0.4 * page_h[1] for b in p1) else 0
        if big > body_size + 1:
            tl = [ln for b in p1 for ln in b.lines if ln.size >= big - 0.5 and ln.y0 < 0.4 * page_h[1]]
            title = normalize_with_map(_join_lines(tl, vocab).replace(SUP_OPEN, "").replace(SUP_CLOSE, ""))[0]
    if not title and meta_title:
        title = meta_title.strip()
    title_lines = {id(ln) for b in p1 for ln in b.lines
                   if title and ln.size > body_size + 1 and ln.plain.strip() and ln.plain.strip() in title}

    # ---- classify blocks into items
    items: list[_Item] = []
    in_refs = False
    ref_lines: list[tuple[_Line, float]] = []
    seen_heading = False
    n_nonprose = 0
    for bi, b in enumerate(blocks):
        if all(id(ln) in title_lines for ln in b.lines):
            continue
        first = b.lines[0].plain.strip()
        h = _heading(b, body_size)
        if h and _REF_HEAD.match(h[2]) and not in_refs:
            in_refs = True
            continue
        if in_refs:
            if h and (h[2].lower().startswith("appendix") or (h[1] and re.fullmatch(r"[A-Z]", h[1] or ""))):
                in_refs = False
            else:
                left = min(ln.x0 for ln in b.lines)
                ref_lines.extend((ln, left) for ln in b.lines)
                continue
        if not in_refs and seen_heading and bi > len(blocks) // 2 and _ref_number(b.lines[0]) == 1 \
                and _looks_like_numbered_refs(blocks, bi):
            in_refs = True
            left = min(ln.x0 for ln in b.lines)
            ref_lines.extend((ln, left) for ln in b.lines)
            continue
        if h:
            seen_heading = True
            items.append(_Item("heading", b.lines, b.page, level=h[0], num=h[1], title=h[2]))
            continue
        if _CAPTION_RE.match(first):
            items.append(_Item("caption", b.lines, b.page))
            continue
        nch = sum(len(ln.plain) for ln in b.lines) or 1
        avg_size = sum(ln.size * len(ln.plain) for ln in b.lines) / nch
        math_frac = statistics.mean(ln.math_frac for ln in b.lines)
        text = b.plain
        if not _is_prose(text, math_frac):
            n_nonprose += 1
            continue
        if avg_size < body_size - 1.4 and b.y0 > 0.6 * page_h.get(b.page, 792):
            items.append(_Item("footnote", b.lines, b.page))
            continue
        for grp in _split_paragraph_lines(b, body_size):
            items.append(_Item("body", grp, b.page, x_left=min(ln.x0 for ln in grp)))

    # ---- bibliography
    ref_texts, numbered = _split_references(ref_lines, vocab)
    bibliography: list[BibEntry] = [make_bib_entry(f"bib.bib{i + 1}", t) for i, t in enumerate(ref_texts)]
    n_refs = len(bibliography)

    # ---- abstract: explicit heading/prefix, else the longest prose block before the first heading
    first_head = next((i for i, it in enumerate(items) if it.kind == "heading"), None)
    abstract_idx: list[int] = []
    for i, it in enumerate(items):
        if it.kind == "heading" and it.title.lower().startswith("abstract"):
            j = i + 1
            while j < len(items) and items[j].kind == "body":
                abstract_idx.append(j)
                j += 1
            break
        if it.kind == "body" and re.match(r"^\s*(abstract|ABSTRACT)\b[.:—\-\s]", it.lines[0].plain):
            abstract_idx.append(i)
            break
    if not abstract_idx and first_head is not None and first_head > 0:
        cands = [(len(_join_lines(items[i].lines, vocab)), i) for i in range(first_head) if items[i].kind == "body"]
        cands = [c for c in cands if c[0] >= 300]
        if cands:
            abstract_idx = [max(cands)[1]]

    # ---- assemble sections
    sections: list[Section] = []
    seen_ids: set[str] = set()
    counters: dict[str, int] = {}

    def new_section(sid: str, title_: str, level: int) -> Section:
        s = Section(id=unique_id(sid, seen_ids), title=title_, level=level)
        sections.append(s)
        return s

    abs_sec = new_section("abs", "Abstract", 1) if abstract_idx else None
    front = new_section("front", "", 1)
    cur = front
    sec_num = [0, 0, 0, 0]
    app_num = 0
    in_appendix = False
    open_para: tuple[Section, list[str], int, str] | None = None   # (section, raw parts, page, kind)
    pending: list[tuple[Section, str, int, str]] = []                # flushed paragraphs (sec, raw, page, kind)

    def flush() -> None:
        nonlocal open_para
        if open_para:
            s, parts, pg, kind = open_para
            pending.append((s, " ".join(parts), pg, kind))
            open_para = None

    for i, it in enumerate(items):
        if it.kind == "heading":
            flush()
            t = it.title
            if t.lower().startswith("abstract"):
                continue
            is_app = t.lower().startswith("appendix") or (in_appendix and it.level == 1)
            if t.lower().startswith("appendix"):
                in_appendix = True
            lvl = max(1, min(it.level, 4))
            if is_app and lvl == 1:
                app_num += 1
                sid = f"A{app_num}"
                sec_num = [0, 0, 0, 0]
            else:
                sec_num[lvl - 1] += 1
                for k in range(lvl, 4):
                    sec_num[k] = 0
                base = f"A{app_num}" if in_appendix and app_num else f"S{sec_num[0]}"
                sid = base + "".join(f".{'SS' if k == 1 else 'SSS' if k == 2 else 'P'}{sec_num[k]}"
                                     for k in range(1, lvl))
            full_title = f"{it.num} {t}" if it.num else t
            cur = new_section(sid, normalize_with_map(full_title)[0], lvl)
            continue
        if first_head is not None and i < first_head and i not in abstract_idx:
            continue          # front matter (authors, affiliations, dates)
        raw = _join_lines(it.lines, vocab)
        target = abs_sec if (i in abstract_idx and abs_sec is not None) else cur
        kind = "abstract" if target is abs_sec else it.kind
        if it.kind == "body" and kind == "body":
            if open_para and open_para[3] == "body":
                prev = open_para[1][-1].rstrip()
                cont = (not prev.endswith((".", "!", "?", ":")) or prev.endswith("-")) and \
                    re.match(r"^[\s\x01\x02]*[a-z(\[]", raw) is not None
                if cont:
                    if prev.endswith("-") and raw[:1].isalpha():
                        open_para[1][-1] = vocab.join(prev, raw)
                    else:
                        open_para[1].append(raw)
                    continue
            flush()
            open_para = (target, [raw], it.page, "body")
            continue
        if it.kind in ("caption", "footnote"):
            # do not break an open body paragraph: emit right away
            pending.append((target, raw, it.page, it.kind))
            continue
        flush()
        pending.append((target, raw, it.page, kind))
    flush()

    # stable order: paragraphs appear in the order they were completed per section
    kinds_map: dict[str, ParagraphKind] = {"body": "body", "abstract": "abstract", "caption": "caption",
                                           "footnote": "footnote"}
    for sec, raw, pg, kind in pending:
        text, cites = _render_paragraph(raw, n_refs, numbered)
        if not text:
            continue
        k = kinds_map.get(kind, "body")
        suffix = {"caption": "cap", "footnote": "fn"}.get(k, "P")
        ck = f"{sec.id}.{suffix}"
        counters[ck] = counters.get(ck, 0) + 1
        pid = unique_id(f"{ck}{counters[ck]}", seen_ids)
        sec.paragraphs.append(Paragraph(id=pid, kind=k, text=text, char_start=0, char_end=len(text),
                                        page=pg, cites=cites))

    abstract = "\n\n".join(p.text for p in abs_sec.paragraphs) if abs_sec else ""
    doc = Document(paper_key=paper_key, source=source, source_url=url, parser_version=PARSER_VERSION,
                   title=title, abstract=abstract, sections=sections, bibliography=bibliography,
                   n_pages=n_pages)
    finalize_document(doc)

    n_heads = sum(1 for it in items if it.kind == "heading")
    if n_heads == 0:
        flags.append("pdf_no_headings")
    if not abstract:
        flags.append("no_abstract")
    if not bibliography:
        flags.append("no_bibliography")
    if len(doc.full_text) < 2000:
        flags.append("short_text")
    if n_pages and len(doc.full_text) / n_pages < 500:
        flags.append("ocr_suspect")
    if any(c for _, p in doc.iter_paragraphs() for c in p.cites):
        flags.append("numeric_cites_linked")
    doc.quality_flags = flags
    return doc
