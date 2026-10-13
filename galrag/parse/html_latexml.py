"""Parser for LaTeXML HTML (arxiv.org/html and ar5iv.labs.arxiv.org).

Structure handled
-----------------
* ``article.ltx_document`` is the only part of the page looked at (site
  chrome, navigation, "Report issue" dialogs, footers are outside it).
* ``h1.ltx_title_document`` -> title; ``div.ltx_abstract`` -> section ``abs``.
* ``section.ltx_section / ltx_subsection / ltx_subsubsection / ltx_paragraph /
  ltx_appendix`` -> flat list of :class:`Section` with ``level`` and the
  LaTeXML id (``S1``, ``S1.SS2``, ``A1``); titles keep their numbering tag
  (``"III.1 Color-Color Diagram"``).
* ``div.ltx_para``: the ``p.ltx_p`` children (and display equations between
  them) are merged into one paragraph; lists become ``kind="list"`` paragraphs.
* inline ``<math>`` -> ``$alttext$``; display equations -> ``$...$`` inside
  the paragraph (or a ``kind="equation"`` paragraph when free-standing).
* ``<cite class="ltx_cite">`` -> :class:`CiteMarker` with the ``#bib.bibN``
  keys and the cite's visible text.
* footnotes (``span.ltx_note``) are removed from the running text and emitted
  after the paragraph as ``kind="footnote"`` paragraphs; figure/table captions
  are emitted as ``kind="caption"`` paragraphs with id ``<float id>.cap``.
* ``li.ltx_bibitem`` -> :class:`BibEntry` (key ``bib.bibN``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from bs4 import BeautifulSoup, NavigableString, Tag
from bs4.element import PreformattedString

from galrag.models import CiteMarker, Document, FullTextSource, Paragraph, ParagraphKind, Section
from galrag.parse.bib import make_bib_entry
from galrag.parse.common import (
    HARD_BREAK,
    finalize_document,
    map_span,
    normalize_with_map,
    unique_id,
)

PARSER_VERSION = "latexml-1"

_SKIP_TAGS = {"script", "style", "svg", "img", "object", "button", "input", "noscript", "nav",
              "header", "footer", "dialog", "iframe", "video", "audio", "canvas", "form", "embed",
              "template", "link", "meta"}
_BLOCK_TAGS = {"p", "div", "li", "tr", "td", "th", "table", "tbody", "thead", "ul", "ol", "dl", "dt",
               "dd", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "section", "figure",
               "figcaption", "pre"}
_SKIP_CLASSES = {"ltx_note_mark", "ltx_ERROR", "ltx_page_header", "ltx_page_footer", "ltx_page_navbar",
                 "ltx_page_logo", "ltx_TOC", "ltx_role_refnum", "ltx_dates", "ltx_pagination",
                 "ar5iv-footer", "ltx_rdf", "ltx_tag_note", "ltx_keywords",
                 "ltx_classification", "ltx_authors"}
_SECTION_LEVELS = {"ltx_chapter": 1, "ltx_section": 1, "ltx_appendix": 1, "ltx_subsection": 2,
                   "ltx_subsubsection": 3, "ltx_paragraph": 4, "ltx_subparagraph": 5}
_LIST_CLASSES = {"ltx_itemize", "ltx_enumerate", "ltx_description"}
_EQ_CLASSES = {"ltx_equation", "ltx_equationgroup", "ltx_eqn_table"}


def _classes(t: Tag) -> set[str]:
    c = t.get("class") or []
    return set(c if isinstance(c, list) else str(c).split())


def math_text(m: Tag) -> str:
    """LaTeX source of a ``<math>`` element, from ``alttext`` or the TeX annotation."""
    t = m.get("alttext")
    if not t:
        ann = m.find("annotation", attrs={"encoding": "application/x-tex"})
        t = ann.get_text() if ann else m.get_text(" ")
    t = " ".join(str(t).split())
    return f"${t}$" if t else ""


def equation_text(t: Tag) -> str:
    """Display equation table -> ``$...$`` per row (equation numbers dropped)."""
    rows = []
    trs = t.find_all("tr")
    if trs:
        for tr in trs:
            if tr.find_parent("tr") is not None and tr.find_parent("tr") in trs:
                continue
            maths = [math_text(m)[1:-1] for m in tr.find_all("math")]
            maths = [x for x in maths if x]
            if maths:
                rows.append("$" + " ".join(maths) + "$")
    else:
        rows = [x for x in (math_text(m) for m in t.find_all("math")) if x]
    # de-duplicate rows that appear through nested tables
    out: list[str] = []
    for r in rows:
        if r not in out:
            out.append(r)
    return " ".join(out)


@dataclass
class _Inline:
    """Accumulates raw text + cite spans while walking inline markup."""

    buf: list[str] = field(default_factory=list)
    n: int = 0
    cites: list[tuple[int, int, list[str]]] = field(default_factory=list)
    notes: list[Tag] = field(default_factory=list)
    floats: list[Tag] = field(default_factory=list)

    def add(self, s: str) -> None:
        if s:
            self.buf.append(s)
            self.n += len(s)

    @property
    def raw(self) -> str:
        return "".join(self.buf)

    def has_text(self) -> bool:
        return bool(self.raw.replace(HARD_BREAK, "").strip())


def _render(node, b: _Inline) -> None:
    if isinstance(node, PreformattedString):      # comments, CDATA, doctype, PIs
        return
    if isinstance(node, NavigableString):
        b.add(str(node))
        return
    if not isinstance(node, Tag):
        return
    name = node.name
    if name in _SKIP_TAGS:
        return
    cls = _classes(node)
    if name == "math":
        b.add(math_text(node))
        return
    if "ltx_note" in cls:                       # footnote / endnote: out of the running text
        b.notes.append(node)
        return
    if cls & _SKIP_CLASSES:
        return
    if name == "cite" and "ltx_cite" in cls:
        keys = []
        for a in node.find_all("a", href=True):
            h = a["href"]
            if "#bib." in h:
                k = h.split("#", 1)[1]
                if k not in keys:
                    keys.append(k)
        start = b.n
        for ch in node.children:
            _render(ch, b)
        if keys:
            b.cites.append((start, b.n, keys))
        return
    if name == "figure" or "ltx_float" in cls:
        b.floats.append(node)
        return
    if name == "table" and cls & _EQ_CLASSES:
        b.add(" " + equation_text(node) + " ")
        return
    if name == "br":
        b.add(" ")
        return
    if name in ("ul", "ol", "dl") or cls & _LIST_CLASSES:
        b.add(HARD_BREAK)
        _render_list_items(node, b)
        b.add(HARD_BREAK)
        return
    block = name in _BLOCK_TAGS
    if block:
        b.add(" ")
    for ch in node.children:
        _render(ch, b)
    if block:
        b.add(" ")


def _render_list_items(lst: Tag, b: _Inline) -> None:
    items = [li for li in lst.find_all(["li", "dt", "dd"]) if li.find_parent(["ul", "ol", "dl"]) is lst]
    if not items:
        for ch in lst.children:
            _render(ch, b)
        return
    for i, li in enumerate(items):
        if i:
            b.add(HARD_BREAK)
        for ch in li.children:
            _render(ch, b)


class _Builder:
    """Creates paragraphs/sections for one document."""

    def __init__(self) -> None:
        self.sections: list[Section] = []
        self.seen_ids: set[str] = set()
        self.counters: dict[str, int] = {}
        self.seen_floats: set[int] = set()

    def new_section(self, sid: str, title: str, level: int) -> Section:
        sec = Section(id=unique_id(sid, self.seen_ids), title=title, level=level)
        self.sections.append(sec)
        return sec

    def make_paragraph(self, b: _Inline, pid: str, kind: ParagraphKind) -> Paragraph | None:
        text, m = normalize_with_map(b.raw)
        if not text:
            return None
        cites = []
        for s, e, keys in b.cites:
            a, z = map_span(text, m, s, e)
            if z > a:
                cites.append(CiteMarker(bib_keys=keys, char_start=a, char_end=z, text=text[a:z]))
        return Paragraph(id=unique_id(pid, self.seen_ids), kind=kind, text=text, char_start=0,
                         char_end=len(text), cites=cites)

    def next_pid(self, sec: Section) -> str:
        k = self.counters.get(sec.id, 0) + 1
        self.counters[sec.id] = k
        return f"{sec.id}.P{k}"

    def emit(self, sec: Section, b: _Inline, kind: ParagraphKind = "body", pid: str | None = None) -> None:
        """Flush ``b`` into ``sec`` followed by its footnotes and float captions."""
        if b.has_text():
            p = self.make_paragraph(b, pid or self.next_pid(sec), kind)
            if p:
                sec.paragraphs.append(p)
        self.emit_notes(sec, b.notes)
        self.emit_floats(sec, b.floats)
        b.notes, b.floats = [], []

    def emit_notes(self, sec: Section, notes: list[Tag]) -> None:
        for note in notes:
            content = note.find(class_="ltx_note_content") or note
            nb = _Inline()
            for ch in content.children:
                _render(ch, nb)
            nid = note.get("id") or f"fn{len(self.seen_ids)}"
            self.emit(sec, nb, "footnote", f"{sec.id}.{nid}")

    def emit_floats(self, sec: Section, floats: list[Tag]) -> None:
        for fl in floats:
            if id(fl) in self.seen_floats:
                continue
            self.seen_floats.add(id(fl))
            for fc in fl.find_all("figcaption"):
                owner = fc.find_parent(["figure"]) or fl
                fid = owner.get("id") or fl.get("id") or f"{sec.id}.F"
                cb = _Inline()
                for ch in fc.children:
                    _render(ch, cb)
                self.emit(sec, cb, "caption", f"{fid}.cap")


def _heading_text(h: Tag) -> str:
    b = _Inline()
    for ch in h.children:
        _render(ch, b)
    return normalize_with_map(b.raw)[0]


def _walk(container: Tag, sec: Section, bld: _Builder, kind: ParagraphKind = "body") -> None:
    """Walk block-level children of a section-like container."""
    loose = _Inline()      # inline content sitting directly in the container

    def flush_loose() -> None:
        nonlocal loose
        if loose.has_text() or loose.notes or loose.floats:
            bld.emit(sec, loose, kind)
        loose = _Inline()

    for ch in container.children:
        if isinstance(ch, NavigableString):
            if not isinstance(ch, PreformattedString):
                loose.add(str(ch))
            continue
        if not isinstance(ch, Tag) or ch.name in _SKIP_TAGS:
            continue
        cls = _classes(ch)
        if cls & _SKIP_CLASSES:
            continue
        lvl = next((v for k, v in _SECTION_LEVELS.items() if k in cls), None)
        if ch.name in ("section", "div") and lvl is not None:
            flush_loose()
            h = ch.find(lambda t: t.name in ("h1", "h2", "h3", "h4", "h5", "h6") and "ltx_title" in _classes(t),
                        recursive=False)
            title = _heading_text(h) if h else ""
            sub = bld.new_section(ch.get("id") or f"{sec.id}.X", title, lvl)
            _walk(ch, sub, bld)
            continue
        if "ltx_bibliography" in cls or "ltx_abstract" in cls or "ltx_authors" in cls:
            continue   # handled separately
        if ch.name in ("h1", "h2", "h3", "h4", "h5", "h6") and "ltx_title" in cls:
            continue
        if "ltx_acknowledgements" in cls:
            flush_loose()
            h = ch.find(class_="ltx_title")
            sub = bld.new_section(ch.get("id") or "ack", _heading_text(h) if h else "Acknowledgements", 1)
            _walk(ch, sub, bld)
            continue
        if "ltx_para" in cls:
            flush_loose()
            _para(ch, sec, bld, kind)
            continue
        if ch.name == "p":
            flush_loose()
            b = _Inline()
            _render(ch, b)
            bld.emit(sec, b, kind)
            continue
        if ch.name == "figure" or "ltx_float" in cls:
            flush_loose()
            bld.emit_floats(sec, [ch])
            continue
        if ch.name == "table" and cls & _EQ_CLASSES:
            flush_loose()
            b = _Inline()
            b.add(equation_text(ch))
            bld.emit(sec, b, "equation")
            continue
        if ch.name in ("ul", "ol", "dl") or cls & _LIST_CLASSES:
            flush_loose()
            b = _Inline()
            _render_list_items(ch, b)
            bld.emit(sec, b, "list")
            continue
        if ch.name in ("div", "section", "blockquote", "article", "span") and ch.find(
                class_=lambda c: c in ("ltx_para", "ltx_p") or c in _SECTION_LEVELS):
            flush_loose()
            _walk(ch, sec, bld, kind)    # theorem, proof, quote, logical-block ...
            continue
        _render(ch, loose)
    flush_loose()


def _para(div: Tag, sec: Section, bld: _Builder, kind: ParagraphKind) -> None:
    """One ``div.ltx_para``: merge its p's (and display equations) into one paragraph."""
    b = _Inline()
    for ch in div.children:
        if isinstance(ch, NavigableString):
            if not isinstance(ch, PreformattedString):
                b.add(str(ch))
            continue
        if not isinstance(ch, Tag) or ch.name in _SKIP_TAGS:
            continue
        cls = _classes(ch)
        if ch.name in ("ul", "ol", "dl") or cls & _LIST_CLASSES:
            bld.emit(sec, b, kind)
            b = _Inline()
            lb = _Inline()
            _render_list_items(ch, lb)
            bld.emit(sec, lb, "list")
            continue
        if "ltx_para" in cls:
            bld.emit(sec, b, kind)
            b = _Inline()
            _para(ch, sec, bld, kind)
            continue
        if ch.name == "figure" or "ltx_float" in cls:
            b.floats.append(ch)
            continue
        b.add(" ")
        _render(ch, b)
        b.add(" ")
    bld.emit(sec, b, kind)


def _parse_bibliography(article: Tag) -> list:
    entries = []
    seen = set()
    for li in article.select("li.ltx_bibitem"):
        key = li.get("id") or f"bib.bib{len(entries) + 1}"
        if key in seen:
            continue
        seen.add(key)
        blocks = li.find_all(class_="ltx_bibblock")
        b = _Inline()
        for bl in blocks or [li]:
            b.add(" ")
            _render(bl, b)
        raw = normalize_with_map(b.raw)[0]
        if not raw:
            continue
        hrefs = [a["href"] for a in li.find_all("a", href=True)]
        title = None
        if len(blocks) >= 3:
            title = normalize_with_map(blocks[1].get_text(" "))[0].strip(" .,") or None
        entries.append(make_bib_entry(key, raw, hrefs, title))
    return entries


def is_latexml_document(html: str) -> bool:
    """True if the page contains a real LaTeXML article (not a placeholder/error page)."""
    return ("ltx_document" in html) and ("ltx_page_main" in html or "<article" in html)


def parse_latexml_html(html: str, paper_key: str, source: FullTextSource = "arxiv_html",
                       url: str | None = None, fetched_at: datetime | None = None) -> Document:
    soup = BeautifulSoup(html, "lxml")
    article = soup.find("article", class_="ltx_document") or soup.find(class_="ltx_document") \
        or soup.find(class_="ltx_page_content") or soup.body or soup
    flags: list[str] = []

    title_el = article.find(class_="ltx_title_document")
    title = _heading_text(title_el) if title_el else ""
    if not title and soup.title:
        title = normalize_with_map(soup.title.get_text())[0]

    bld = _Builder()
    abs_el = article.find(class_="ltx_abstract")
    abstract_text = ""
    if abs_el:
        abs_sec = bld.new_section("abs", "Abstract", 1)
        _walk(abs_el, abs_sec, bld, "abstract")
        abstract_text = "\n\n".join(p.text for p in abs_sec.paragraphs)

    front = bld.new_section("front", "", 1)    # content before the first \section
    _walk(article, front, bld)

    bibliography = _parse_bibliography(article)

    doc = Document(paper_key=paper_key, source=source, source_url=url, fetched_at=fetched_at,
                   parser_version=PARSER_VERSION, title=title, abstract=abstract_text,
                   sections=bld.sections, bibliography=bibliography)
    finalize_document(doc)

    body_secs = [s for s in doc.sections if s.id not in ("abs", "front")]
    if not body_secs:
        flags.append("no_sections")
    if not abstract_text:
        flags.append("no_abstract")
    if len(doc.full_text) < 2000:
        flags.append("short_text")
    if not bibliography:
        flags.append("no_bibliography")
    if "ltx_ERROR" in html:
        flags.append("latexml_errors")
    doc.quality_flags = flags
    return doc
