"""Section-aware, paragraph-packing chunker.

Offset contract
---------------
For every chunk with ``kind != "abstract"``::

    doc.full_text[chunk.char_start:chunk.char_end] == chunk.text

The abstract chunk (``kind == "abstract"``, always ordinal 0 when an abstract
is known) has text ``"<title>\\n\\n<abstract>"`` which is generally *not* a
substring of ``full_text``.  Its offsets span the abstract paragraphs if the
document contains them (or the located abstract text), and are
``char_start = char_end = -1`` when the abstract is not in ``full_text``
(``-1`` means "not in full_text").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from galrag.config import Settings, get_settings
from galrag.models import Chunk, Document, PaperMeta, Paragraph, Section

TINY_TOKENS = 15

# --------------------------------------------------------------------------
# token counting
# --------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _tokenizer(model_name: str):
    """Fast ``tokenizers`` tokenizer for the embedding model (importing the full
    ``transformers`` package is far slower, especially on NFS)."""
    try:
        get_settings()  # sets HF_HOME
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        return Tokenizer.from_file(hf_hub_download(model_name, "tokenizer.json"))
    except Exception:  # offline / not installed  # noqa: BLE001
        return None


def count_tokens(text: str, model_name: str | None = None) -> int:
    """Token count with the embedding model's tokenizer (no special tokens);
    falls back to ``words * 1.3`` if the tokenizer is unavailable."""
    if not text:
        return 0
    tok = _tokenizer(model_name or get_settings().embed_model)
    if tok is None:
        return int(len(text.split()) * 1.3) + 1
    try:
        return len(tok.encode(text, add_special_tokens=False).ids)
    except Exception:  # noqa: BLE001
        return int(len(text.split()) * 1.3) + 1


# --------------------------------------------------------------------------
# sentence splitting
# --------------------------------------------------------------------------

_ABBREV = {
    "al", "fig", "figs", "eq", "eqs", "e.g", "i.e", "cf", "vs", "sec", "secs", "tab",
    "ref", "refs", "approx", "dr", "no", "resp", "etc", "ca", "sim", "viz", "vol", "mr",
    "mrs", "prof", "st", "jr", "inc", "col", "sect", "app", "eg", "ie", "pp", "p", "ed",
}
_CAND = re.compile(r"[.!?][\"')\]]*(\s+)(?=[\"'(\[]?[A-Z0-9\\$])")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans of sentences in ``text`` (contiguous apart
    from the whitespace between them; spans never include leading/trailing
    whitespace)."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _CAND.finditer(text):
        end = m.start(1)
        head = text[start:end]
        # last word before the period
        w = re.search(r"([A-Za-z.]+)[.!?][\"')\]]*$", head)
        word = w.group(1).lower().strip(".") if w else ""
        if text[m.start()] == ".":
            if word in _ABBREV:
                continue
            if len(word) == 1 and word.isalpha() and (w and w.group(1).isupper()):
                continue  # initials "J. Smith"
            # "e.g." / "i.e." with internal dots
            if re.search(r"(?:e\.g|i\.e)$", head.lower().rstrip(".")):
                continue
        if end > start:
            spans.append((start, end))
        start = m.end(1)
    if start < len(text.rstrip()):
        spans.append((start, len(text.rstrip())))
    return [(a, b) for a, b in spans if text[a:b].strip()]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_SEC_NUM = re.compile(r"^S?(\d+|[A-Z]\d*)((?:\.(?:SS|SSS)?\d+)*)$")


def _section_label(sec: Section) -> str:
    title = (sec.title or "").strip()
    if re.match(r"^(\d+(\.\d+)*|[A-Z](\.\d+)*)[.\s)]", title) or not title:
        return title or sec.id
    m = re.match(r"^S(\d+)((?:\.SS+\d+)*)$", sec.id)
    if m:
        nums = [m.group(1)] + re.findall(r"(\d+)", m.group(2))
        return f"{'.'.join(nums)} {title}"
    return title


@dataclass
class _Unit:
    para: Paragraph
    start: int
    end: int
    tokens: int
    kind: str


def _meta_fields(doc: Document, meta: PaperMeta | None) -> dict:
    return {
        "year": meta.year if meta else None,
        "bibcode": meta.bibcode if meta else None,
        "arxiv_id": meta.arxiv_id if meta else None,
        "first_author": (meta.first_author if meta else "") or "",
        "refereed": bool(meta.refereed) if meta else False,
        "citation_count": int(meta.citation_count) if meta else 0,
        "source": doc.source,
    }


def _abstract_text(title: str, abstract: str) -> str:
    return f"{title.strip()}\n\n{abstract.strip()}".strip()


def chunk_abstract_only(meta: PaperMeta) -> list[Chunk]:
    """One abstract chunk for a paper with no full text (offsets are -1)."""
    text = _abstract_text(meta.title, meta.abstract)
    if not text:
        return []
    return [Chunk(
        chunk_id=f"{meta.paper_key}#0000", paper_key=meta.paper_key, ordinal=0,
        kind="abstract", section_id="abs", section_title="Abstract",
        section_path="Abstract", para_ids=[], text=text, char_start=-1, char_end=-1,
        n_tokens=count_tokens(text), year=meta.year, bibcode=meta.bibcode,
        arxiv_id=meta.arxiv_id, first_author=meta.first_author or "",
        refereed=bool(meta.refereed), citation_count=int(meta.citation_count or 0),
        source="abstract_only",
    )]


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------


def chunk_document(doc: Document, meta: PaperMeta | None = None,
                   settings: Settings | None = None) -> list[Chunk]:
    s = settings or get_settings()
    target, max_tok, overlap = s.chunk_target_tokens, s.chunk_max_tokens, s.chunk_overlap_paragraphs
    ft = doc.full_text
    mf = _meta_fields(doc, meta)
    out: list[Chunk] = []

    def emit(kind, sec, path, para_ids, start, end, pages, text=None):
        text = ft[start:end] if text is None else text
        out.append(Chunk(
            chunk_id=f"{doc.paper_key}#{len(out):04d}", paper_key=doc.paper_key,
            ordinal=len(out), kind=kind, section_id=sec.id, section_title=sec.title,
            section_path=path, para_ids=para_ids, text=text, char_start=start,
            char_end=end, page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None, n_tokens=count_tokens(text), **mf))

    # ---- abstract chunk ----
    title = doc.title or (meta.title if meta else "")
    abstract = doc.abstract or (meta.abstract if meta else "")
    abs_paras = [p for _, p in doc.iter_paragraphs()
                 if p.kind == "abstract" and 0 <= p.char_start < p.char_end <= len(ft)]
    has_abstract = bool(abstract.strip())
    if has_abstract:
        a0, a1 = -1, -1
        if abs_paras:
            a0, a1 = min(p.char_start for p in abs_paras), max(p.char_end for p in abs_paras)
        else:
            i = ft.find(abstract.strip()) if ft else -1
            if i >= 0:
                a0, a1 = i, i + len(abstract.strip())
        out.append(Chunk(
            chunk_id=f"{doc.paper_key}#0000", paper_key=doc.paper_key, ordinal=0,
            kind="abstract", section_id="abs", section_title="Abstract",
            section_path="Abstract", para_ids=[p.id for p in abs_paras],
            text=_abstract_text(title, abstract), char_start=a0, char_end=a1,
            n_tokens=count_tokens(_abstract_text(title, abstract)), **mf))

    # ---- body ----
    stack: list[tuple[int, str]] = []
    for sec in doc.sections:
        while stack and stack[-1][0] >= sec.level:
            stack.pop()
        stack.append((sec.level, _section_label(sec)))
        path = " > ".join(lbl for _, lbl in stack)

        units: list[_Unit] = []
        for p in sec.paragraphs:
            if has_abstract and p.kind in ("abstract", "title"):
                continue
            if not (0 <= p.char_start < p.char_end <= len(ft)) or not ft[p.char_start:p.char_end].strip():
                continue
            units.append(_Unit(p, p.char_start, p.char_end,
                               count_tokens(ft[p.char_start:p.char_end]), p.kind))

        # tiny captions/footnotes merge into the packing flow as body text
        def own(u: _Unit) -> bool:
            return u.kind in ("caption", "footnote") and u.tokens >= TINY_TOKENS

        cur: list[_Unit] = []
        cur_tok = 0

        def flush(cur_units, sec=sec, path=path):
            if not cur_units:
                return
            u0 = cur_units[0]
            if (len(cur_units) == 1 and u0.kind in ("caption", "footnote") and u0.tokens < TINY_TOKENS
                    and out and out[-1].section_id == sec.id and out[-1].kind != "abstract"
                    and out[-1].char_end <= u0.start):
                prev = out[-1]  # tiny caption/footnote: merge into the preceding chunk
                text = ft[prev.char_start:u0.end]
                n = count_tokens(text)
                if n <= max_tok:
                    prev.text, prev.char_end, prev.n_tokens = text, u0.end, n
                    prev.para_ids = prev.para_ids + [u0.para.id]
                    if u0.para.page:
                        prev.page_end = max(prev.page_end or 0, u0.para.page)
                    return
            kinds = {u.kind for u in cur_units}
            kind = "body" if "body" in kinds or len(kinds) > 1 else next(iter(kinds))
            emit(kind, sec, path, [u.para.id for u in cur_units], cur_units[0].start,
                 cur_units[-1].end, [u.para.page for u in cur_units if u.para.page])

        new_in_cur = 0
        for u in units:
            if own(u) or u.tokens > max_tok:
                if new_in_cur:
                    flush(cur)
                cur, cur_tok, new_in_cur = [], 0, 0
                if own(u) and u.tokens <= max_tok:
                    flush([u])
                else:
                    _emit_split(u, ft, target, max_tok, sec, path, emit, own(u))
                continue
            if cur and cur_tok + u.tokens > target and new_in_cur:
                flush(cur)
                keep = cur[-overlap:] if overlap > 0 else []
                keep_tok = sum(k.tokens for k in keep)
                if keep and keep_tok + u.tokens > target:
                    keep, keep_tok = [], 0
                cur, cur_tok, new_in_cur = list(keep), keep_tok, 0
            cur.append(u)
            cur_tok += u.tokens
            new_in_cur += 1
        if new_in_cur:
            flush(cur)
    return out


def _emit_split(u: _Unit, ft: str, target: int, max_tok: int, sec, path, emit, is_cap: bool):
    """Split an over-long paragraph at sentence boundaries into sub-chunks."""
    text = ft[u.start:u.end]
    kind = u.kind if u.kind in ("caption", "footnote") else "body"
    pieces: list[tuple[int, int, int]] = []  # (start, end, tokens) relative to text
    for a, b in split_sentences(text):
        t = count_tokens(text[a:b])
        if t > max_tok:  # pathological: hard-split on whitespace
            words = [(m.start() + a, m.end() + a) for m in re.finditer(r"\S+", text[a:b])]
            i = 0
            while i < len(words):
                j, tok = i, 0
                while j < len(words) and (tok < target or j == i):
                    tok += count_tokens(text[words[j][0]:words[j][1]]) or 1
                    j += 1
                pieces.append((words[i][0], words[j - 1][1], tok))
                i = j
        else:
            pieces.append((a, b, t))
    cur: list[tuple[int, int, int]] = []
    tot = 0
    for pc in pieces + [None]:
        if pc is None or (cur and tot + pc[2] > target):
            if cur:
                emit(kind, sec, path, [u.para.id], u.start + cur[0][0], u.start + cur[-1][1],
                     [u.para.page] if u.para.page else [])
            cur, tot = [], 0
        if pc is not None:
            cur.append(pc)
            tot += pc[2]
