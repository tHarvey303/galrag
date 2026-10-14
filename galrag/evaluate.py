"""Retrieval evaluation against ``tests/queries.yaml``.

Each query lists expected bibcodes.  For every query we check the top-k passage hits
(``VectorStore.search``) and the top-k paper hits (``VectorStore.search_papers``):

* ``recall``: fraction of expected papers found in the top-k passages;
* ``recall_retrievable``: same, but only over expected papers that are in the corpus
  (``db.resolve`` succeeds), so "not ingested" is separated from "missed";
* ``rr``: reciprocal rank of the first expected paper among the distinct papers of the
  passage hits (MRR is its mean);
* ``paper_recall``: recall of the paper-level (title+abstract) search.

Expected papers are classified as ``not_in_corpus``, ``abstract_only`` (in the corpus but
no full-text embedding) or full text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from galrag.models import Status


def load_queries(path: str | Path) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    if isinstance(data, dict):
        data = data.get("queries", [])
    out = []
    for q in data:
        if not q or not q.get("query"):
            continue
        out.append({"query": q["query"], "expected_bibcodes": list(q.get("expected_bibcodes") or []),
                    "notes": q.get("notes", "")})
    return out


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def run_eval(store, db, queries: list[dict], k: int = 10, mode: str = "hybrid",
             rerank: bool | None = None, papers: bool = True) -> dict[str, Any]:
    rows: list[dict] = []
    for q in queries:
        expected = q["expected_bibcodes"]
        exp_keys: dict[str, str | None] = {b: db.resolve(b) for b in expected}
        not_in_corpus = [b for b, kk in exp_keys.items() if kk is None]
        abstract_only = []
        for b, kk in exp_keys.items():
            if kk is not None:
                r = db.get_row(kk)
                if r is None or r["status"] != Status.EMBEDDED:
                    abstract_only.append(b)

        hits = store.search(q["query"], k=k, mode=mode, rerank=rerank)
        hit_papers: list[str] = []
        for h in hits:
            if h.chunk.paper_key not in hit_papers:
                hit_papers.append(h.chunk.paper_key)
        found = [b for b, kk in exp_keys.items() if kk is not None and kk in hit_papers]
        rr = 0.0
        for i, pk in enumerate(hit_papers, 1):
            if pk in {kk for kk in exp_keys.values() if kk}:
                rr = 1.0 / i
                break
        paper_found: list[str] = []
        if papers:
            pk_hits = [r["paper_key"] for r in store.search_papers(q["query"], k=k)]
            paper_found = [b for b, kk in exp_keys.items() if kk is not None and kk in pk_hits]
        n_exp = len(expected)
        n_retr = n_exp - len(not_in_corpus)
        rows.append({
            "query": q["query"], "expected": expected, "found": found, "paper_found": paper_found,
            "not_in_corpus": not_in_corpus, "abstract_only": abstract_only,
            "hit": bool(found), "recall": len(found) / n_exp if n_exp else None,
            "recall_retrievable": len(found) / n_retr if n_retr else None,
            "paper_recall": len(paper_found) / n_exp if (n_exp and papers) else None,
            "rr": rr, "top_papers": hit_papers[:5],
        })
    scored = [r for r in rows if r["expected"]]
    retr = [r for r in scored if r["recall_retrievable"] is not None]
    summary = {
        "n_queries": len(rows), "k": k, "mode": mode, "rerank": rerank,
        f"recall@{k}": _mean([r["recall"] for r in scored]),
        f"recall_retrievable@{k}": _mean([r["recall_retrievable"] for r in retr]),
        f"hit_rate@{k}": _mean([float(r["hit"]) for r in scored]),
        f"hit_rate_retrievable@{k}": _mean([float(r["hit"]) for r in retr]),
        "mrr": _mean([r["rr"] for r in scored]),
        f"paper_recall@{k}": _mean([r["paper_recall"] for r in scored]) if papers else None,
        "n_expected": sum(len(r["expected"]) for r in scored),
        "n_expected_not_in_corpus": sum(len(r["not_in_corpus"]) for r in scored),
        "n_expected_abstract_only": sum(len(r["abstract_only"]) for r in scored),
        "n_queries_unretrievable": len(scored) - len(retr),
    }
    return {"summary": summary, "rows": rows}
