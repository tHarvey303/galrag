"""Benchmark CPU embedding throughput for several (workers x threads) layouts.

Usage: python scripts/bench_embed.py [--n 2000] [--quick]
"""

from __future__ import annotations

import argparse
import random
import time

from galrag.config import get_settings

VOCAB = ["galaxy", "redshift", "JWST", "NIRCam", "spectroscopic", "photometric", "star", "formation", "rate", "stellar", "mass", "emission", "line", "Lyman", "alpha", "break", "reionization", "UV", "luminosity", "function", "dust", "metallicity", "[O", "III]", "H-beta", "ionizing", "photon", "escape", "fraction", "quasar", "halo", "dark", "matter", "simulation", "morphology", "size", "compact", "outflow", "burst", "bursty", "age", "SED", "fitting", "template", "detection", "candidate", "confirmed", "survey", "field", "depth", "magnitude", "selection", "completeness"]


def synth(n: int, words: int = 230, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.choice(VOCAB) for _ in range(words)) + f" z={rng.uniform(5, 15):.2f}."
            for _ in range(n)]


def single(texts, threads, bs=64):
    from galrag.embed import Embedder

    e = Embedder(threads=threads)
    e.embed_passages(texts[:16], bs)  # warm-up / model load
    t = time.time()
    e.embed_passages(texts, bs)
    return len(texts) / (time.time() - t)


def parallel(texts, nw, th, bs=64):
    from galrag.embed import embed_parallel

    # include model-load once per worker separately: warm with a tiny run is not possible
    # across pools, so time a 2-pass difference instead (small + full).
    t0 = time.time()
    embed_parallel(texts[: nw * 8], nw, th, bs, shard_size=8)
    t_small = time.time() - t0
    t0 = time.time()
    embed_parallel(texts, nw, th, bs)
    t_full = time.time() - t0
    return len(texts) / max(t_full - t_small, 1e-9), t_small  # subtract pool start + model load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--combos", default="6x24,12x12,16x12,24x8,48x4")
    ap.add_argument("--single", default="32,64,96")
    a = ap.parse_args()
    get_settings()
    texts = synth(a.n)
    from galrag.chunk import count_tokens

    print(f"{len(texts)} passages, mean tokens ~{sum(count_tokens(t) for t in texts[:50]) / 50:.0f}")
    rows = []
    for th in [int(x) for x in a.single.split(",") if x]:
        r = single(texts, th)
        rows.append(("single", 1, th, r))
        print(f"single  threads={th:3d}  {r:7.1f} chunks/s", flush=True)
    for c in [x for x in a.combos.split(",") if x]:
        nw, th = map(int, c.split("x"))
        r, tl = parallel(texts, nw, th)
        rows.append(("pool", nw, th, r))
        print(f"pool    {nw:3d} x {th:3d}  {r:7.1f} chunks/s (incl. model load ~{tl:.0f}s)", flush=True)
    print("\nmode   workers threads chunks/s")
    for m, nw, th, r in sorted(rows, key=lambda x: -x[3]):
        print(f"{m:6s} {nw:7d} {th:7d} {r:9.1f}")
    best = max((r for r in rows if r[0] == "pool"), key=lambda x: x[3], default=None)
    if best:
        print(f"\nRecommended: GALRAG_EMBED_WORKERS={best[1]} GALRAG_EMBED_THREADS_PER_WORKER={best[2]}"
              f"  ({best[3]:.0f} chunks/s -> 2M chunks in {2e6 / best[3] / 3600:.1f} h)")


if __name__ == "__main__":
    main()
