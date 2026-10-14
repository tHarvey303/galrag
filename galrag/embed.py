"""CPU embedding (bge) and cross-encoder reranking, lazily loaded."""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence

import numpy as np

from galrag.config import Settings, get_settings


def _set_threads(n: int) -> None:
    import torch

    torch.set_num_threads(max(1, int(n)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # already started
        pass


class Embedder:
    """Wraps a SentenceTransformer; the model is loaded on first use."""

    def __init__(self, settings: Settings | None = None, threads: int | None = None):
        self.settings = settings or get_settings()
        self.threads = threads if threads is not None else self.settings.embed_threads_per_worker
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    get_settings()  # ensure HF_HOME is set
                    _set_threads(self.threads)
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.settings.embed_model, device="cpu")
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_passages(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = self.model.encode(list(texts), batch_size=batch_size or self.settings.embed_batch_size,
                              normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        pre = self.settings.embed_query_prefix
        return self.embed_passages([pre + t for t in texts], batch_size=32)


# ---------------------------------------------------------------- parallel

_W: dict = {}


def _worker_init(model_name: str, threads: int, hf_home: str) -> None:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(threads)
    os.environ["HF_HOME"] = hf_home
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _set_threads(threads)
    from sentence_transformers import SentenceTransformer

    _W["m"] = SentenceTransformer(model_name, device="cpu")


def _worker_encode(args) -> np.ndarray:
    texts, batch_size = args
    v = _W["m"].encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(v, dtype=np.float32)


class PoolEmbedder:
    """Persistent pool of ``n_workers`` spawned processes, each holding one model.

    Use as a context manager when embedding many batches: worker start-up (spawn +
    model load) is paid once instead of per call.  Order of outputs is preserved.
    """

    def __init__(self, n_workers: int | None = None, threads_per_worker: int | None = None,
                 batch_size: int | None = None, shard_size: int | None = None,
                 settings: Settings | None = None):
        s = settings or get_settings()
        self.settings = s
        self.n_workers = max(1, n_workers or s.embed_workers)
        self.threads = threads_per_worker or s.embed_threads_per_worker
        self.batch_size = batch_size or s.embed_batch_size
        self.shard_size = shard_size or self.batch_size * 4
        self._pool = None
        self._local: Embedder | None = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _ensure(self):
        if self.n_workers == 1:
            if self._local is None:
                self._local = Embedder(self.settings, threads=self.threads)
            return
        if self._pool is None:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            hf_home = os.environ.get("HF_HOME", str(self.settings.models_dir))
            self._pool = ctx.Pool(self.n_workers, initializer=_worker_init,
                                  initargs=(self.settings.embed_model, self.threads, hf_home))

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        self._ensure()
        if self._local is not None:
            return self._local.embed_passages(texts, batch_size=self.batch_size)
        shard = self.shard_size
        shards = [(texts[i:i + shard], self.batch_size) for i in range(0, len(texts), shard)]
        parts = list(self._pool.imap(_worker_encode, shards))  # imap preserves order
        return np.concatenate(parts, axis=0)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


def embed_parallel(texts: Sequence[str], n_workers: int | None = None,
                   threads_per_worker: int | None = None, batch_size: int | None = None,
                   shard_size: int | None = None, settings: Settings | None = None) -> np.ndarray:
    """One-shot parallel embedding (spawns and tears down a :class:`PoolEmbedder`)."""
    with PoolEmbedder(n_workers, threads_per_worker, batch_size, shard_size, settings) as pe:
        return pe.embed_passages(texts)


class Reranker:
    """Cross-encoder reranker (lazy)."""

    def __init__(self, settings: Settings | None = None, threads: int | None = None):
        self.settings = settings or get_settings()
        self.threads = threads if threads is not None else self.settings.embed_threads_per_worker
        self._model = None

    @property
    def model(self):
        if self._model is None:
            get_settings()
            _set_threads(self.threads)
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.settings.rerank_model, device="cpu")
        return self._model

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict([(query, p) for p in passages], batch_size=16,
                                    show_progress_bar=False)
        return [float(x) for x in np.asarray(scores).reshape(-1)]
