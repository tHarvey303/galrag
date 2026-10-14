"""Runtime configuration.

Everything is overridable through environment variables prefixed ``GALRAG_``
or a ``.env`` file in the working directory / repo root.  Use
:func:`get_settings` (cached) rather than instantiating ``Settings`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

# The corpus definition.  Kept here (not in code paths) so it can be tuned.
TOPIC_TERMS = (
    '"high redshift" OR "high-redshift" OR "early universe" OR reionization OR '
    '"cosmic dawn" OR "cosmic noon" OR "Lyman break" OR "Lyman-break" OR '
    '"Lyman alpha emitter" OR "Lyman-alpha emitter" OR JWST OR "epoch of reionization"'
)
TOPIC_QUERY = f"collection:astronomy AND abs:(galaxy OR galaxies) AND abs:({TOPIC_TERMS})"

# Fields requested from ADS for every record.  ``reference`` is the cited-bibcode list.
ADS_FIELDS = [
    "bibcode", "alternate_bibcode", "title", "author", "first_author", "year", "pubdate",
    "entry_date", "abstract", "doi", "identifier", "arxiv_class", "keyword", "bibstem", "pub",
    "volume", "page", "doctype", "property", "esources", "citation_count", "read_count",
    "reference",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GALRAG_",
        env_file=(".env", str(REPO_ROOT / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- credentials / identity ---
    ads_token: str = Field(default="", description="NASA ADS / SciX API token")
    contact: str = Field(default="", description="Contact string placed in the User-Agent")

    # --- paths ---
    # ``data_dir`` is the archive on shared network storage (Nexus): the MCP server reads it,
    # and the loops push finished results into it in bulk (``galrag sync push``).
    data_dir: Path = Field(default=REPO_ROOT / "data")
    # ``work_dir`` is machine-local scratch for hot I/O: the live SQLite and the raw
    # HTML/PDF cache.  Unset = everything lives in ``data_dir`` (single-machine layout).
    work_dir: Path | None = Field(default=None, description="Local scratch dir (GALRAG_WORK_DIR)")
    raw_archive_mb: int = 500        # tar the local raw cache to data_dir/raw_archive once it exceeds this
    # Explicit LanceDB location (GALRAG_LANCEDB_DIR), e.g. local disk on the GPU machine.
    lancedb_dir_override: Path | None = Field(default=None, validation_alias="GALRAG_LANCEDB_DIR")

    # --- corpus ---
    topic_query: str = TOPIC_QUERY
    year_min: int = 2000
    tier_a_year_min: int = 2015
    tier_a_min_citations: int = 50
    review_bibstems: tuple[str, ...] = ("ARA&A", "A&ARv", "PhR", "NewAR", "SSRv")

    # --- network etiquette ---
    ads_rows_per_page: int = 2000
    arxiv_delay_s: float = 4.0       # min seconds between requests to arxiv.org
    ar5iv_delay_s: float = 2.0
    pub_delay_s: float = 5.0
    http_timeout_s: float = 60.0
    max_pdf_mb: int = 60
    fetch_pub_pdf: bool = True        # try publisher PDFs via the ADS link gateway when no arXiv version

    # --- text processing ---
    chunk_target_tokens: int = 350
    chunk_max_tokens: int = 480
    chunk_overlap_paragraphs: int = 1

    # --- embeddings / index ---
    embed_model: str = "BAAI/bge-base-en-v1.5"
    embed_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embed_batch_size: int = 64
    embed_batch_size_gpu: int = 256
    embed_device: str = "auto"       # auto | cpu | cuda | cuda:N
    embed_fp16: bool = True          # half precision on CUDA
    embed_workers: int = 2           # processes; this shared node gives one session ~3 cores, so keep it lean
    embed_threads_per_worker: int = 4   # raise both (e.g. 12x8) on a quiet machine
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_enabled: bool = True
    rerank_top_n: int = 50

    @field_validator("work_dir", "lancedb_dir_override", mode="before")
    @classmethod
    def _empty_is_none(cls, v):
        return None if v is None or (isinstance(v, str) and not v.strip()) else v

    # --- derived paths ---
    @property
    def sqlite_path(self) -> Path:
        """The live SQLite (local scratch when ``work_dir`` is set)."""
        return (self.work_dir or self.data_dir) / "galrag.sqlite"

    @property
    def archive_sqlite_path(self) -> Path:
        """The SQLite copy on the shared data dir (= ``sqlite_path`` without a work dir)."""
        return self.data_dir / "galrag.sqlite"

    @property
    def raw_dir(self) -> Path:
        """Raw HTML/PDF cache: machine-local, never synced."""
        return (self.work_dir or self.data_dir) / "raw"

    @property
    def docs_dir(self) -> Path:
        """Legacy per-document JSON directory; only read by ``galrag migrate-docs``."""
        return self.data_dir / "docs"

    @property
    def lancedb_dir(self) -> Path:
        return self.lancedb_dir_override or self.archive_lancedb_dir

    @property
    def raw_archive_dir(self) -> Path:
        """Where :func:`galrag.sync.archive_raw` puts tarballs of the raw cache (on Nexus)."""
        return self.data_dir / "raw_archive"

    @property
    def archive_lancedb_dir(self) -> Path:
        return self.data_dir / "lancedb"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def user_agent(self) -> str:
        from galrag import __version__

        base = f"galrag/{__version__} (+https://github.com/tharvey/galrag; python-httpx)"
        return f"{base} {self.contact}".strip()

    @property
    def chunks_table(self) -> str:
        return "chunks__" + self.embed_model.replace("/", "__").replace("-", "_").replace(".", "_")

    @property
    def papers_table(self) -> str:
        return "papers__" + self.embed_model.replace("/", "__").replace("-", "_").replace(".", "_")

    def ensure_dirs(self) -> None:
        dirs = [self.data_dir, self.raw_dir / "html", self.raw_dir / "pdf", self.lancedb_dir,
                self.models_dir, self.logs_dir]
        if self.work_dir is not None:
            dirs.append(self.work_dir)
        for p in dirs:
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    # Keep HF downloads inside the data dir so the corpus is self-contained.
    import os

    os.environ.setdefault("HF_HOME", str(s.models_dir))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return s
