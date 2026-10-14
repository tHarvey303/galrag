"""Runtime configuration.

Everything is overridable through environment variables prefixed ``GALRAG_``
or a ``.env`` file in the working directory / repo root.  Use
:func:`get_settings` (cached) rather than instantiating ``Settings`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
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
    )

    # --- credentials / identity ---
    ads_token: str = Field(default="", description="NASA ADS / SciX API token")
    contact: str = Field(default="", description="Contact string placed in the User-Agent")

    # --- paths ---
    data_dir: Path = Field(default=REPO_ROOT / "data")

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
    embed_workers: int = 16          # processes (many small workers beat one big one on this node)
    embed_threads_per_worker: int = 8
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_enabled: bool = True
    rerank_top_n: int = 50

    # --- derived paths ---
    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "galrag.sqlite"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "docs"

    @property
    def lancedb_dir(self) -> Path:
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
        for p in (self.data_dir, self.raw_dir / "html", self.raw_dir / "pdf", self.docs_dir,
                  self.lancedb_dir, self.models_dir, self.logs_dir):
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
