from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    groq_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    streaming: bool = True

    embedding_model: str = "all-MiniLM-L6-v2"

    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "documents"

    sqlite_db_path: str = "./data/copilot.db"

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 4

    upload_dir: str = "./data/uploads"
    max_file_size_mb: int = 20

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    api_base_url: str = "http://localhost:8000"

    log_level: str = "INFO"
    log_file: str = "./logs/app.log"

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chroma_path(self) -> Path:
        p = Path(self.chroma_persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def sqlite_path(self) -> Path:
        p = Path(self.sqlite_db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def log_path(self) -> Path:
        p = Path(self.log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# Convenient module-level alias used by other modules:
#   from backend.config import settings
settings = get_settings()
