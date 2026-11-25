"""Environment-driven application configuration."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

LLMProvider = Literal["groq", "ollama", "openai"]

_COLLECTION_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{1,61}[A-Za-z0-9]")
_SUPPORTED_PROVIDERS = {"groq", "ollama", "openai"}
_PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "llama-3.1-8b-instant"),
    "ollama": ("http://localhost:11434/v1", "llama3.2"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
}
_PROVIDER_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped or default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime settings.

    Construct directly in tests, or call :meth:`from_env` in production.
    """

    environment: str = "development"
    chroma_dir: Path = Path("data/chroma")
    documents_dir: Path = Path("documents")
    auto_ingest: bool = True
    collection_name: str = "docuchat"
    embedding_dimensions: int = 384
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_k: int = 4
    max_upload_mb: int = 20
    llm_provider: LLMProvider = "groq"
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.1-8b-instant"
    llm_timeout_seconds: float = 60.0
    critic_max_revisions: int = 1

    def __post_init__(self) -> None:
        for field_name in ("chroma_dir", "documents_dir"):
            value = getattr(self, field_name)
            if isinstance(value, str):
                object.__setattr__(self, field_name, Path(value).expanduser())

        if not _COLLECTION_NAME_PATTERN.fullmatch(self.collection_name):
            raise ValueError(
                "collection_name must be 3-63 characters of letters, digits, "
                "underscores, or hyphens, and must start and end with a letter or digit"
            )
        if self.embedding_dimensions < 16:
            raise ValueError("embedding_dimensions must be at least 16")
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if self.retrieval_k < 1:
            raise ValueError("retrieval_k must be at least 1")
        if self.max_upload_mb < 1:
            raise ValueError("max_upload_mb must be at least 1")
        if self.llm_provider not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"llm_provider must be one of {sorted(_SUPPORTED_PROVIDERS)}, "
                f"got {self.llm_provider!r}"
            )
        if self.llm_timeout_seconds <= 0:
            raise ValueError("llm_timeout_seconds must be greater than 0")
        if self.llm_api_key is not None and not self.llm_api_key.strip():
            raise ValueError("llm_api_key must not be blank when set")
        if not self.llm_base_url.strip():
            raise ValueError("llm_base_url must not be blank")
        if not self.llm_model.strip():
            raise ValueError("llm_model must not be blank")
        if not 0 <= self.critic_max_revisions <= 3:
            raise ValueError("critic_max_revisions must be between 0 and 3")

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables.

        Provider defaults:

        - ``groq``: ``GROQ_API_KEY``, ``llama-3.1-8b-instant``
        - ``ollama``: local ``llama3.2``; a dummy key is supplied because the
          OpenAI-compatible client requires one
        - ``openai``: ``OPENAI_API_KEY``, ``gpt-4o-mini``

        ``LLM_API_KEY`` overrides the provider-specific key.
        """

        provider = _env_str("LLM_PROVIDER", "groq").lower()
        if provider not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"LLM_PROVIDER must be one of {sorted(_SUPPORTED_PROVIDERS)}, "
                f"got {provider!r}"
            )
        default_base_url, default_model = _PROVIDER_DEFAULTS[provider]
        api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv(_PROVIDER_KEY_ENV[provider])
            or ("ollama" if provider == "ollama" else "")
        ).strip()

        return cls(
            environment=_env_str("DOCUCHAT_ENV", "development"),
            chroma_dir=Path(_env_str("CHROMA_DIR", "data/chroma")).expanduser(),
            documents_dir=Path(_env_str("DOCUMENTS_DIR", "documents")).expanduser(),
            auto_ingest=_env_bool("AUTO_INGEST", True),
            collection_name=_env_str("COLLECTION_NAME", "docuchat"),
            embedding_dimensions=_env_int("EMBEDDING_DIMENSIONS", 384),
            chunk_size=_env_int("CHUNK_SIZE", 800),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 120),
            retrieval_k=_env_int("RETRIEVAL_K", 4),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 20),
            llm_provider=provider,  # type: ignore[arg-type]
            llm_api_key=api_key or None,
            llm_base_url=_env_str("LLM_BASE_URL", default_base_url),
            llm_model=_env_str("LLM_MODEL", default_model),
            llm_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 60.0),
            critic_max_revisions=_env_int("CRITIC_MAX_REVISIONS", 1),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings, loaded once from the environment."""

    return Settings.from_env()
