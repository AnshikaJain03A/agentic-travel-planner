"""
Centralised application configuration.

All runtime configuration is loaded from environment variables (optionally via a
local ``.env`` file) using ``pydantic-settings``. Importing :data:`settings`
anywhere in the codebase guarantees a single, validated source of truth.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings sourced from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- LLM provider configuration ----
    llm_provider: str = Field(default="openai", description="openai | anthropic | google")
    openai_model: str = Field(default="gpt-4o")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20240620")
    google_model: str = Field(default="gemini-3.5-flash")
    llm_temperature: float = Field(default=0.4, ge=0.0, le=1.0)

    # ---- API keys ----
    openai_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")
    google_api_key: str = Field(default="")
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    google_embedding_model: str = Field(default="models/gemini-embedding-001")

    # ---- Storage ----
    sqlite_db_path: str = Field(default="./data/travel_planner.db")
    chroma_persist_dir: str = Field(default="./data/chroma")
    rag_docs_dir: str = Field(default="./rag/documents")

    # ---- API server ----
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_base_url: str = Field(default="http://localhost:8000")

    # ---- Behaviour ----
    allow_mock_fallback: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # ---- Derived helpers ----
    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.startswith("sk-"))

    @property
    def has_anthropic_key(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_api_key.startswith("sk-"))

    @property
    def has_google_key(self) -> bool:
        # Google AI Studio keys typically start with "AIza"; accept any non-empty value.
        return bool(self.google_api_key and self.google_api_key.strip())

    @property
    def active_model(self) -> str:
        provider = self.llm_provider.lower()
        if provider == "anthropic":
            return self.anthropic_model
        if provider in ("google", "gemini"):
            return self.google_model
        return self.openai_model

    def ensure_directories(self) -> None:
        """Create the storage directories if they do not yet exist."""
        Path(self.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        Path(self.rag_docs_dir).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton :class:`Settings` instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings


# Convenient module-level singleton.
settings = get_settings()
