
"""
MediRAG AI - Central Application Configuration.

Environment-based settings for:
    FastAPI
    OpenRouter
    PDF ingestion
    ChromaDB storage
    SentenceTransformer embeddings

Compatible with:
    app.main
    app.llm
    app.rag

Python 3.13
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

# Expected location:
# project_root/backend/app/core.py
#
# parents[0] = app
# parents[1] = backend
# parents[2] = project_root

BASE = Path(__file__).resolve().parents[2]

BACKEND_DIR = BASE / "backend"

ENV_FILE = BASE / ".env"

STORAGE = BACKEND_DIR / "storage"

CHROMA_DIR = STORAGE / "chroma"


# ---------------------------------------------------------
# Application settings
# ---------------------------------------------------------

class Settings(BaseSettings):
    """
    Central configuration loaded from environment
    variables or project-root .env file.

    Environment variables take precedence over .env.
    """

    # Application
    app_name: str = "MediRAG AI"
    app_env: Literal[
        "development",
        "testing",
        "production",
    ] = "development"

    debug: bool = False

    # OpenRouter
    openrouter_api_key: str = Field(
        default="",
        repr=False,
    )

    openrouter_model: str = (
        "openai/gpt-4o-mini"
    )

    openrouter_base_url: str = (
        "https://openrouter.ai/api/v1"
    )

    openrouter_timeout_seconds: int = Field(
        default=90,
        ge=10,
        le=300,
    )

    openrouter_max_tokens: int = Field(
        default=850,
        ge=100,
        le=8192,
    )

    # FastAPI / CORS
    cors_origins: str = (
        "http://localhost:5173,"
        "http://127.0.0.1:5173"
    )

    # Upload limits
    max_upload_mb: int = Field(
        default=15,
        ge=1,
        le=100,
    )

    # PDF processing
    pdf_parser: Literal[
        "auto",
        "pymupdf",
        "pdfplumber",
    ] = "auto"

    pdf_max_pages: int = Field(
        default=300,
        ge=1,
        le=5000,
    )

    # Embeddings
    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    embedding_batch_size: int = Field(
        default=64,
        ge=1,
        le=512,
    )

    # Chunking
    chunk_size: int = Field(
        default=950,
        ge=100,
        le=10000,
    )

    chunk_overlap: int = Field(
        default=150,
        ge=0,
        le=5000,
    )

    # Retrieval
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
    )

    # ChromaDB
    chroma_collection_name: str = (
        "guidelines"
    )

    # Logging
    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    # -----------------------------------------------------
    # Validators
    # -----------------------------------------------------

    @field_validator("openrouter_base_url")
    @classmethod
    def validate_openrouter_url(
        cls,
        value: str,
    ) -> str:
        """
        Require HTTPS for OpenRouter API requests.
        """

        value = value.strip().rstrip("/")

        if not value.startswith("https://"):
            raise ValueError(
                "OPENROUTER_BASE_URL must use HTTPS"
            )

        return value

    @field_validator("openrouter_model")
    @classmethod
    def validate_model(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "OPENROUTER_MODEL cannot be empty"
            )

        return value

    @field_validator("cors_origins")
    @classmethod
    def validate_cors(
        cls,
        value: str,
    ) -> str:
        """
        Preserve comma-separated string format
        expected by existing main.py.
        """

        origins = [
            item.strip().rstrip("/")
            for item in value.split(",")
            if item.strip()
        ]

        if not origins:
            raise ValueError(
                "At least one CORS origin is required"
            )

        if "*" in origins:
            raise ValueError(
                "Use explicit CORS origins, not '*'"
            )

        return ",".join(dict.fromkeys(origins))

    @field_validator("chroma_collection_name")
    @classmethod
    def validate_collection_name(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "CHROMA_COLLECTION_NAME cannot be empty"
            )

        return value

    # -----------------------------------------------------
    # Cross-field validation
    # -----------------------------------------------------

    def model_post_init(
        self,
        __context: object,
    ) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "CHUNK_OVERLAP must be smaller "
                "than CHUNK_SIZE"
            )

        if self.app_env == "production" and self.debug:
            raise ValueError(
                "DEBUG must be false in production"
            )

    # -----------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            item.strip()
            for item in self.cors_origins.split(",")
            if item.strip()
        ]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def storage_path(self) -> Path:
        return STORAGE

    @property
    def chroma_path(self) -> Path:
        return CHROMA_DIR

    @property
    def has_openrouter_key(self) -> bool:
        return bool(
            self.openrouter_api_key.strip()
            and self.openrouter_api_key.strip()
            not in {
                "replace_me",
                "your_actual_api_key",
                "your-openrouter-api-key",
            }
        )

    def validate_llm_configuration(self) -> None:
        """
        Validate the API key only when the LLM
        is actually needed.

        This allows PDF parsing and document
        indexing without an OpenRouter key.
        """

        if not self.has_openrouter_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing "
                "or contains a placeholder."
            )


# ---------------------------------------------------------
# Cached settings
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def settings() -> Settings:
    """
    Return cached application configuration.

    Configuration is loaded once per process.
    Restart the server after changing .env.
    """
    return Settings()


# ---------------------------------------------------------
# Storage initialization
# ---------------------------------------------------------

def initialize_storage() -> None:
    """
    Create local storage directories.

    Call during application startup when needed.
    """

    try:
        STORAGE.mkdir(
            parents=True,
            exist_ok=True,
        )

        CHROMA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    except OSError:
        LOGGER.exception(
            "Unable to initialize storage"
        )
        raise


# Preserve existing behavior:
# storage is available when app.core is imported.
initialize_storage()
