"""
Centralized application configuration using pydantic-settings.

Why pydantic-settings over python-dotenv alone:
- Type validation at startup (fail fast on bad config)
- Default values with override from .env
- Nested config groups for organization
- Auto-documentation of all config options

Why not a YAML/TOML config file:
- Environment variables are the 12-factor app standard
- Works seamlessly with Docker, CI/CD, and cloud platforms
- Secrets never touch the filesystem
"""

import os

# Suppress HuggingFace rate-limit warnings, telemetry, and progress bars
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Provider (Groq)
    groq_api_key: str = Field(
        default="",
        description="Groq API key for LLM inference"
    )
    llm_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq model identifier"
    )
    llm_temperature: float = Field(
        default=0.3,
        description="LLM temperature (0=deterministic, 1=creative)"
    )
    llm_max_tokens: int = Field(
        default=2048,
        description="Max tokens in LLM response"
    )

    # Vector Database (Qdrant)
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL (cloud or local)"
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key (required for cloud)"
    )

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )

    # Embedding Model
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace model ID for embeddings"
    )
    embedding_dimension: int = Field(
        default=384,
        description="Embedding vector dimension"
    )

    # RAG Parameters
    chunk_size: int = Field(
        default=256,
        description="Words per transcript chunk"
    )
    chunk_overlap: int = Field(
        default=50,
        description="Word overlap between chunks"
    )
    top_k: int = Field(
        default=5,
        description="Number of chunks to retrieve per query"
    )
    similarity_threshold: float = Field(
        default=0.3,
        description="Minimum cosine similarity for retrieval"
    )

    # Whisper
    whisper_model_size: str = Field(
        default="base",
        description="faster-whisper model size (tiny/base/small/medium/large)"
    )

    # App Config
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        description="Comma-separated CORS origins"
    )

    # Cache TTLs (seconds)
    transcript_cache_ttl: int = Field(default=3600)
    metadata_cache_ttl: int = Field(default=3600)
    session_ttl: int = Field(default=86400)

    # Temp directory for audio files
    temp_dir: str = Field(
        default="/tmp/rag-chatbot",
        description="Temporary directory for downloaded audio files"
    )

    @property
    def cors_origin_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @field_validator("groq_api_key")
    @classmethod
    def validate_groq_key(cls, v: str) -> str:
        """Warn if Groq API key is missing (non-fatal for testing)."""
        if not v:
            import warnings
            warnings.warn(
                "GROQ_API_KEY is not set. LLM features will not work. "
                "Get a free key at https://console.groq.com/keys"
            )
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


# Singleton instance — import this everywhere
settings = Settings(
    _env_file=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".env"
    )
)
