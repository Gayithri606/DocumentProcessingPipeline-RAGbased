import logging
import os
from datetime import timedelta
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field




load_dotenv(dotenv_path="./.env")


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3


class OpenAISettings(LLMSettings):
    """OpenAI-specific settings extending LLMSettings."""

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = Field(default="gpt-4o")
    embedding_model: str = Field(default="text-embedding-3-small")


class DatabaseSettings(BaseModel):
    """Database connection settings."""

    service_url: str = Field(default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL"))


class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""

    #table_name: str = "embeddings"
    table_name: str = "document_embeddings"  # changed to this temporarily for testing
    embedding_dimensions: int = 1536
    time_partition_interval: timedelta = timedelta(days=7)

class ChunkingSettings(BaseModel):
    """Settings for the HybridChunker."""
    embedding_model: str = "text-embedding-3-small"  # must match your OpenAI embedding model
    max_tokens: int = 8191  # text-embedding-3-small supports up to 8191 tokens

class RedisSettings(BaseModel):
    """Settings for the Redis broker/backend."""
    url: str = Field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))

class LangfuseSettings(BaseModel):
    """Settings for Langfuse observability."""
    public_key: str = Field(default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", ""))
    secret_key: str = Field(default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", ""))
    host: str = Field(default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com"))

class Settings(BaseModel):
    """Main settings class combining all sub-settings."""

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)  
    redis: RedisSettings = Field(default_factory=RedisSettings) 
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)



@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    settings = Settings()
    setup_logging()
    return settings
