"""
DocuMind - Configuration Module
Manages application settings and environment variables using Pydantic Settings.
"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application configuration settings loaded from environment or .env file."""

    # Project metadata
    PROJECT_NAME: str = "DocuMind"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Enterprise RAG-Powered Document Intelligence Platform"

    # API Keys & LLM Settings
    GOOGLE_API_KEY: str = Field(default="", description="Google Gemini API Key")
    EMBEDDING_MODEL: str = Field(
        default="models/text-embedding-004",
        description="Google GenAI embedding model"
    )
    LLM_MODEL: str = Field(
        default="gemini-1.5-flash",
        description="Gemini generative model (e.g. gemini-1.5-flash, gemini-1.5-pro)"
    )
    LLM_TEMPERATURE: float = Field(default=0.2, description="Sampling temperature for LLM")
    MAX_OUTPUT_TOKENS: int = Field(default=2048, description="Maximum output tokens for response")

    # Document Chunking Settings
    CHUNK_SIZE: int = Field(default=1000, description="Maximum character count per text chunk")
    CHUNK_OVERLAP: int = Field(default=150, description="Character overlap between consecutive chunks")
    TOP_K_RESULTS: int = Field(default=4, description="Number of context chunks to retrieve")

    # Storage Paths
    CHROMA_PERSIST_DIRECTORY: str = Field(
        default=str(BASE_DIR / "data" / "chroma_db"),
        description="Directory for ChromaDB persistence"
    )
    UPLOAD_DIR: str = Field(
        default=str(BASE_DIR / "data" / "uploads"),
        description="Directory for uploaded source documents"
    )

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    ALLOWED_ORIGINS: List[str] = ["*"]
    MAX_UPLOAD_SIZE_MB: int = 25

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Instantiate singleton settings
settings = Settings()

# Ensure critical data directories exist
os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
