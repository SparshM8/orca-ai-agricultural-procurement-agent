"""Core configuration and settings for ORCA."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable override support."""

    APP_NAME: str = "ORCA - Global AI Agricultural Procurement Agent"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    DATABASE_URL: str = "sqlite+aiosqlite:///./orca_local.db"
    DEFAULT_REGION: str = "GLOBAL_DEFAULT"
    DEFAULT_LANGUAGE: str = "en"

    # AI / LLM Configuration
    AI_PROVIDER: str = "rule_based"  # "rule_based", "openai_compatible", "ollama", "gemini"
    AI_BASE_URL: Optional[str] = None  # e.g., "http://localhost:11434/v1"
    AI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    AI_MODEL_NAME: str = "llama3.2"
    AI_TEMPERATURE: float = 0.1
    AI_TIMEOUT_SECONDS: float = 10.0
    AI_FALLBACK_ON_FAILURE: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
