"""Core configuration and settings for ORCA."""

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
    AI_MODEL_NAME: str = "gemini-1.5-pro"
    AI_TEMPERATURE: float = 0.1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
