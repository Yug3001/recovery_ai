"""
config.py
─────────
Central settings loaded from .env (or environment).
All modules import `settings` from here — no scattered os.getenv() calls.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(ROOT_DIR, ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Groq
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # MySQL
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_DATABASE: str = "recovery_ai"

    # App
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # Model
    MODEL_PATH: str = "model/retry_model.pkl"


settings = Settings()
