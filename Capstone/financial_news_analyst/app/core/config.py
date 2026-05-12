"""
app/core/config.py
------------------
Centralised configuration loaded from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── LLM ──────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))

    # ── Market data ───────────────────────────────────────────────────────────
    FINNHUB_API_KEY: str = os.getenv("FINNHUB_API_KEY", "")

    # ── News ──────────────────────────────────────────────────────────────────
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

    # ── RAG ───────────────────────────────────────────────────────────────────
    QDRANT_PATH: str = os.getenv("QDRANT_PATH", "./qdrant_db")
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "4"))

    # ── App ───────────────────────────────────────────────────────────────────
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))


settings = Settings()
