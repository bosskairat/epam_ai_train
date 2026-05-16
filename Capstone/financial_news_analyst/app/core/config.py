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

    # ── History ───────────────────────────────────────────────────────────────
    DB_PATH: str = os.getenv("DB_PATH", "./app.db")

    # ── App ───────────────────────────────────────────────────────────────────
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    HISTORY_RETENTION_DAYS: int = int(os.getenv("HISTORY_RETENTION_DAYS", "90"))  # ~3 months; GDPR-friendly default
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "300"))
    TOKEN_QUOTA_PER_KEY: int = int(os.getenv("TOKEN_QUOTA_PER_KEY", "0"))

    # ── RAG quality ───────────────────────────────────────────────────────────
    # 0.28 is empirically tuned: low enough to capture paraphrased context,
    # high enough to filter unrelated documents from a mixed financial corpus.
    RAG_MIN_SIMILARITY: float = float(os.getenv("RAG_MIN_SIMILARITY", "0.28"))

    # ── Content & privacy ─────────────────────────────────────────────────────
    # When enabled, checks both RAG documents on upsert and LLM output for
    # harmful content (violence, PII exfiltration, etc.) before serving.
    ENABLE_CONTENT_MODERATION: bool = os.getenv("ENABLE_CONTENT_MODERATION", "true").lower() in (
        "1", "true", "yes"
    )
    STORE_FULL_STATE: bool = os.getenv("STORE_FULL_STATE", "true").lower() in ("1", "true", "yes")
    MAX_ARTICLE_CHARS_STORED: int = int(os.getenv("MAX_ARTICLE_CHARS_STORED", "800"))

    # ── Auth (username + password → JWT) ─────────────────────────────────────
    AUTH_ENABLED: bool = True  # always required
    AUTH_BOOTSTRAP_USERNAME: str = os.getenv("AUTH_BOOTSTRAP_USERNAME", "")
    AUTH_BOOTSTRAP_PASSWORD: str = os.getenv("AUTH_BOOTSTRAP_PASSWORD", "")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "")
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))
    # Registration open to all by default when auth is enabled
    ALLOW_USER_REGISTRATION: bool = os.getenv("ALLOW_USER_REGISTRATION", "true").lower() in (
        "1", "true", "yes"
    )
    # Comma-separated usernames with admin privileges (defaults to bootstrap user)
    ADMIN_USERNAMES: str = os.getenv(
        "ADMIN_USERNAMES", os.getenv("AUTH_BOOTSTRAP_USERNAME", "")
    )

    @property
    def authorization_required(self) -> bool:
        return self.AUTH_ENABLED


settings = Settings()
