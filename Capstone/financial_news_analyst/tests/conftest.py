"""
tests/conftest.py
------------------
Shared pytest configuration.
Sets environment variables before any test module is imported
so the app never needs real API keys during tests.
"""

import os
import pytest

# ── Set test env vars before imports ─────────────────────────────────────────
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("NEWS_API_KEY", "")
os.environ.setdefault("LOG_LEVEL", "WARNING")  # quieter test output

