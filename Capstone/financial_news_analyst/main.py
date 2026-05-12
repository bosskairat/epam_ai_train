"""
main.py
--------
Application entry point.

Usage:
    python main.py    # starts FastAPI server on port 8000
"""

import uvicorn
from app.api.app import create_app
from app.core.logger import get_logger

logger = get_logger("main")

if __name__ == "__main__":
    logger.info("Starting FastAPI server on http://0.0.0.0:8000")
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
