"""
main.py
--------
Application entry point.

Usage:
    python main.py              # starts FastAPI server on port 8000
    python main.py --demo       # runs example queries without starting server
"""

import sys
import uvicorn
from app.api.app import create_app
from app.core.logger import get_logger

logger = get_logger("main")


def run_demo():
    """Run a quick demo of the pipeline without the HTTP server."""
    from app.agents.supervisor_agent import run_pipeline
    from app.core.security import validate_query, ValidationError

    demo_queries = [
        "Why did Tesla stock drop today?",
        "Summarize current market sentiment for S&P 500",
    ]

    for query in demo_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print("="*60)
        try:
            q = validate_query(query)
            state = run_pipeline(q)
            analysis = state["analysis"]
            print(f"Sentiment  : {analysis.get('sentiment', 'N/A')}")
            print(f"Summary    : {analysis.get('summary', 'N/A')}")
            print(f"Key Drivers: {analysis.get('key_drivers', [])}")
            print(f"Sources    : {state['rag_sources']}")
            print(f"Tokens     : {state['token_usage']}")
            print(f"Latency    : {state['total_latency_s']}s")
        except ValidationError as e:
            print(f"Validation error: {e}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        logger.info("Running in demo mode")
        run_demo()
    else:
        logger.info("Starting FastAPI server on http://0.0.0.0:8000")
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
