"""
run.py — start FastAPI and Streamlit in parallel processes.
Usage:  python run.py
Stop:   Ctrl+C
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

FASTAPI = [
    sys.executable, "main.py",
]

STREAMLIT = [
    sys.executable, "-m", "streamlit", "run",
    str(ROOT / "ui" / "streamlit_app.py"),
    "--server.port", "8501",
    "--server.headless", "true",
]

print()
print("=" * 55)
print("  Financial News Analyst — starting services")
print("=" * 55)

procs = []
try:
    procs.append(subprocess.Popen(FASTAPI, cwd=ROOT))
    print("  FastAPI    →  http://localhost:8000/api/v1/health")

    time.sleep(1)  # give FastAPI a moment before Streamlit connects

    procs.append(subprocess.Popen(STREAMLIT, cwd=ROOT))
    print("  Streamlit  →  http://localhost:8501")

    print("=" * 55)
    print("  Press Ctrl+C to stop all services.")
    print()

    for p in procs:
        p.wait()

except KeyboardInterrupt:
    print("\nStopping services…")
    for p in procs:
        p.terminate()
    for p in procs:
        p.wait()
    print("Done.")
