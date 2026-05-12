@echo off
setlocal

set ROOT=%~dp0

echo.
echo =======================================================
echo   Financial News Analyst -- starting services
echo =======================================================

start "FastAPI" python "%ROOT%main.py"
echo   FastAPI   -^>  http://localhost:8000/api/v1/health

start "Streamlit" python -m streamlit run "%ROOT%ui\streamlit_app.py" ^
  --server.port 8501 --server.headless true
echo   Streamlit -^>  http://localhost:8501

echo =======================================================
echo   Close the service windows to stop all services.
echo.

endlocal
