@echo off
REM Campus Eco Sorter - first run installs dependencies, then starts the server.
cd /d "%~dp0backend"
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
echo.
echo Open http://localhost:8000 in your browser  (Ctrl+C to stop)
echo.
uvicorn main:app --reload --port 8000
pause
