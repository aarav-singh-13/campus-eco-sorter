@echo off
REM Prints the model's test accuracy and "correct bin chosen" % (same test split as the Kaggle notebook)
cd /d "%~dp0"
if not exist backend\.venv\Scripts\python.exe (
  echo Run run.bat once first so the Python environment exists.
  pause
  exit /b 1
)
backend\.venv\Scripts\python.exe -c "import sklearn" 2>nul || backend\.venv\Scripts\pip.exe install scikit-learn
backend\.venv\Scripts\python.exe training\test_accuracy.py
pause
