@echo off
REM Converts the trained model to ONNX (small enough for Vercel) and checks it matches PyTorch.
cd /d "%~dp0"
if not exist backend\.venv\Scripts\python.exe (
  echo Run run.bat once first so the Python environment exists.
  pause
  exit /b 1
)
echo Installing ONNX tools (one time)...
backend\.venv\Scripts\python.exe -m pip install -q onnx onnxruntime onnxscript
backend\.venv\Scripts\python.exe training\export_onnx.py
pause
