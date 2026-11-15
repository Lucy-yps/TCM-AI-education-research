@echo off
chcp 65001 >nul
rem ============================================================
rem  TCM-AI teaching/research assistant - launcher
rem  Knowledge search : http://127.0.0.1:5000/
rem  Quiz training   : http://127.0.0.1:5000/quiz.html
rem  Close this window to stop the server.
rem ============================================================
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if not defined PORT set PORT=5000

echo [i] Knowledge search : http://127.0.0.1:%PORT%/
echo [i] Quiz training    : http://127.0.0.1:%PORT%/quiz.html
echo [i] Close this window to stop.
echo.

python server.py
if errorlevel 1 (
  echo.
  echo [x] Failed to start. Possible reasons:
  echo     1^) dependencies missing  -^>  pip install -r requirements.txt
  echo     2^) python not on PATH    -^>  try:  py server.py
  echo     3^) port %PORT% in use    -^>  set PORT=5001 ^&^& /¨.bat
  pause
)