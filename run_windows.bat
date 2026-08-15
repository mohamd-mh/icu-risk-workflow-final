@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 goto error
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 goto error

python -m pip install -r requirements.txt
if errorlevel 1 goto error

python app.py
if errorlevel 1 goto error

exit /b 0

:error
echo.
echo An error occurred while starting the app.
pause
exit /b 1
