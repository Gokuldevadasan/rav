@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo RAV virtual environment not found: "%PYTHON%"
    echo Create the environment and install the mobile-camera dependencies first.
    pause
    exit /b 1
)

"%PYTHON%" -m rav.interfaces.mobile_camera.server --host 0.0.0.0 --port 8000 --quantization none --offline --open-browser
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo RAV server exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%
