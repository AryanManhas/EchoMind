@echo off
setlocal

REM Run backend and Flutter in separate terminals on Windows
cd /d "%~dp0backend"
if not exist ".venv\Scripts\activate.bat" (
  echo Virtualenv not found at %cd%\.venv\Scripts\activate.bat
  pause
  exit /b 1
)

start "EchoMInd Backend" cmd /k "cd /d %~dp0backend && .venv\Scripts\activate && uvicorn main:app --host 0.0.0.0 --port 5000"

cd /d "%~dp0mobile_app"
start "EchoMInd Mobile" cmd /k "cd /d %~dp0mobile_app && flutter pub get && flutter run -d chrome"

echo Launched backend and mobile app windows.
endlocal
