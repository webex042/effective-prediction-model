@echo off
rem EPM Chat launcher for Windows: uses the project venv, trains starter models on first run, then starts the server.
setlocal
cd /d "%~dp0\.."
set "PY=venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
if not exist "epm_chatbot\var\models\m_*.json" (
  echo Training the starter models (first run only, about 1 minute^)...
  "%PY%" -m epm_chatbot.bootstrap
)
"%PY%" -m epm_chatbot %*
