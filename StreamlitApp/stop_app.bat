@echo off
echo Stopping Multilingual Abuse Detection App...

echo Stopping Streamlit (Frontend)...
taskkill /F /IM streamlit.exe /T 2>nul
powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'streamlit' -and $_.Name -eq 'python.exe' } | Stop-Process -Force -ErrorAction SilentlyContinue"

echo Stopping FastAPI (Backend)...
taskkill /F /IM uvicorn.exe /T 2>nul
powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'uvicorn' -and $_.Name -eq 'python.exe' } | Stop-Process -Force -ErrorAction SilentlyContinue"

echo Both services have been stopped.
pause
