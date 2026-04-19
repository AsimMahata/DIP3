@echo off
echo Starting Streamlit Frontend...
echo Connecting directly to Hugging Face Cloud Backend!

REM Change directory to the location of this batch script
cd /d "%~dp0"

echo Launching browser...
start cmd /k "..\.venv\Scripts\activate.bat && streamlit run app.py"

echo ==========================================
echo URL: http://localhost:8501
echo ==========================================
