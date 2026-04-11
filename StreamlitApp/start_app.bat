@echo off
echo Starting Multilingual Abuse Detection App...
echo Starting FastAPI Backend...
start cmd /k ".\venv\Scripts\activate.bat && uvicorn api:app --host 0.0.0.0 --port 8000"

echo Waiting 10 seconds for backend to start (large model loading)...
timeout /t 10 /nobreak

echo Starting Streamlit Frontend...
start cmd /k ".\venv\Scripts\activate.bat && streamlit run app.py"

echo Both services are now running!
echo   API:      http://localhost:8000
echo   Frontend: http://localhost:8501
