
echo ==============================
echo Starting Frontend server...
echo ==============================
start cmd /k "cd frontend\SE-THESIS && npm run dev"

echo ==============================
echo Starting Backend server...
echo ==============================
start cmd /k "cd server && npm run dev"

echo ==============================
echo Starting Python server...
echo ==============================
start cmd /k "cd backend && call venv\Scripts\activate.bat && uvicorn app:app --host 0.0.0.0 --port 8000 --reload"
