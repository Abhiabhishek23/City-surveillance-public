#!/bin/bash
# run_local.sh - Orchestrator for the Urban Monitoring System

# --- Configuration ---
export FCM_KEY="BEKVPEj_fQjxsywFRr4DSZ4Nnz2twH3VKQZ9xmDEBw3vBQdu-8Qfdh_JGtCEZyZ8XfheBLzRb-xuio_RD6DUOJs"
export BACKEND_URL="http://127.0.0.1:8000"

# --- Start Backend ---
echo "🚀 Starting Backend Server..."
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..
echo "✅ Backend started with PID $BACKEND_PID"

# --- Start Dashboard ---
echo "🚀 Starting Dashboard..."
cd dashboard
npm install && npm start &
DASHBOARD_PID=$!
cd ..
echo "✅ Dashboard started with PID $DASHBOARD_PID"

echo -e "\n--- All services are running ---"
echo "Backend API: $BACKEND_URL"
echo "Dashboard UI: http://localhost:3000"
echo "Press CTRL+C to stop all services."

# Wait for user to exit
wait $BACKEND_PID $DASHBOARD_PID