#!/bin/bash

# --- Start the FastAPI Backend Server ---
echo "Starting FastAPI RAG Backend on http://127.0.0.1:8000..."
# Assuming your FastAPI application entry point is 'main:app'
uvicorn main:app --reload &
BACKEND_PID=$!

# --- Start the Flask Frontend Server ---
echo "Starting Flask Frontend Server (app.py) on http://127.0.0.1:5000..."
# Assuming your Flask app is in 'app.py' and runs on default port 5000
# Run the Flask app in the background
python app.py & 
FRONTEND_PID=$!

# --- Wait for Servers to Start ---
# Give both servers a moment to boot up before opening the browser
sleep 5 

# --- Open the Frontend in the Browser ---
echo "Opening Frontend URL (http://127.0.0.1:5000/) in default browser..."

# Use the correct command for your OS (Mac uses 'open')
if command -v open >/dev/null; then
  # Open the URL served by the Flask app
  open http://127.0.0.1:5000/
# Include other OS checks for completeness, though Mac is the primary target
elif command -v xdg-open >/dev/null; then
  xdg-open http://127.0.0.1:5000/ # Linux
elif command -v start >/dev/null; then
  start http://127.0.0.1:5000/    # Windows (WSL/Git Bash might use 'start')
fi

echo ""
echo "--------------------------------------------------------"
echo "Both FastAPI (8000) and Flask (5000) services are running."
echo "Press Ctrl+C in this window to stop both backends."
echo "--------------------------------------------------------"
echo ""

# Wait indefinitely until the user manually stops the script
# This makes Ctrl+C signal work correctly for cleanup
wait

# --- Cleanup (Executed when user presses Ctrl+C) ---
echo "Stopping all services..."
kill $BACKEND_PID $FRONTEND_PID 2>/dev/null

echo "Services stopped."
