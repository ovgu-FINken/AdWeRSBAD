#!/bin/bash

echo "Starting setup and execution..."

# Set up backend
echo "Setting up backend..."
cd backend || exit 1

# Start FastAPI backend
echo "Starting FastAPI backend..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

#give backend time to connect to db
sleep 3

# Set up frontend
echo "Setting up frontend..."
cd frontend || exit 1

# Install Node.js dependencies
echo "📦 Installing frontend dependencies..."
npm install

# Start frontend
echo "🚀 Starting frontend..."
npm run dev &
FRONTEND_PID=$!

echo "✅ Setup complete. Backend running on http://127.0.0.1:8000, Frontend running on http://localhost:3000"
sleep 3

read -p "Press [Enter] to stop the application..."

kill $BACKEND_PID
kill $FRONTEND_PID

echo "🛑 Backend and frontend stopped."
