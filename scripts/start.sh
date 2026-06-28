#!/bin/bash
cd ~/snapon

echo "Starting SnapOn server..."
python -m server.app &
SERVER_PID=$!
echo "Server PID: $SERVER_PID — listening on http://localhost:8000"

sleep 2

echo "Starting SnapOn UI..."
python client/app.py &
UI_PID=$!
echo "UI PID: $UI_PID — listening on http://localhost:5000"

echo ""
echo "==================================="
echo " Open http://localhost:5000"
echo "==================================="
echo "Press Ctrl+C to stop both services"

trap "kill $SERVER_PID $UI_PID 2>/dev/null; echo 'Stopped.'" SIGINT SIGTERM
wait
