#!/bin/sh
# On first boot the mounted volume at /app/state is empty.
# Seed it from the bundled state_init/ if goal.yaml is missing.
set -e

STATE_DIR="/app/state"
INIT_DIR="/app/state_init"

if [ ! -f "$STATE_DIR/goal.yaml" ]; then
    echo "First boot — seeding state files from bundled defaults..."
    mkdir -p "$STATE_DIR/history"
    cp "$INIT_DIR/goal.yaml"      "$STATE_DIR/goal.yaml"
    cp "$INIT_DIR/strategy.yaml"  "$STATE_DIR/strategy.yaml"
    cp "$INIT_DIR/heartbeat.json" "$STATE_DIR/heartbeat.json"
    touch "$STATE_DIR/trades.jsonl"
    touch "$STATE_DIR/hypotheses.jsonl"
    echo "State seeded."
else
    echo "State volume already initialised — skipping seed."
fi

# Start the live dashboard in the background on port 8080.
# Use the venv Python directly (avoids uv-sync races with the worker's uv run).
echo "Starting dashboard on :8080..."
/app/.venv/bin/python -m uvicorn hermes_trading.dashboard:app \
    --host 0.0.0.0 --port 8080 --log-level info \
    >> "$STATE_DIR/dashboard.log" 2>&1 &
echo "Dashboard PID=$!"

# Start the trading worker in the foreground
exec uv run python -m hermes_trading.run
