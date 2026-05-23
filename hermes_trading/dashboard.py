"""
Live dashboard server — FastAPI + SSE.
Reads state files from /app/state/ and streams updates to the browser.
Runs alongside the trading worker on port 8080.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# On Fly.io the volume is at /app/state; locally it's alongside the package root.
_pkg_root = Path(__file__).parent.parent
STATE_DIR = Path(os.getenv("HERMES_STATE_DIR", str(_pkg_root / "state")))
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Hermes Trading Dashboard")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _read_jsonl(path: Path, n: int = 100) -> list[dict]:
    try:
        lines = path.read_text().strip().splitlines()
        return [json.loads(l) for l in lines[-n:] if l.strip()]
    except Exception:
        return []


def _read_yaml(path: Path) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _build_state() -> dict:
    heartbeat = _read_json(STATE_DIR / "heartbeat.json")
    trades = _read_jsonl(STATE_DIR / "trades.jsonl", 200)
    strategy = _read_yaml(STATE_DIR / "strategy.yaml")
    goal = _read_yaml(STATE_DIR / "goal.yaml")
    hypotheses = _read_jsonl(STATE_DIR / "hypotheses.jsonl", 20)

    closed = [t for t in trades if t.get("status") == "closed"]
    open_pos = [t for t in trades if t.get("status") == "open"]

    wins = [t for t in closed if t.get("pnl_pct", 0) > 0]
    win_rate = len(wins) / len(closed) if closed else 0.0

    total_pnl = sum(t.get("pnl_usd", 0) for t in closed)
    starting = goal.get("starting_balance", 100_000)
    pnl_pct = total_pnl / starting if starting else 0.0

    # Equity curve — running balance over time
    equity_curve = []
    balance = starting
    for t in closed:
        balance += t.get("pnl_usd", 0)
        equity_curve.append({
            "time": t.get("exit_time", ""),
            "balance": round(balance, 2),
        })

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "heartbeat": heartbeat,
        "goal": {
            "target_pct": goal.get("target_return_14d", 0.45) * 100,
            "timeframe_days": goal.get("timeframe_days", 14),
            "soft_dd_ceiling": goal.get("max_drawdown", {}).get("soft_ceiling", 0.225) * 100,
            "starting_balance": starting,
        },
        "portfolio": {
            "balance": heartbeat.get("portfolio_balance", starting),
            "peak": heartbeat.get("portfolio_peak", starting),
            "drawdown_pct": heartbeat.get("portfolio_drawdown_pct", 0),
            "total_pnl_usd": round(total_pnl, 2),
            "total_pnl_pct": round(pnl_pct * 100, 3),
            "win_rate": round(win_rate * 100, 1),
            "total_trades": heartbeat.get("total_trades", 0),
            "open_positions": heartbeat.get("open_positions", 0),
        },
        "equity_curve": equity_curve[-200:],
        "open_positions": open_pos,
        "recent_trades": list(reversed(closed[-50:])),
        "strategy": strategy,
        "hypotheses": list(reversed(hypotheses)),
        "mode": os.getenv("HERMES_TRADING_MODE", "paper"),
        "status": heartbeat.get("status", "unknown"),
        "last_tick": heartbeat.get("last_tick"),
        "last_reflection": heartbeat.get("last_reflection"),
        "trades_since_reflection": heartbeat.get("trades_since_last_reflection", 0),
    }


@app.get("/api/state")
async def get_state():
    return _build_state()


@app.get("/api/stream")
async def stream():
    async def event_generator():
        while True:
            try:
                data = json.dumps(_build_state())
                yield f"data: {data}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            await asyncio.sleep(5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Dashboard loading...</h1>")
