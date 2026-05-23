"""
Scores a list of closed trades against goal.yaml.
Returns a float in [-1, +1]:
  +1  = all goals met or exceeded
   0  = break-even on all dimensions
  -1  = hard failure (below failure_below threshold)
"""
from __future__ import annotations

import math
from typing import Any


def _realised_return(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    pnl = sum(t.get("pnl_pct", 0.0) for t in trades)
    return pnl


def _max_drawdown(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    peak = 1.0
    equity = 1.0
    max_dd = 0.0
    for t in trades:
        equity *= 1.0 + t.get("pnl_pct", 0.0)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _sharpe(trades: list[dict]) -> float:
    if len(trades) < 2:
        return 0.0
    returns = [t.get("pnl_pct", 0.0) for t in trades]
    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance) if variance > 0 else 1e-9
    # Annualise assuming ~20 trades/day, 252 trading days
    ann_factor = math.sqrt(20 * 252)
    return (mean_r / std) * ann_factor


def score(trades: list[dict], goal: dict) -> float:
    if not trades:
        return 0.0

    closed = [t for t in trades if t.get("status") == "closed"]
    if not closed:
        return 0.0

    # Dimensions
    ret = _realised_return(closed)
    target = goal.get("target_return_14d", 0.45)
    ret_score = min(1.0, ret / target) if target > 0 else 0.0

    dd = _max_drawdown(closed)
    soft_ceil = goal.get("max_drawdown", {}).get("soft_ceiling", 0.225)
    dd_score = max(0.0, 1.0 - (dd / soft_ceil)) if soft_ceil > 0 else 1.0

    sharpe = _sharpe(closed)
    sharpe_floor = goal.get("min_sharpe", {}).get("floor", 0.8)
    sharpe_score = min(1.0, sharpe / max(sharpe_floor, 1.2)) if sharpe_floor > 0 else 1.0

    composite = ret_score * 0.5 + dd_score * 0.3 + sharpe_score * 0.2

    failure_below = goal.get("failure_below", -0.04)
    if ret < failure_below:
        composite = max(-1.0, composite - 1.0)

    return round(float(composite * 2.0 - 1.0), 4)


def regime(trades: list[dict], lookback: int = 20) -> str:
    """Simple regime classifier based on rolling returns."""
    recent = [t.get("pnl_pct", 0.0) for t in trades[-lookback:]]
    if not recent:
        return "unknown"
    avg = sum(recent) / len(recent)
    if avg > 0.01:
        return "bull"
    if avg < -0.01:
        return "bear"
    return "sideways"
