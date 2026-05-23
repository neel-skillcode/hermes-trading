"""
Reflection engine — two modes:
  --fallback  deterministic rules, used before Hermes is installed
  --hermes    AI-powered, calls `hermes` subprocess with full context

Both modes:
  - Read last N trades + current strategy
  - Fetch recent news for context
  - Change exactly ONE variable
  - Bump version, archive prior, append hypothesis
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from hermes_trading import score as score_mod
from hermes_trading.adapters import news as news_adapter

STATE_DIR = Path(__file__).parent.parent / "state"
STRATEGY_FILE = STATE_DIR / "strategy.yaml"
TRADES_FILE = STATE_DIR / "trades.jsonl"
HYPOTHESES_FILE = STATE_DIR / "hypotheses.jsonl"
HISTORY_DIR = STATE_DIR / "history"
GOAL_FILE = STATE_DIR / "goal.yaml"


def _load_strategy() -> dict:
    with open(STRATEGY_FILE) as f:
        return yaml.safe_load(f)


def _load_goal() -> dict:
    with open(GOAL_FILE) as f:
        return yaml.safe_load(f)


def _load_trades(n: int = 25) -> list[dict]:
    if not TRADES_FILE.exists():
        return []
    lines = TRADES_FILE.read_text().strip().splitlines()
    trades = [json.loads(l) for l in lines if l.strip()]
    return trades[-n:]


def _bump_version(current: str) -> str:
    try:
        n = int(current.lstrip("0") or "0")
    except ValueError:
        n = 0
    return str(n + 1).zfill(2)


def _archive_strategy(strategy: dict):
    HISTORY_DIR.mkdir(exist_ok=True)
    ver = strategy.get("version", "00")
    dest = HISTORY_DIR / f"v{ver}.yaml"
    with open(dest, "w") as f:
        yaml.dump(strategy, f, default_flow_style=False)


def _save_strategy(strategy: dict):
    with open(STRATEGY_FILE, "w") as f:
        yaml.dump(strategy, f, default_flow_style=False)


def _append_hypothesis(hypothesis: dict):
    with open(HYPOTHESES_FILE, "a") as f:
        f.write(json.dumps(hypothesis) + "\n")


def _fallback_reflect(trades: list[dict], strategy: dict, goal: dict) -> dict:
    """
    Deterministic single-variable adjustment.
    Rule: if return < target → loosen entry threshold
          if drawdown > soft ceiling → tighten stop loss
    Returns the mutation dict describing the change.
    """
    s = score_mod.score(trades, goal)
    target = goal.get("target_return_14d", 0.45)
    soft_ceil = goal.get("max_drawdown", {}).get("soft_ceiling", 0.225)

    ret = sum(t.get("pnl_pct", 0.0) for t in trades)
    dd_trades = [t for t in trades if t.get("status") == "closed"]

    if ret < target * 0.5:
        # Not enough return — loosen entry to get more trades
        old = strategy["entry"]["indicators"]["rsi"]["oversold"]
        new = min(45, old + 2)
        variable = "entry.indicators.rsi.oversold"
        old_val, new_val = old, new
        strategy["entry"]["indicators"]["rsi"]["oversold"] = new
        reasoning = f"Realised return {ret:.1%} below 50% of target {target:.1%}. Loosening RSI oversold threshold {old}→{new} to increase entry frequency."
    elif ret < 0 and score_mod._max_drawdown(dd_trades) > soft_ceil:
        # Drawdown too high — tighten stop loss ATR multiplier
        old = strategy["position_sizing"].get("max_position_pct", 0.20)
        new = round(max(0.05, old - 0.03), 3)
        variable = "position_sizing.max_position_pct"
        old_val, new_val = old, new
        strategy["position_sizing"]["max_position_pct"] = new
        reasoning = f"Drawdown above soft ceiling {soft_ceil:.1%}. Reducing max position size {old:.0%}→{new:.0%}."
    else:
        # Doing OK — tighten entry slightly for quality
        old = strategy["entry"].get("min_signal_confluence", 2)
        new = min(3, old + 1)
        variable = "entry.min_signal_confluence"
        old_val, new_val = old, new
        strategy["entry"]["min_signal_confluence"] = new
        reasoning = f"Score {s:.3f} is acceptable. Raising signal confluence bar {old}→{new} to improve trade quality."

    return {
        "variable": variable,
        "old_value": old_val,
        "new_value": new_val,
        "reasoning": reasoning,
    }


def _build_hermes_prompt(trades: list[dict], strategy: dict, goal: dict, news_context: list[str]) -> str:
    trade_summary = json.dumps(trades[-25:], indent=2)
    strategy_yaml = yaml.dump(strategy, default_flow_style=False)
    goal_yaml = yaml.dump(goal, default_flow_style=False)
    s = score_mod.score(trades, goal)
    regime = score_mod.regime(trades)

    news_block = "\n".join(f"  - {h}" for h in news_context[:20]) if news_context else "  (no recent headlines)"

    return f"""You are the reflection engine of a self-improving trading agent.

CURRENT SCORE: {s:.4f} (range -1 to +1, target > 0.5)
MARKET REGIME: {regime}

GOAL:
{goal_yaml}

CURRENT STRATEGY:
{strategy_yaml}

RECENT NEWS HEADLINES (use these to contextualise performance):
{news_block}

LAST 25 CLOSED TRADES:
{trade_summary}

TASK:
1. Analyse the trade outcomes in the context of the news and market regime.
2. Identify the ONE variable in the strategy that, if changed, would most improve the score.
3. Propose exactly ONE change. Do not suggest more than one.
4. State your confidence (0–1) and your reasoning.

Respond in this exact JSON format (no markdown, no extra text):
{{
  "variable": "dot.path.to.variable",
  "old_value": <current value>,
  "new_value": <proposed value>,
  "confidence": <0.0–1.0>,
  "reasoning": "<one paragraph>"
}}"""


async def _collect_news_context(trades: list[dict]) -> list[str]:
    assets = list({t.get("asset") for t in trades if t.get("asset")})[:5]
    headlines = []
    for asset in assets:
        try:
            nd = await news_adapter.fetch(asset)
            headlines.extend(nd.get("headlines", []))
        except Exception:
            pass
    return headlines


def run_fallback():
    strategy = _load_strategy()
    goal = _load_goal()
    trades = _load_trades(25)

    if not trades:
        print("No trades yet — skipping reflection.")
        return

    mutation = _fallback_reflect(trades, strategy, goal)

    old_ver = strategy.get("version", "00")
    new_ver = _bump_version(old_ver)
    _archive_strategy({**strategy, "version": old_ver})
    strategy["version"] = new_ver
    _save_strategy(strategy)

    hypothesis = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "fallback",
        "version_from": old_ver,
        "version_to": new_ver,
        "variable": mutation["variable"],
        "old_value": mutation["old_value"],
        "new_value": mutation["new_value"],
        "reasoning": mutation["reasoning"],
        "confidence": None,
    }
    _append_hypothesis(hypothesis)

    print(f"✓ Reflection complete (fallback mode)")
    print(f"  Strategy: v{old_ver} → v{new_ver}")
    print(f"  Changed:  {mutation['variable']} {mutation['old_value']} → {mutation['new_value']}")
    print(f"  Reason:   {mutation['reasoning']}")


def run_hermes():
    strategy = _load_strategy()
    goal = _load_goal()
    trades = _load_trades(25)

    if not trades:
        print("No trades yet — skipping reflection.")
        return

    news_context = asyncio.run(_collect_news_context(trades))
    prompt = _build_hermes_prompt(trades, strategy, goal, news_context)

    try:
        result = subprocess.run(
            ["hermes", "--json", "--prompt", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"Hermes subprocess failed: {result.stderr}")
            sys.exit(1)
        mutation = json.loads(result.stdout.strip())
    except subprocess.TimeoutExpired:
        print("Hermes timed out — falling back to deterministic reflection.")
        run_fallback()
        return
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Could not parse Hermes output: {e}\nFalling back.")
        run_fallback()
        return

    # Apply the single variable change via dot-path
    keys = mutation["variable"].split(".")
    node = strategy
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = mutation["new_value"]

    old_ver = strategy.get("version", "00")
    new_ver = _bump_version(old_ver)
    _archive_strategy({**strategy, "version": old_ver})
    strategy["version"] = new_ver
    _save_strategy(strategy)

    hypothesis = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "hermes",
        "version_from": old_ver,
        "version_to": new_ver,
        **mutation,
    }
    _append_hypothesis(hypothesis)

    print(f"✓ Reflection complete (Hermes mode)")
    print(f"  Strategy: v{old_ver} → v{new_ver}")
    print(f"  Changed:  {mutation['variable']} {mutation.get('old_value')} → {mutation['new_value']}")
    print(f"  Confidence: {mutation.get('confidence')}")
    print(f"  Reason:   {mutation.get('reasoning', '')[:120]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fallback", action="store_true")
    group.add_argument("--hermes", action="store_true")
    args = parser.parse_args()

    if args.fallback:
        run_fallback()
    else:
        run_hermes()
