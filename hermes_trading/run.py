"""Entry point — loads goal.yaml and starts the async trading loop."""
import asyncio
import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rich.console import Console

from hermes_trading.loop import TradingLoop

console = Console()

STATE_DIR = Path(__file__).parent.parent / "state"
HEARTBEAT_FILE = STATE_DIR / "heartbeat.json"


def _write_error_heartbeat(msg: str):
    """Write a minimal heartbeat so the commit step always has something to push."""
    try:
        existing = {}
        if HEARTBEAT_FILE.exists():
            existing = json.loads(HEARTBEAT_FILE.read_text())
        existing.update({
            "status": "error",
            "error": msg[:300],
            "last_tick": datetime.now(timezone.utc).isoformat(),
        })
        HEARTBEAT_FILE.write_text(json.dumps(existing, indent=2))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Hermes Trading Worker")
    parser.add_argument("--dry-run", action="store_true", help="Log signals without entering trades")
    parser.add_argument("--single-tick", action="store_true", help="Run exactly one tick then exit (used by GitHub Actions)")
    args = parser.parse_args()

    goal_path = Path(__file__).parent.parent / "state" / "goal.yaml"
    if not goal_path.exists():
        console.print(f"[bold red]goal.yaml not found at {goal_path}[/bold red]")
        raise SystemExit(1)

    with open(goal_path) as f:
        goal = yaml.safe_load(f)

    console.print("[bold green]Booting hermes-trading worker[/bold green]")
    console.print(f"  Universe: Forex majors + US equities (AI-selected)")
    console.print(f"  Target:   +{goal.get('target_return_14d', 0.45)*100:.0f}% in {goal.get('timeframe_days', 14)} days")
    console.print(f"  Balance:  ${goal.get('starting_balance', 100_000):,.0f} (paper)")

    try:
        loop = TradingLoop(goal, dry_run=args.dry_run)
        asyncio.run(loop.run(single_tick=args.single_tick))
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        console.print(f"[bold red]Fatal tick error: {err}[/bold red]")
        console.print(traceback.format_exc())
        _write_error_heartbeat(err)
        # Exit 0 so GitHub Actions proceeds to the "Commit state" step
        raise SystemExit(0)


if __name__ == "__main__":
    main()
