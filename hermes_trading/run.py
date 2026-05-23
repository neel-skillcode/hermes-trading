"""Entry point — loads goal.yaml and starts the async trading loop."""
import asyncio
import argparse
from pathlib import Path

import yaml
from rich.console import Console

from hermes_trading.loop import TradingLoop

console = Console()


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

    loop = TradingLoop(goal, dry_run=args.dry_run)
    asyncio.run(loop.run(single_tick=args.single_tick))


if __name__ == "__main__":
    main()
