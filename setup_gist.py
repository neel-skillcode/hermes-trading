"""
One-time setup: creates the GitHub Gist for state persistence.

Usage:
  GITHUB_TOKEN=ghp_xxxx python setup_gist.py

Prints the GIST_ID you need to set in Koyeb.
"""
import asyncio
import os
from pathlib import Path

async def main():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("ERROR: set GITHUB_TOKEN env var first.")
        print("  Create one at: https://github.com/settings/tokens")
        print("  Required scopes: gist")
        raise SystemExit(1)

    from hermes_trading.adapters.gist_state import create_gist
    state_dir = Path(__file__).parent / "state"
    gist_id = await create_gist(state_dir)
    print()
    print("=" * 60)
    print(f"  GITHUB_GIST_ID = {gist_id}")
    print("=" * 60)
    print()
    print("Add these two env vars to your Koyeb service:")
    print(f"  GITHUB_TOKEN    = {token[:8]}…  (keep this secret)")
    print(f"  GITHUB_GIST_ID  = {gist_id}")
    print()
    print("Dashboard URL (GitHub Pages):")
    print(f"  https://YOUR_USERNAME.github.io/hermes-trading/?gist={gist_id}")

asyncio.run(main())
