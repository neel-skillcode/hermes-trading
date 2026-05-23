"""
GitHub Gist state persistence — free, no credit card required.

On each tick the trading loop calls `save()` to persist:
  heartbeat.json, trades.jsonl, strategy.yaml, hypotheses.jsonl, goal.yaml

On startup the loop calls `load()` to restore state after a Koyeb restart.

Requires two env vars:
  GITHUB_TOKEN      — Personal Access Token with 'gist' scope
  GITHUB_GIST_ID    — ID of the Gist created at startup (see README)

If either var is missing, all calls are no-ops and local files are used instead.
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx

_TOKEN   = os.getenv("GITHUB_TOKEN", "")
_GIST_ID = os.getenv("GITHUB_GIST_ID", "")

_STATE_FILES = [
    "heartbeat.json",
    "trades.jsonl",
    "strategy.yaml",
    "hypotheses.jsonl",
    "goal.yaml",
]

_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


def _enabled() -> bool:
    return bool(_TOKEN and _GIST_ID)


async def load(state_dir: Path) -> bool:
    """
    Restore state files from the Gist.
    Returns True if successful, False if skipped/failed.
    """
    if not _enabled():
        return False

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"https://api.github.com/gists/{_GIST_ID}",
                headers={**_HEADERS, "Authorization": f"Bearer {_TOKEN}"},
            )
            r.raise_for_status()
            data = r.json()

        files = data.get("files", {})
        restored = 0
        for name in _STATE_FILES:
            if name in files and files[name].get("content"):
                (state_dir / name).write_text(files[name]["content"])
                restored += 1

        print(f"[gist_state] Restored {restored} state files from Gist {_GIST_ID[:8]}…")
        return True

    except Exception as e:
        print(f"[gist_state] Load failed (using local state): {e}")
        return False


async def save(state_dir: Path) -> bool:
    """
    Push current state files to the Gist.
    Returns True if successful, False if skipped/failed.
    """
    if not _enabled():
        return False

    payload: dict[str, dict] = {}
    for name in _STATE_FILES:
        path = state_dir / name
        if path.exists():
            content = path.read_text()
            # Gist API rejects empty files — pad with a space
            payload[name] = {"content": content or " "}

    if not payload:
        return False

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.patch(
                f"https://api.github.com/gists/{_GIST_ID}",
                headers={**_HEADERS, "Authorization": f"Bearer {_TOKEN}"},
                json={"files": payload},
            )
            r.raise_for_status()
        return True

    except Exception as e:
        print(f"[gist_state] Save failed (state still on disk): {e}")
        return False


async def create_gist(state_dir: Path) -> str:
    """
    One-time helper: create the Gist and return its ID.
    Call this manually during first-time setup.
    """
    if not _TOKEN:
        raise RuntimeError("GITHUB_TOKEN env var not set")

    payload: dict[str, dict] = {}
    for name in _STATE_FILES:
        path = state_dir / name
        payload[name] = {"content": path.read_text() if path.exists() else " "}

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            "https://api.github.com/gists",
            headers={**_HEADERS, "Authorization": f"Bearer {_TOKEN}"},
            json={
                "description": "hermes-trading state — auto-managed, do not edit manually",
                "public": True,   # public so the dashboard can read it without auth
                "files": payload,
            },
        )
        r.raise_for_status()
        gist_id = r.json()["id"]

    print(f"[gist_state] Created Gist: https://gist.github.com/{gist_id}")
    print(f"[gist_state] Set GITHUB_GIST_ID={gist_id} in your Koyeb env vars")
    return gist_id
