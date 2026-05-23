"""
Macro adapter — fetches key economic indicators.
Uses FRED API (free). Falls back to static defaults if unavailable.
Set FRED_API_KEY in .env for live data (free at fred.stlouisfed.org).
"""
import asyncio
import os
from datetime import datetime, timezone

import httpx

SCHEMA_VERSION = "macro/v1"

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi_yoy": "CPIAUCSL",
    "unemployment": "UNRATE",
    "gdp_growth": "A191RL1Q225SBEA",
    "vix": "VIXCLS",
    "dxy": "DTWEXBGS",
    "ten_yr_yield": "DGS10",
    "two_yr_yield": "DGS2",
}

DEFAULTS = {
    "fed_funds_rate": 5.25,
    "cpi_yoy": 3.2,
    "unemployment": 4.1,
    "gdp_growth": 2.1,
    "vix": 18.0,
    "dxy": 104.0,
    "ten_yr_yield": 4.3,
    "two_yr_yield": 4.7,
}


async def _fetch_series(series_id: str, api_key: str) -> float | None:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": "1",
    }
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(FRED_BASE, params=params)
            r.raise_for_status()
            observations = r.json().get("observations", [])
            if observations:
                val = observations[0].get("value", ".")
                if val != ".":
                    return float(val)
    except Exception:
        pass
    return None


async def fetch() -> dict:
    api_key = os.getenv("FRED_API_KEY", "")
    indicators: dict = {}

    if api_key:
        tasks = {name: _fetch_series(sid, api_key) for name, sid in SERIES.items()}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for name, result in zip(tasks.keys(), results):
            indicators[name] = result if isinstance(result, float) else DEFAULTS[name]
    else:
        indicators = dict(DEFAULTS)

    yield_curve = indicators.get("ten_yr_yield", 4.3) - indicators.get("two_yr_yield", 4.7)
    vix = indicators.get("vix", 18.0)

    if vix < 15:
        regime = "low_vol_bull"
    elif vix < 25:
        regime = "normal"
    elif vix < 35:
        regime = "elevated_vol"
    else:
        regime = "crisis"

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "indicators": indicators,
        "derived": {
            "yield_curve_spread": round(yield_curve, 3),
            "yield_curve_inverted": yield_curve < 0,
            "vix_regime": regime,
        },
        "source": "fred" if api_key else "defaults",
    }
