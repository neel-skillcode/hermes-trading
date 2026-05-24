"""
Universe adapter — scans Forex majors + US equities + crypto and ranks them.

Returns two structures:
  all_ranked  — every scored asset, sorted by rank_score DESC,
                includes _price_data / _news_data so the loop can skip
                re-fetching them.
  candidates  — top N display-friendly dicts (no internal fields),
                written to heartbeat for the dashboard.

Scoring uses: RSI extremity, MACD crossover, 5d & 1d momentum, Bollinger
Band position, volume spike, and news sentiment/impact.
"""
import asyncio
from datetime import datetime, timezone

from hermes_trading.adapters import price as price_adapter
from hermes_trading.adapters import news as news_adapter

SCHEMA_VERSION = "universe/v2"

# ── Asset universe ────────────────────────────────────────────────────────────

FOREX_PAIRS = [
    # Majors
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD",
    # Crosses
    "EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "EUR/AUD",
    "GBP/AUD", "EUR/CAD", "EUR/CHF", "GBP/CHF", "AUD/NZD", "CAD/JPY",
    # Liquid exotics
    "USD/MXN", "USD/SGD", "USD/HKD", "USD/NOK", "USD/SEK",
]

CRYPTO_ASSETS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
    "XRP-USD", "AVAX-USD", "DOGE-USD", "MATIC-USD",
]

US_EQUITIES = [
    # ── Mega-cap tech ──────────────────────────────────────────────────────────
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "ORCL", "CRM",
    # ── Finance ────────────────────────────────────────────────────────────────
    "JPM", "GS", "MS", "BAC", "C", "WFC", "V", "MA", "AXP", "PYPL",
    # ── Energy ─────────────────────────────────────────────────────────────────
    "XOM", "CVX", "OXY", "SLB", "MPC", "VLO", "HAL",
    # ── Healthcare / Biotech ───────────────────────────────────────────────────
    "LLY", "MRNA", "BIIB", "REGN", "VRTX", "PFE", "ABBV", "UNH", "BMY",
    # ── Semis ──────────────────────────────────────────────────────────────────
    "AMD", "INTC", "QCOM", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "ON",
    # ── Consumer / Retail ──────────────────────────────────────────────────────
    "HD", "COST", "NKE", "MCD", "SBUX", "TGT", "WMT", "AMZN",
    # ── Sector ETFs (macro exposure) ───────────────────────────────────────────
    "SPY", "QQQ", "IWM", "GLD", "TLT", "XLE", "XLF", "XLK", "XLV", "ARKK",
    # ── High-beta momentum ─────────────────────────────────────────────────────
    "PLTR", "COIN", "HOOD", "RBLX", "SNAP", "UBER", "LYFT", "SOFI", "RIVN", "LCID",
    # ── Small/mid cap momentum ─────────────────────────────────────────────────
    "SMCI", "ARM", "IONQ", "SOUN", "RXRX",
    # ── Commodity ETFs ─────────────────────────────────────────────────────────
    "USO", "UNG", "SLV",
]

# Deduplicate while preserving order
US_EQUITIES = list(dict.fromkeys(US_EQUITIES))


# ── Scoring ───────────────────────────────────────────────────────────────────

def _rank_asset(price_data: dict, news_data: dict, strategy: dict) -> float:
    """Composite ranking score [0, 1]. Higher = stronger signal to trade now."""
    score = 0.0

    # RSI extremity — the further from 50, the more actionable
    rsi = price_data.get("rsi", 50.0)
    score += (abs(rsi - 50.0) / 50.0) * 0.15

    # MACD crossover = strong momentum shift
    macd = price_data.get("macd", {})
    if macd.get("crossover"):
        score += 0.20
    elif abs(macd.get("histogram", 0.0)) > 0:
        score += 0.06

    # 5-day momentum
    score += min(0.18, abs(price_data.get("momentum_5d", 0.0)) * 2.5)

    # 1-day momentum (day-trade signal strength)
    score += min(0.15, abs(price_data.get("momentum_1d", 0.0)) * 4.0)

    # Bollinger Band extremity: 0 = lower band, 1 = upper band
    pct_b = price_data.get("bollinger", {}).get("pct_b", 0.5)
    score += abs(pct_b - 0.5) * 2.0 * 0.14   # max 0.14

    # Volume spike — confirms momentum with capital flows
    vol = price_data.get("volume_spike", {})
    if vol.get("spike"):
        score += min(0.12, (vol.get("ratio", 1.0) - 1.0) * 0.10)

    # Volume above 60th percentile
    if price_data.get("volume_above_60th_pct"):
        score += 0.04

    # News sentiment (magnitude × confidence)
    sentiment = news_data.get("sentiment", {})
    news_abs = abs(sentiment.get("score", 0.0)) * sentiment.get("confidence", 0.0)
    score += news_abs * 0.10

    # High-impact catalyst (Fed, earnings, M&A, etc.)
    if sentiment.get("high_impact"):
        score += 0.08

    return min(1.0, score)


# ── Per-asset scoring ─────────────────────────────────────────────────────────

async def _score_single(asset: str, strategy: dict) -> dict | None:
    try:
        price_data, news_data = await asyncio.gather(
            price_adapter.fetch(asset),
            news_adapter.fetch(asset),
        )
        rank = _rank_asset(price_data, news_data, strategy)
        return {
            # ── Display fields (written to heartbeat) ──────────────────────────
            "asset": asset,
            "rank_score": round(rank, 4),
            "price": price_data.get("price"),
            "rsi": price_data.get("rsi"),
            "momentum_5d": price_data.get("momentum_5d"),
            "momentum_1d": price_data.get("momentum_1d"),
            "macd_crossover": price_data.get("macd", {}).get("crossover"),
            "atr_pct": price_data.get("atr_pct"),
            "bollinger_pct_b": price_data.get("bollinger", {}).get("pct_b"),
            "volume_spike": price_data.get("volume_spike", {}).get("spike"),
            "volume_spike_ratio": price_data.get("volume_spike", {}).get("ratio"),
            "news_sentiment": news_data.get("sentiment", {}).get("score"),
            "news_confidence": news_data.get("sentiment", {}).get("confidence"),
            "news_headlines": news_data.get("headlines", [])[:3],
            "high_impact_news": news_data.get("sentiment", {}).get("high_impact"),
            # ── Internal — used by loop._tick() to avoid double-fetching ───────
            "_price_data": price_data,
            "_news_data": news_data,
        }
    except Exception:
        return None


# ── Main fetch ────────────────────────────────────────────────────────────────

async def fetch(strategy: dict) -> dict:
    sel = strategy.get("universe_selection", {})
    display_n = sel.get("display_candidates", 20)

    # Score all assets concurrently; semaphore avoids rate-limiting
    sem = asyncio.Semaphore(10)

    async def guarded(asset: str):
        async with sem:
            return await _score_single(asset, strategy)

    all_assets = FOREX_PAIRS + CRYPTO_ASSETS + US_EQUITIES
    results = await asyncio.gather(*[guarded(a) for a in all_assets])
    scored = [r for r in results if r is not None]

    # Sort all by rank — loop iterates ALL of them, high-score first
    all_ranked = sorted(scored, key=lambda x: x["rank_score"], reverse=True)

    # Display-friendly top-N for heartbeat / dashboard (strip internal fields)
    candidates = [
        {k: v for k, v in c.items() if not k.startswith("_")}
        for c in all_ranked[:display_n]
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_scanned": len(scored),
        "all_ranked": all_ranked,   # includes _price_data, _news_data for loop
        "candidates": candidates,   # clean, for heartbeat
    }
