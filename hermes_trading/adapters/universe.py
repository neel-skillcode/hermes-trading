"""
Universe adapter — scans Forex majors + US equities and ranks them.
Returns top N candidates scored on momentum, volatility, RSI position,
news sentiment, and volume trend.
AI selection mode (--hermes) passes candidates to Hermes for final ranking.
"""
import asyncio
from datetime import datetime, timezone

from hermes_trading.adapters import price as price_adapter
from hermes_trading.adapters import news as news_adapter

SCHEMA_VERSION = "universe/v1"

FOREX_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
    "USD/CAD", "USD/CHF", "NZD/USD", "EUR/GBP",
    "EUR/JPY", "GBP/JPY", "AUD/JPY", "EUR/AUD",
    "GBP/AUD", "EUR/CAD", "USD/MXN", "USD/SGD",
]

US_EQUITIES = [
    # Mega-cap tech (high liquidity, high vol)
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO",
    # Finance
    "JPM", "GS", "MS", "BAC", "BRK-B",
    # Energy
    "XOM", "CVX", "OXY",
    # Healthcare / Biotech (high vol)
    "LLY", "MRNA", "BIIB", "REGN", "VRTX",
    # Semis
    "AMD", "INTC", "QCOM", "MU", "AMAT", "LRCX",
    # Consumer
    "AMZN", "HD", "COST", "NKE",
    # ETFs for macro exposure
    "SPY", "QQQ", "IWM", "GLD", "TLT",
    # High-beta momentum names
    "PLTR", "COIN", "HOOD", "RBLX", "SNAP", "UBER", "LYFT",
    # Small/mid cap momentum
    "SMCI", "ARM", "AEHR", "IONQ",
]

# Deduplicate
US_EQUITIES = list(dict.fromkeys(US_EQUITIES))


def _rank_asset(price_data: dict, news_data: dict, strategy: dict) -> float:
    """Composite ranking score [0, 1] — higher is better candidate to trade."""
    score = 0.0

    # RSI: prefer oversold (long bias) or overbought (short bias)
    rsi = price_data.get("rsi", 50.0)
    rsi_score = max(0.0, (50.0 - abs(rsi - 50.0)) / 50.0)
    score += rsi_score * 0.20

    # MACD crossover is a strong signal
    macd = price_data.get("macd", {})
    if macd.get("crossover"):
        score += 0.20
    elif abs(macd.get("histogram", 0.0)) > 0:
        score += 0.08

    # Momentum: absolute value (we can go long or short)
    momentum = abs(price_data.get("momentum_5d", 0.0))
    score += min(0.25, momentum * 2.5)

    # Volume above 60th percentile
    if price_data.get("volume_above_60th_pct"):
        score += 0.10

    # News sentiment
    sentiment = news_data.get("sentiment", {})
    news_weight = strategy.get("entry", {}).get("news_sentiment_weight", 0.35)
    news_confidence = sentiment.get("confidence", 0.0)
    news_score = abs(sentiment.get("score", 0.0)) * news_confidence
    score += news_score * news_weight * 0.25

    # High impact news is a bonus
    if sentiment.get("high_impact"):
        score += 0.10

    return min(1.0, score)


async def _score_single(asset: str, strategy: dict) -> dict | None:
    try:
        price_data, news_data = await asyncio.gather(
            price_adapter.fetch(asset),
            news_adapter.fetch(asset),
        )
        rank = _rank_asset(price_data, news_data, strategy)
        return {
            "asset": asset,
            "rank_score": round(rank, 4),
            "price": price_data.get("price"),
            "rsi": price_data.get("rsi"),
            "momentum_5d": price_data.get("momentum_5d"),
            "macd_crossover": price_data.get("macd", {}).get("crossover"),
            "atr_pct": price_data.get("atr_pct"),
            "news_sentiment": news_data.get("sentiment", {}).get("score"),
            "news_confidence": news_data.get("sentiment", {}).get("confidence"),
            "news_headlines": news_data.get("headlines", [])[:3],
            "high_impact_news": news_data.get("sentiment", {}).get("high_impact"),
        }
    except Exception as e:
        return None


async def fetch(strategy: dict, top_n: int = 5) -> dict:
    sel = strategy.get("universe_selection", {})
    forex_weight = sel.get("forex_weight", 0.4)
    equity_weight = sel.get("equity_weight", 0.6)
    max_candidates = sel.get("max_candidates", top_n)

    # Score all assets concurrently (with semaphore to avoid rate limits)
    sem = asyncio.Semaphore(8)

    async def guarded(asset):
        async with sem:
            return await _score_single(asset, strategy)

    all_assets = FOREX_PAIRS + US_EQUITIES
    results = await asyncio.gather(*[guarded(a) for a in all_assets])
    scored = [r for r in results if r is not None]

    forex_scored = sorted(
        [r for r in scored if r["asset"] in FOREX_PAIRS],
        key=lambda x: x["rank_score"],
        reverse=True,
    )
    equity_scored = sorted(
        [r for r in scored if r["asset"] in US_EQUITIES],
        key=lambda x: x["rank_score"],
        reverse=True,
    )

    n_forex = max(1, round(max_candidates * forex_weight))
    n_equity = max(1, round(max_candidates * equity_weight))
    candidates = forex_scored[:n_forex] + equity_scored[:n_equity]
    candidates.sort(key=lambda x: x["rank_score"], reverse=True)

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_scanned": len(scored),
        "candidates": candidates[:max_candidates],
    }
