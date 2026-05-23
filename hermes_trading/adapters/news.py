"""
News adapter — fetches asset-specific and macro news.
Primary: yfinance news feed (no API key).
Optional: NewsAPI (set NEWS_API_KEY in .env for more coverage).
Sentiment scoring uses keyword weighting — no external NLP deps.
"""
import asyncio
import os
import re
from datetime import datetime, timezone
from typing import Any

import httpx
import yfinance as yf

SCHEMA_VERSION = "news/v1"

POSITIVE_WORDS = {
    "surge", "rally", "gain", "jump", "rise", "beat", "record", "strong",
    "growth", "profit", "upgrade", "buy", "bullish", "outperform", "boost",
    "expansion", "recovery", "positive", "optimistic", "breakthrough", "deal",
    "approval", "launch", "partnership", "revenue", "earnings", "beat",
}

NEGATIVE_WORDS = {
    "crash", "fall", "drop", "decline", "miss", "weak", "loss", "sell",
    "bearish", "underperform", "cut", "risk", "concern", "uncertainty",
    "lawsuit", "fraud", "recall", "tariff", "sanction", "inflation", "recession",
    "default", "downgrade", "warning", "layoff", "investigation",
}

HIGH_IMPACT_WORDS = {
    "fed", "federal reserve", "interest rate", "cpi", "inflation", "gdp",
    "recession", "earnings", "merger", "acquisition", "ipo", "bankruptcy",
    "sec", "fda", "war", "crisis", "sanctions",
}


def _score_headline(text: str) -> tuple[float, float]:
    """Returns (sentiment [-1, 1], confidence [0, 1])."""
    lower = text.lower()
    words = re.findall(r"\w+", lower)
    word_set = set(words)

    pos = len(word_set & POSITIVE_WORDS)
    neg = len(word_set & NEGATIVE_WORDS)
    high_impact = any(term in lower for term in HIGH_IMPACT_WORDS)

    total = pos + neg
    if total == 0:
        return 0.0, 0.2

    sentiment = (pos - neg) / total
    confidence = min(0.9, 0.4 + total * 0.1 + (0.2 if high_impact else 0.0))
    return float(sentiment), float(confidence)


def _aggregate_sentiment(articles: list[dict]) -> dict:
    if not articles:
        return {"score": 0.0, "confidence": 0.1, "article_count": 0, "high_impact": False}

    scores, confs = [], []
    high_impact = False

    for a in articles:
        title = a.get("title", "") or ""
        summary = a.get("summary", "") or a.get("description", "") or ""
        text = f"{title} {summary}"
        s, c = _score_headline(text)
        scores.append(s)
        confs.append(c)
        if any(term in text.lower() for term in HIGH_IMPACT_WORDS):
            high_impact = True

    weighted = sum(s * c for s, c in zip(scores, confs))
    total_conf = sum(confs)
    avg_sentiment = weighted / total_conf if total_conf > 0 else 0.0
    avg_conf = min(0.95, total_conf / len(confs))

    return {
        "score": round(float(avg_sentiment), 3),
        "confidence": round(float(avg_conf), 3),
        "article_count": len(articles),
        "high_impact": high_impact,
    }


def _yf_ticker(asset: str) -> str:
    if "/" in asset and not asset.endswith("=X"):
        return asset.replace("/", "") + "=X"
    return asset


async def _fetch_yfinance_news(asset: str) -> list[dict]:
    ticker_str = _yf_ticker(asset)

    def _get():
        t = yf.Ticker(ticker_str)
        return t.news or []

    try:
        raw = await asyncio.get_event_loop().run_in_executor(None, _get)
        articles = []
        for item in raw[:15]:
            articles.append({
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "publisher": item.get("publisher", ""),
                "timestamp": item.get("providerPublishTime", 0),
                "source": "yfinance",
            })
        return articles
    except Exception:
        return []


async def _fetch_newsapi(asset: str) -> list[dict]:
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key:
        return []

    query = asset.replace("/", " ").replace("=X", "").strip()
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": api_key,
        "language": "en",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            articles = []
            for item in data.get("articles", []):
                articles.append({
                    "title": item.get("title", ""),
                    "summary": item.get("description", ""),
                    "publisher": item.get("source", {}).get("name", ""),
                    "timestamp": item.get("publishedAt", ""),
                    "source": "newsapi",
                })
            return articles
    except Exception:
        return []


async def fetch(asset: str) -> dict:
    yf_articles, api_articles = await asyncio.gather(
        _fetch_yfinance_news(asset),
        _fetch_newsapi(asset),
    )

    all_articles = yf_articles + api_articles
    sentiment = _aggregate_sentiment(all_articles)

    headlines = [a["title"] for a in all_articles[:10] if a.get("title")]

    return {
        "schema_version": SCHEMA_VERSION,
        "asset": asset,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sentiment": sentiment,
        "headlines": headlines,
        "article_count": len(all_articles),
    }
