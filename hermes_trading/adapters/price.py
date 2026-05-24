"""
Price data adapter — yfinance for US equities and Forex pairs.
Computes RSI, MACD, ATR, momentum from OHLCV history.
"""
import asyncio
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

SCHEMA_VERSION = "price/v1"

# yfinance Forex ticker format: EURUSD=X
FOREX_SUFFIX = "=X"


class SchemaError(Exception):
    pass


def _to_yf_ticker(asset: str) -> str:
    """Convert our canonical asset format to yfinance ticker."""
    if "/" in asset and not asset.endswith("=X"):
        return asset.replace("/", "") + FOREX_SUFFIX
    return asset


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    if len(closes) < slow + signal:
        return {"macd": 0.0, "signal": 0.0, "histogram": 0.0, "crossover": False}

    def ema(data: np.ndarray, n: int) -> np.ndarray:
        k = 2.0 / (n + 1)
        out = np.zeros(len(data))
        out[0] = data[0]
        for i in range(1, len(data)):
            out[i] = data[i] * k + out[i - 1] * (1 - k)
        return out

    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    crossover = bool(macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2])
    return {
        "macd": float(macd_line[-1]),
        "signal": float(signal_line[-1]),
        "histogram": float(hist[-1]),
        "crossover": crossover,
    }


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    if len(close) < 2:
        return 0.0
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )
    if len(tr) < period:
        return float(np.mean(tr))
    return float(np.mean(tr[-period:]))


def _momentum(closes: np.ndarray, lookback: int = 5) -> float:
    if len(closes) < lookback + 1:
        return 0.0
    return float((closes[-1] - closes[-(lookback + 1)]) / closes[-(lookback + 1)])


def _bollinger(closes: np.ndarray, period: int = 20, std_dev: float = 2.0) -> dict:
    """Bollinger Bands. pct_b=0 → at lower band, 1 → at upper band."""
    if len(closes) < period:
        return {"upper": 0.0, "middle": 0.0, "lower": 0.0, "pct_b": 0.5, "bandwidth": 0.05}
    window = closes[-period:]
    mid = float(np.mean(window))
    std = float(np.std(window, ddof=0))
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    price = float(closes[-1])
    denom = upper - lower
    pct_b = float((price - lower) / denom) if denom > 0 else 0.5
    bandwidth = float((upper - lower) / mid) if mid > 0 else 0.05
    return {
        "upper": round(upper, 6),
        "middle": round(mid, 6),
        "lower": round(lower, 6),
        "pct_b": round(min(1.5, max(-0.5, pct_b)), 4),
        "bandwidth": round(bandwidth, 6),
    }


def _volume_spike(volumes: np.ndarray, period: int = 20, threshold: float = 1.5) -> dict:
    """Detect unusual volume vs recent average (excluding current bar)."""
    if len(volumes) < period + 1:
        return {"spike": False, "ratio": 1.0}
    avg = float(np.mean(volumes[-(period + 1):-1]))
    current = float(volumes[-1])
    ratio = round(current / avg, 2) if avg > 0 else 1.0
    return {"spike": ratio >= threshold, "ratio": ratio}


async def fetch(asset: str, period: str = "60d", interval: str = "1h") -> dict:
    ticker = _to_yf_ticker(asset)

    def _download():
        return yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)

    df: pd.DataFrame = await asyncio.get_event_loop().run_in_executor(None, _download)

    if df is None or df.empty:
        raise SchemaError(f"No price data returned for {asset} ({ticker})")

    # yfinance ≥0.2.x may return MultiIndex columns — flatten to Series
    def _col(name: str):
        if isinstance(df.columns, pd.MultiIndex):
            cols = [c for c in df.columns if c[0] == name]
            if not cols:
                raise SchemaError(f"Column {name} missing for {asset}")
            return df[cols[0]]
        return df[name]

    closes = _col("Close").values.flatten().astype(float)
    highs = _col("High").values.flatten().astype(float)
    lows = _col("Low").values.flatten().astype(float)
    volumes = _col("Volume").values.flatten().astype(float)

    if len(closes) < 30:
        raise SchemaError(f"Insufficient price history for {asset}: {len(closes)} bars")

    rsi_val = _rsi(closes)
    macd_val = _macd(closes)
    atr_val = _atr(highs, lows, closes)
    mom_val = _momentum(closes, 5)
    mom_1d_val = _momentum(closes, 24)   # ~1 trading day of 1h bars
    bb_val = _bollinger(closes)
    vol_spike_val = _volume_spike(volumes)
    vol_percentile = float(np.percentile(volumes, 60))
    current_vol = float(volumes[-1]) if len(volumes) > 0 else 0.0

    price = float(closes[-1])

    return {
        "schema_version": SCHEMA_VERSION,
        "asset": asset,
        "ticker": ticker,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "price": price,
        "open": float(_col("Open").values.flatten()[-1]),
        "high": float(highs[-1]),
        "low": float(lows[-1]),
        "volume": current_vol,
        "volume_above_60th_pct": current_vol >= vol_percentile,
        "rsi": rsi_val,
        "macd": macd_val,
        "atr": atr_val,
        "atr_pct": atr_val / price if price > 0 else 0.0,
        "momentum_5d": mom_val,
        "momentum_1d": mom_1d_val,
        "bollinger": bb_val,
        "volume_spike": vol_spike_val,
        "closes_30": closes[-30:].tolist(),
    }
