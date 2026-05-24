"""
Core trading loop — runs 24/7 (or as single-tick via GitHub Actions).

Every tick:
  1. Update open positions: check stop/target/trailing-stop hit
  2. Check portfolio drawdown against goal
  3. Scan full universe (Forex + Equities + Crypto, 90+ assets)
  4. Evaluate ALL ranked candidates — use cached price/news data from scan
  5. Enter day trades and swing trades up to capacity limits
  6. Check reflection schedule (Mon/Wed/Fri 09:30 ET)
  7. Write heartbeat + save positions
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytz
import yaml
from rich.console import Console
from rich.table import Table

from hermes_trading.adapters import macro as macro_adapter
from hermes_trading.adapters import news as news_adapter
from hermes_trading.adapters import price as price_adapter
from hermes_trading.adapters import universe as universe_adapter
from hermes_trading.adapters import gist_state

STATE_DIR = Path(__file__).parent.parent / "state"
TRADES_FILE = STATE_DIR / "trades.jsonl"
HEARTBEAT_FILE = STATE_DIR / "heartbeat.json"
POSITIONS_FILE = STATE_DIR / "positions.json"
STRATEGY_FILE = STATE_DIR / "strategy.yaml"
GOAL_FILE = STATE_DIR / "goal.yaml"

ET = pytz.timezone("America/New_York")
REFLECTION_DAYS = {"MON": 0, "WED": 2, "FRI": 4}

console = Console()


# ── Market hours helpers ──────────────────────────────────────────────────────

def _is_equity(asset: str) -> bool:
    """True if the asset is a US equity/ETF (not forex or crypto)."""
    return "/" not in asset and not asset.endswith("-USD")

def _us_market_open() -> bool:
    """True during NYSE regular hours: Mon–Fri 09:30–16:00 ET."""
    now = datetime.now(ET)
    if now.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    mins = now.hour * 60 + now.minute
    return 570 <= mins < 960        # 9:30 = 570, 16:00 = 960

def _approaching_market_close() -> bool:
    """True Mon–Fri 15:45–16:00 ET — day trades should be closed before bell."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    return 945 <= mins < 960        # 15:45–16:00

def _forex_or_crypto(asset: str) -> bool:
    """True for assets that trade 24/7 (never need market-hours gate)."""
    return "/" in asset or asset.endswith("-USD")


# ── Circuit breaker ───────────────────────────────────────────────────────────

class CircuitBreaker:
    def __init__(self, threshold: int = 5):
        self.threshold = threshold
        self.failures = 0
        self.open = False

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.threshold:
            self.open = True
            console.print("[bold red]Circuit breaker OPEN — too many consecutive failures[/bold red]")

    def record_success(self):
        self.failures = 0
        if self.open:
            self.open = False
            console.print("[bold green]Circuit breaker reset[/bold green]")


async def _with_retry(coro_fn, retries: int = 3, breaker: CircuitBreaker | None = None):
    for attempt in range(retries):
        try:
            result = await coro_fn()
            if breaker:
                breaker.record_success()
            return result
        except Exception as e:
            if attempt == retries - 1:
                if breaker:
                    breaker.record_failure()
                raise
            await asyncio.sleep(2 ** attempt)


# ── Trading loop ──────────────────────────────────────────────────────────────

class TradingLoop:
    def __init__(self, goal: dict, dry_run: bool = False):
        self.goal = goal
        self.dry_run = dry_run
        self.strategy: dict = {}
        self.open_positions: dict[str, dict] = {}
        self.portfolio_balance = goal.get("starting_balance", 100_000.0)
        self.portfolio_peak = self.portfolio_balance
        self.total_trades = 0
        self.trades_since_reflection = 0
        self.last_reflection_date: str | None = None
        self.last_candidates: list = []
        self.last_scan_total: int = 0
        self.circuit_breaker = CircuitBreaker()
        self._load_state()

    # ── State I/O ─────────────────────────────────────────────────────────────

    def _load_state(self):
        with open(STRATEGY_FILE) as f:
            self.strategy = yaml.safe_load(f)
        hb = self._read_heartbeat()
        self.portfolio_balance = hb.get("portfolio_balance", self.portfolio_balance)
        self.portfolio_peak = hb.get("portfolio_peak", self.portfolio_peak)
        self.total_trades = hb.get("total_trades", 0)
        self.trades_since_reflection = hb.get("trades_since_last_reflection", 0)

        # Restore open positions from disk
        if POSITIONS_FILE.exists():
            try:
                raw = json.loads(POSITIONS_FILE.read_text())
                # Migrate old positions that lack trade_type
                for key, pos in raw.items():
                    if "trade_type" not in pos:
                        pos["trade_type"] = "swing"
                self.open_positions = raw
                if self.open_positions:
                    n_day = sum(1 for p in self.open_positions.values() if p.get("trade_type") == "day")
                    n_sw  = sum(1 for p in self.open_positions.values() if p.get("trade_type") == "swing")
                    console.print(
                        f"[dim]Restored {len(self.open_positions)} position(s) "
                        f"({n_day} day / {n_sw} swing)[/dim]"
                    )
            except Exception:
                self.open_positions = {}

    def _save_positions(self):
        """Write open positions to disk so they survive between ticks."""
        POSITIONS_FILE.write_text(json.dumps(self.open_positions, indent=2))

    def _read_heartbeat(self) -> dict:
        if HEARTBEAT_FILE.exists():
            return json.loads(HEARTBEAT_FILE.read_text())
        return {}

    def _write_heartbeat(self, extra: dict | None = None):
        dd = (self.portfolio_peak - self.portfolio_balance) / self.portfolio_peak if self.portfolio_peak > 0 else 0.0
        n_day  = sum(1 for p in self.open_positions.values() if p.get("trade_type") == "day")
        n_swing = sum(1 for p in self.open_positions.values() if p.get("trade_type") == "swing")
        day_usd = sum(p.get("size_usd", 0) for p in self.open_positions.values() if p.get("trade_type") == "day")
        sw_usd  = sum(p.get("size_usd", 0) for p in self.open_positions.values() if p.get("trade_type") == "swing")

        # Build a current-price lookup from the latest scan so we can show unrealized P&L
        price_lookup: dict[str, float] = {c["asset"]: c["price"] for c in self.last_candidates if c.get("price")}

        total_unrealized_usd = 0.0
        pos_list = []
        for pos in self.open_positions.values():
            asset = pos["asset"]
            entry = pos["entry_price"]
            current = price_lookup.get(asset, entry)
            direction = pos.get("direction", "long")
            pct = (current - entry) / entry if direction == "long" else (entry - current) / entry
            unreal_usd = pos.get("size_usd", 0) * pct
            total_unrealized_usd += unreal_usd
            pos_list.append({
                **pos,
                "current_price": round(current, 6),
                "unrealized_pnl_pct": round(pct, 6),
                "unrealized_pnl_usd": round(unreal_usd, 2),
            })

        data = {
            "status": "running",
            "portfolio_balance": round(self.portfolio_balance, 2),
            "portfolio_peak": round(self.portfolio_peak, 2),
            "portfolio_drawdown_pct": round(dd * 100, 2),
            "unrealized_pnl_usd": round(total_unrealized_usd, 2),
            "equity_value": round(self.portfolio_balance + total_unrealized_usd, 2),
            "open_positions": len(self.open_positions),
            "day_trades_open": n_day,
            "swing_trades_open": n_swing,
            "day_capital_used_usd": round(day_usd, 2),
            "swing_capital_used_usd": round(sw_usd, 2),
            "total_trades": self.total_trades,
            "last_tick": datetime.now(timezone.utc).isoformat(),
            "last_reflection": self.last_reflection_date,
            "trades_since_last_reflection": self.trades_since_reflection,
            "universe_scanned": self.last_scan_total,
            "market_open": _us_market_open(),
            "open_position_list": pos_list,
            "last_candidates": self.last_candidates,
        }
        if extra:
            data.update(extra)
        HEARTBEAT_FILE.write_text(json.dumps(data, indent=2))

    def _reload_strategy(self):
        with open(STRATEGY_FILE) as f:
            self.strategy = yaml.safe_load(f)

    # ── Portfolio math ────────────────────────────────────────────────────────

    def _drawdown_pct(self) -> float:
        if self.portfolio_peak <= 0:
            return 0.0
        return (self.portfolio_peak - self.portfolio_balance) / self.portfolio_peak

    def _drawdown_ok(self) -> bool:
        return self._drawdown_pct() < self.goal.get("max_drawdown", {}).get("hard_ceiling", 0.40)

    # ── Trade type classification ─────────────────────────────────────────────

    def _determine_trade_type(self, price_data: dict, signal: dict) -> str:
        """Classify a signal as a 'day' trade (intraday) or 'swing' (multi-day)."""
        mom_1d = abs(price_data.get("momentum_1d", 0.0))
        vol_spike = price_data.get("volume_spike", {}).get("spike", False)
        macd_crossover = price_data.get("macd", {}).get("crossover", False)
        pct_b = price_data.get("bollinger", {}).get("pct_b", 0.5)

        # Strong intraday surge + volume → ride the wave today
        if mom_1d > 0.008 or (mom_1d > 0.004 and vol_spike):
            return "day"
        # BB extreme with volume burst → mean-reversion scalp
        if (pct_b < 0.10 or pct_b > 0.90) and vol_spike:
            return "day"
        # MACD crossover = multi-day setup = swing
        if macd_crossover:
            return "swing"
        # Moderate intraday with any volume → day
        if mom_1d > 0.004:
            return "day"
        return "swing"

    # ── Stop loss & position sizing ───────────────────────────────────────────

    def _calculate_stop_loss(self, atr_pct: float, confidence: float,
                              trade_type: str, strategy: dict) -> float:
        sl_cfg = strategy.get("stop_loss", {}).get(f"{trade_type}_trade", {})
        atr_mult = sl_cfg.get("atr_multiplier", 2.0)
        min_sl = sl_cfg.get("min_pct", 0.003)
        max_sl = sl_cfg.get("max_pct", 0.05)
        # Higher confidence → slightly wider stop (more conviction = more room)
        conf_scale = 0.7 + confidence * 0.6
        sl = atr_pct * atr_mult * conf_scale
        return max(min_sl, min(max_sl, sl))

    def _calculate_position_size(self, confidence: float,
                                  trade_type: str, strategy: dict) -> float:
        """Returns size in USD, respecting the capital pool for this trade type."""
        ps_cfg = strategy.get("position_sizing", {})
        type_cfg = ps_cfg.get(f"{trade_type}_trade", {})
        default_max = 0.04 if trade_type == "day" else 0.08
        max_pos_pct = type_cfg.get("max_position_pct", default_max)
        kelly_f = ps_cfg.get("kelly_fraction", 0.40)
        size_pct = min(max_pos_pct, max(0.004, max_pos_pct * kelly_f * confidence))

        # Capital pool constraint
        sel = strategy.get("universe_selection", {})
        cap_key = "day_trade_capital_pct" if trade_type == "day" else "swing_capital_pct"
        pool_pct = sel.get(cap_key, 0.40)
        pool_capital = self.portfolio_balance * pool_pct
        pool_used = sum(
            p.get("size_usd", 0) for p in self.open_positions.values()
            if p.get("trade_type") == trade_type
        )
        available = max(0.0, pool_capital - pool_used)

        size_usd = min(self.portfolio_balance * size_pct, available)
        return round(max(0.0, size_usd), 2)

    # ── Signal evaluation ─────────────────────────────────────────────────────

    def _evaluate_signals(self, price_data: dict, news_data: dict, strategy: dict) -> dict:
        """Returns signal evaluation: go/no-go, direction, confidence, reasons."""
        signals_hit = 0
        direction = "long"
        confidence = 0.0
        reasons: list[str] = []

        entry_cfg = strategy.get("entry", {})
        ind_cfg = entry_cfg.get("indicators", {})

        # ── RSI ───────────────────────────────────────────────────────────────
        rsi = price_data.get("rsi", 50.0)
        rsi_cfg = ind_cfg.get("rsi", {})
        if rsi <= rsi_cfg.get("oversold", 40):
            signals_hit += 1
            direction = "long"
            confidence += 0.22
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi >= rsi_cfg.get("overbought", 60):
            signals_hit += 1
            direction = "short"
            confidence += 0.22
            reasons.append(f"RSI overbought ({rsi:.1f})")

        # ── MACD crossover ────────────────────────────────────────────────────
        macd = price_data.get("macd", {})
        if macd.get("crossover"):
            signals_hit += 1
            confidence += 0.25
            reasons.append("MACD crossover")

        # ── 5-day momentum ────────────────────────────────────────────────────
        momentum = price_data.get("momentum_5d", 0.0)
        mom_threshold = ind_cfg.get("momentum", {}).get("min_pct", 0.003)
        if abs(momentum) > mom_threshold:
            signals_hit += 1
            confidence += 0.14
            direction = "long" if momentum > 0 else "short"
            reasons.append(f"5d momentum {momentum:+.2%}")

        # ── 1-day intraday momentum ───────────────────────────────────────────
        mom_1d = price_data.get("momentum_1d", 0.0)
        mom_1d_threshold = ind_cfg.get("momentum_1d", {}).get("min_pct", 0.004)
        if abs(mom_1d) > mom_1d_threshold:
            signals_hit += 1
            confidence += 0.20
            direction = "long" if mom_1d > 0 else "short"
            reasons.append(f"1d momentum {mom_1d:+.2%}")

        # ── Bollinger Bands ───────────────────────────────────────────────────
        bb = price_data.get("bollinger", {})
        pct_b = bb.get("pct_b", 0.5)
        bb_cfg = ind_cfg.get("bollinger", {})
        if pct_b <= bb_cfg.get("entry_pct_b_low", 0.15):
            signals_hit += 1
            direction = "long"
            confidence += 0.20
            reasons.append(f"BB oversold (pct_b={pct_b:.2f})")
        elif pct_b >= bb_cfg.get("entry_pct_b_high", 0.85):
            signals_hit += 1
            direction = "short"
            confidence += 0.20
            reasons.append(f"BB overbought (pct_b={pct_b:.2f})")

        # ── Volume spike ──────────────────────────────────────────────────────
        vol_spike = price_data.get("volume_spike", {})
        if vol_spike.get("spike"):
            signals_hit += 1
            confidence += 0.15
            ratio = vol_spike.get("ratio", 1.0)
            reasons.append(f"Volume spike {ratio:.1f}×")

        # ── News sentiment ────────────────────────────────────────────────────
        sentiment = news_data.get("sentiment", {})
        news_score = sentiment.get("score", 0.0)
        news_conf = sentiment.get("confidence", 0.0)
        news_weight = entry_cfg.get("news_sentiment_weight", 0.25)
        min_news_conf = self.goal.get("news_analysis", {}).get("sentiment_min_confidence", 0.6)

        if abs(news_score) > 0.2 and news_conf >= min_news_conf:
            signals_hit += 1
            news_boost = news_weight * abs(news_score) * news_conf
            confidence += news_boost
            news_dir = "long" if news_score > 0 else "short"
            reasons.append(f"News {news_score:+.2f} (conf {news_conf:.2f}) → {news_dir}")
            if news_dir != direction:
                confidence *= 0.65   # penalise conflicting signals

        if sentiment.get("high_impact"):
            confidence = min(0.95, confidence * 1.15)
            reasons.append("High-impact catalyst")

        # ── Gate check ────────────────────────────────────────────────────────
        min_confluence = entry_cfg.get("min_signal_confluence", 1)
        min_conf = entry_cfg.get("min_confidence", 0.15)
        go = signals_hit >= min_confluence and confidence >= min_conf

        # Drawdown override
        dd = self._drawdown_pct()
        soft_ceil = self.goal.get("max_drawdown", {}).get("soft_ceiling", 0.225)
        override_min = self.goal.get("max_drawdown", {}).get("override_min_confidence", 0.85)
        if dd > soft_ceil:
            if confidence >= override_min:
                reasons.append(f"DD override: dd={dd:.1%} conf={confidence:.2f}≥{override_min}")
            else:
                go = False
                reasons.append(f"Blocked: dd={dd:.1%} conf={confidence:.2f}<{override_min}")

        return {
            "go": go,
            "direction": direction,
            "confidence": round(min(0.99, confidence), 3),
            "signals_hit": signals_hit,
            "reasons": reasons,
        }

    # ── Trailing stop ─────────────────────────────────────────────────────────

    def _update_trailing_stop(self, pos: dict, current_price: float) -> dict:
        """Ratchet the stop price in the direction of profit."""
        if not self.strategy.get("stop_loss", {}).get("trailing", True):
            return pos
        direction = pos["direction"]
        stop_pct = pos["stop_loss_pct"]
        pos = dict(pos)  # shallow copy — don't mutate in place

        if direction == "long":
            peak = pos.get("price_peak", pos["entry_price"])
            if current_price > peak:
                pos["price_peak"] = current_price
                pos["trailing_stop_price"] = round(current_price * (1 - stop_pct), 6)
        else:
            trough = pos.get("price_trough", pos["entry_price"])
            if current_price < trough:
                pos["price_trough"] = current_price
                pos["trailing_stop_price"] = round(current_price * (1 + stop_pct), 6)
        return pos

    # ── Trade log ─────────────────────────────────────────────────────────────

    def _log_trade(self, trade: dict):
        with open(TRADES_FILE, "a") as f:
            f.write(json.dumps(trade) + "\n")
        self.total_trades += 1
        self.trades_since_reflection += 1

    # ── Position management ───────────────────────────────────────────────────

    async def _close_day_trades_eod(self):
        """Force-close all open day trades at end of market day (called 15:45–16:00 ET)."""
        day_positions = [(a, p) for a, p in list(self.open_positions.items())
                         if p.get("trade_type") == "day" and _is_equity(a)]
        if not day_positions:
            return
        console.print(f"[bold yellow]EOD sweep: force-closing {len(day_positions)} day trade(s) before bell[/bold yellow]")
        for asset, pos in day_positions:
            try:
                current_price = await price_adapter.fetch_live(asset)
                if not current_price:
                    current_price = pos["entry_price"]
                entry = pos["entry_price"]
                direction = pos["direction"]
                change_pct = (
                    (current_price - entry) / entry
                    if direction == "long"
                    else (entry - current_price) / entry
                )
                pnl_usd = pos["size_usd"] * change_pct
                self.portfolio_balance += pnl_usd
                if self.portfolio_balance > self.portfolio_peak:
                    self.portfolio_peak = self.portfolio_balance
                trade_record = {
                    **pos,
                    "exit_price": current_price,
                    "exit_time": datetime.now(timezone.utc).isoformat(),
                    "pnl_pct": round(change_pct, 6),
                    "pnl_usd": round(pnl_usd, 2),
                    "exit_reason": "eod_close",
                    "status": "closed",
                }
                self.open_positions.pop(asset)
                self._log_trade(trade_record)
                colour = "green" if pnl_usd >= 0 else "red"
                console.print(f"[{colour}]EOD CLOSE {asset} pnl={change_pct:+.2%} (${pnl_usd:+.0f})[/{colour}]")
            except Exception as e:
                console.print(f"[yellow]EOD close failed for {asset}: {e}[/yellow]")
        self._save_positions()

    async def _update_positions(self):
        """Fetch live prices, apply trailing stop, close on SL/TP."""
        to_close: list[tuple] = []
        updated: list[tuple] = []

        for asset, pos in list(self.open_positions.items()):
            try:
                # Use fast_info live price — doesn't rely on stale 1h bar closes
                current_price = await price_adapter.fetch_live(asset)
                if not current_price:
                    console.print(f"[dim yellow]{asset}: fetch_live returned None — skipping[/dim yellow]")
                    current_price = pos["entry_price"]
                entry    = pos["entry_price"]
                direction = pos["direction"]

                # Update trailing stop (no side-effects — returns new copy)
                pos = self._update_trailing_stop(pos, current_price)
                updated.append((asset, pos))

                change_pct = (
                    (current_price - entry) / entry
                    if direction == "long"
                    else (entry - current_price) / entry
                )
                stop_pct = pos["stop_loss_pct"]
                tp_pct = pos["take_profit_pct"]

                # Check trailing stop first
                trailing = pos.get("trailing_stop_price")
                if trailing:
                    if direction == "long" and current_price <= trailing:
                        to_close.append((asset, current_price, "trailing_stop", change_pct))
                        continue
                    elif direction == "short" and current_price >= trailing:
                        to_close.append((asset, current_price, "trailing_stop", change_pct))
                        continue

                console.print(
                    f"[dim]{asset} [{pos.get('trade_type','?')}] {direction} "
                    f"entry={entry:.5g} live={current_price:.5g} "
                    f"Δ={change_pct:+.2%} SL={-stop_pct:.2%} TP=+{tp_pct:.2%}[/dim]"
                )

                if change_pct <= -stop_pct:
                    to_close.append((asset, current_price, "stop_loss", change_pct))
                elif change_pct >= tp_pct:
                    to_close.append((asset, current_price, "take_profit", change_pct))

            except Exception:
                pass

        # Apply trailing stop updates for positions NOT being closed
        close_assets = {a for a, *_ in to_close}
        for asset, pos in updated:
            if asset not in close_assets:
                self.open_positions[asset] = pos

        # Close positions
        for asset, exit_price, reason, pnl_pct in to_close:
            pos = self.open_positions.pop(asset)
            size_usd = pos["size_usd"]
            pnl_usd = size_usd * pnl_pct
            self.portfolio_balance += pnl_usd
            if self.portfolio_balance > self.portfolio_peak:
                self.portfolio_peak = self.portfolio_balance

            trade_record = {
                **pos,
                "exit_price": exit_price,
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "pnl_pct": round(pnl_pct, 6),
                "pnl_usd": round(pnl_usd, 2),
                "exit_reason": reason,
                "status": "closed",
            }
            self._log_trade(trade_record)
            colour = "green" if pnl_pct > 0 else "red"
            console.print(
                f"[{colour}]CLOSED [{pos.get('trade_type','?')}] {asset} {reason} "
                f"pnl={pnl_pct:+.2%} (${pnl_usd:+.0f})[/{colour}]"
            )

        if to_close:
            self._save_positions()

    # ── Trade entry ───────────────────────────────────────────────────────────

    async def _enter_trade(
        self,
        candidate: dict,
        price_data: dict,
        news_data: dict,
        signal: dict,
        trade_type: str,
    ):
        asset = candidate["asset"]
        if asset in self.open_positions:
            return   # already have a position in this asset

        strategy = self.strategy
        price = price_data.get("price", 0.0)
        atr_pct = price_data.get("atr_pct", 0.005)
        confidence = signal["confidence"]

        stop_loss_pct = self._calculate_stop_loss(atr_pct, confidence, trade_type, strategy)
        tp_cfg = strategy.get("take_profit", {}).get(f"{trade_type}_trade", {})
        rr_ratio = tp_cfg.get("rr_ratio", 2.5)
        news_boost = min(
            strategy.get("take_profit", {}).get("news_boost_max", 0.5),
            confidence * 0.5,
        )
        take_profit_pct = stop_loss_pct * (rr_ratio + news_boost)

        size_usd = self._calculate_position_size(confidence, trade_type, strategy)
        if size_usd < 10:
            return

        position_size_pct = round(size_usd / self.portfolio_balance, 4)
        mode = os.getenv("HERMES_TRADING_MODE", "paper")

        trade = {
            "id": f"{asset}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "asset": asset,
            "trade_type": trade_type,
            "direction": signal["direction"],
            "entry_price": price,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "stop_loss_pct": round(stop_loss_pct, 5),
            "take_profit_pct": round(take_profit_pct, 5),
            "size_usd": round(size_usd, 2),
            "position_size_pct": position_size_pct,
            "confidence": confidence,
            "signals": signal["reasons"],
            "news_sentiment": news_data.get("sentiment", {}).get("score"),
            "news_headlines": news_data.get("headlines", [])[:3],
            "mode": mode,
            "strategy_version": strategy.get("version", "00"),
            "status": "open",
        }

        self.open_positions[asset] = trade
        self._save_positions()

        console.print(
            f"[bold cyan]ENTER [{trade_type.upper()}] {asset} {signal['direction'].upper()} "
            f"@ {price:.5g}  size=${size_usd:.0f}  "
            f"SL={stop_loss_pct:.2%}  TP={take_profit_pct:.2%}  "
            f"conf={confidence:.2f}[/bold cyan]"
        )

    # ── Reflection ────────────────────────────────────────────────────────────

    def _should_reflect_today(self) -> bool:
        now_et = datetime.now(ET)
        if now_et.weekday() not in REFLECTION_DAYS.values():
            return False
        if now_et.hour != 9 or now_et.minute > 5:
            return False
        today = now_et.date().isoformat()
        if self.last_reflection_date == today:
            return False
        min_trades = self.goal.get("reflection_schedule", {}).get("min_trades", 50)
        return self.trades_since_reflection >= min_trades

    def _hermes_available(self) -> bool:
        import shutil
        return shutil.which("hermes") is not None

    # ── Main tick ─────────────────────────────────────────────────────────────

    async def _tick(self):
        if self.circuit_breaker.open:
            return

        self._reload_strategy()
        strategy = self.strategy

        # 1. Update existing positions (check stops, update trailing)
        await _with_retry(self._update_positions, breaker=self.circuit_breaker)

        if not self._drawdown_ok():
            hard = self.goal.get("max_drawdown", {}).get("hard_ceiling", 0.40)
            console.print(f"[bold red]Hard drawdown ceiling {hard:.0%} hit — no new entries[/bold red]")
            self._write_heartbeat()
            return

        # 1b. End-of-day sweep: force-close equity day trades before bell
        if _approaching_market_close():
            await self._close_day_trades_eod()

        market_open = _us_market_open()
        if not market_open:
            console.print(
                f"[dim]US market closed — updating forex/crypto positions only; "
                f"no new equity entries[/dim]"
            )

        # 2. Macro context (best-effort)
        try:
            await _with_retry(macro_adapter.fetch, breaker=self.circuit_breaker)
        except Exception:
            pass

        # 3. Full universe scan — returns cached price/news for each asset
        try:
            universe = await _with_retry(
                lambda: universe_adapter.fetch(strategy),
                breaker=self.circuit_breaker,
            )
        except Exception as e:
            console.print(f"[yellow]Universe scan failed: {e}[/yellow]")
            self._write_heartbeat()
            return

        all_ranked = universe.get("all_ranked", [])
        self.last_candidates = universe.get("candidates", [])   # display-ready, stripped
        self.last_scan_total = universe.get("total_scanned", 0)

        # Print compact scan summary
        table = Table(title=f"Universe scan — {self.last_scan_total} assets", show_header=True, min_width=80)
        table.add_column("Asset", no_wrap=True)
        table.add_column("Score", justify="right")
        table.add_column("RSI", justify="right")
        table.add_column("1d Mom", justify="right")
        table.add_column("BB%b", justify="right")
        table.add_column("Vol↑", justify="center")
        table.add_column("News", justify="right")
        for c in self.last_candidates[:15]:
            table.add_row(
                c["asset"],
                f"{c['rank_score']:.3f}",
                f"{c.get('rsi', 0):.1f}",
                f"{c.get('momentum_1d', 0):+.2%}",
                f"{c.get('bollinger_pct_b', 0.5):.2f}",
                "✓" if c.get("volume_spike") else "",
                f"{c.get('news_sentiment', 0):+.2f}",
            )
        console.print(table)

        # 4. Capacity limits
        sel = strategy.get("universe_selection", {})
        max_positions = sel.get("max_positions", 20)
        max_day = sel.get("max_day_trades", 12)
        max_swing = sel.get("max_swing_trades", 10)

        # 5. Evaluate ALL candidates (use cached data — no double-fetch)
        entered = 0
        skipped = 0
        for candidate in all_ranked:
            if len(self.open_positions) >= max_positions:
                console.print(f"[dim]Portfolio full ({max_positions} positions) — stopping evaluation[/dim]")
                break

            asset = candidate["asset"]
            if asset in self.open_positions:
                continue

            price_data = candidate.get("_price_data", {})
            news_data  = candidate.get("_news_data", {})
            if not price_data:
                continue

            signal = self._evaluate_signals(price_data, news_data, strategy)
            trade_type = self._determine_trade_type(price_data, signal)

            # Gate: don't enter equity positions when US market is closed
            if _is_equity(asset) and not market_open:
                continue

            # Check type-specific capacity
            n_day  = sum(1 for p in self.open_positions.values() if p.get("trade_type") == "day")
            n_swing = sum(1 for p in self.open_positions.values() if p.get("trade_type") == "swing")
            if trade_type == "day" and n_day >= max_day:
                continue
            if trade_type == "swing" and n_swing >= max_swing:
                continue

            if signal["go"]:
                await self._enter_trade(candidate, price_data, news_data, signal, trade_type)
                entered += 1
            else:
                skipped += 1
                if skipped <= 5:  # print first few skips for debugging
                    console.print(
                        f"[dim]{asset} [{trade_type}]: skip "
                        f"(conf={signal['confidence']:.2f}, hits={signal['signals_hit']}) "
                        f"— {'; '.join(signal['reasons'][:2])}[/dim]"
                    )

        console.print(
            f"[dim]Tick: entered={entered} skipped={skipped} "
            f"open={len(self.open_positions)}/{max_positions} "
            f"(day={sum(1 for p in self.open_positions.values() if p.get('trade_type')=='day')} "
            f"swing={sum(1 for p in self.open_positions.values() if p.get('trade_type')=='swing')})[/dim]"
        )

        # 6. Reflection check
        if self._should_reflect_today():
            console.print("[bold magenta]Reflection: Mon/Wed/Fri 09:30 ET[/bold magenta]")
            mode = "--hermes" if self._hermes_available() else "--fallback"
            subprocess.Popen(
                ["python", "-m", "hermes_trading.reflect", mode],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.last_reflection_date = datetime.now(ET).date().isoformat()
            self.trades_since_reflection = 0

        self._write_heartbeat()

    # ── Run ───────────────────────────────────────────────────────────────────

    async def run(self, single_tick: bool = False):
        # Restore state from Gist on startup (Koyeb restarts lose local files)
        restored = await gist_state.load(STATE_DIR)
        if restored:
            self._load_state()

        console.print("[bold green]Trading loop started[/bold green]")
        console.print(f"  Mode:     {os.getenv('HERMES_TRADING_MODE', 'paper')}")
        console.print(f"  Balance:  ${self.portfolio_balance:,.0f}")
        console.print(f"  Target:   +{self.goal.get('target_return_14d', 0.45)*100:.0f}% "
                      f"in {self.goal.get('timeframe_days', 14)}d")
        sel = self.strategy.get("universe_selection", {})
        console.print(
            f"  Capacity: {sel.get('max_positions', 20)} positions "
            f"({sel.get('max_day_trades', 12)} day / {sel.get('max_swing_trades', 10)} swing)"
        )
        if single_tick:
            console.print("  Mode:     [bold yellow]single-tick (GitHub Actions)[/bold yellow]")

        while True:
            try:
                await self._tick()
            except Exception as e:
                console.print(f"[bold red]Tick error: {e}[/bold red]")
                self.circuit_breaker.record_failure()

            await gist_state.save(STATE_DIR)

            if single_tick:
                console.print("[dim]Single-tick complete — exiting.[/dim]")
                return

            await asyncio.sleep(60)
