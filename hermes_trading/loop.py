"""
Core trading loop — runs 24/7.
Every 60s:
  1. Update open positions (check stop/target hit)
  2. Check portfolio drawdown against goal
  3. Scan universe (AI-ranked candidates)
  4. For each candidate: mandatory news analysis + signal evaluation
  5. Enter paper trades when conditions met
  6. Check reflection schedule (Mon/Wed/Fri 09:30 ET, min 50 trades)
  7. Write heartbeat
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
STRATEGY_FILE = STATE_DIR / "strategy.yaml"
GOAL_FILE = STATE_DIR / "goal.yaml"

ET = pytz.timezone("America/New_York")
REFLECTION_DAYS = {"MON": 0, "WED": 2, "FRI": 4}

console = Console()


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
        self.circuit_breaker = CircuitBreaker()
        self._load_state()

    def _load_state(self):
        with open(STRATEGY_FILE) as f:
            self.strategy = yaml.safe_load(f)
        hb = self._read_heartbeat()
        self.portfolio_balance = hb.get("portfolio_balance", self.portfolio_balance)
        self.portfolio_peak = hb.get("portfolio_peak", self.portfolio_peak)
        self.total_trades = hb.get("total_trades", 0)
        self.trades_since_reflection = hb.get("trades_since_last_reflection", 0)

    def _read_heartbeat(self) -> dict:
        if HEARTBEAT_FILE.exists():
            return json.loads(HEARTBEAT_FILE.read_text())
        return {}

    def _write_heartbeat(self, extra: dict | None = None):
        dd = (self.portfolio_peak - self.portfolio_balance) / self.portfolio_peak if self.portfolio_peak > 0 else 0.0
        data = {
            "status": "running",
            "portfolio_balance": round(self.portfolio_balance, 2),
            "portfolio_peak": round(self.portfolio_peak, 2),
            "portfolio_drawdown_pct": round(dd * 100, 2),
            "open_positions": len(self.open_positions),
            "total_trades": self.total_trades,
            "last_tick": datetime.now(timezone.utc).isoformat(),
            "last_reflection": self.last_reflection_date,
            "trades_since_last_reflection": self.trades_since_reflection,
        }
        if extra:
            data.update(extra)
        HEARTBEAT_FILE.write_text(json.dumps(data, indent=2))

    def _reload_strategy(self):
        with open(STRATEGY_FILE) as f:
            self.strategy = yaml.safe_load(f)

    def _drawdown_pct(self) -> float:
        if self.portfolio_peak <= 0:
            return 0.0
        return (self.portfolio_peak - self.portfolio_balance) / self.portfolio_peak

    def _drawdown_ok(self) -> bool:
        dd = self._drawdown_pct()
        hard_ceil = self.goal.get("max_drawdown", {}).get("hard_ceiling", 0.40)
        return dd < hard_ceil

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

    def _calculate_stop_loss(self, atr_pct: float, confidence: float, strategy: dict) -> float:
        """AI-driven stop loss: tighter on low confidence, looser on high."""
        base_mult = strategy.get("stop_loss", {}).get("base_atr_multiplier", 2.0)
        min_sl = self.goal.get("stop_loss", {}).get("min_pct", 0.5) / 100
        max_sl = self.goal.get("stop_loss", {}).get("max_pct", 15.0) / 100
        # Scale multiplier with confidence
        conf_mult = 1.0 + (confidence - 0.5) * 2.0
        sl = atr_pct * base_mult * conf_mult
        return max(min_sl, min(max_sl, sl))

    def _calculate_position_size(self, confidence: float, strategy: dict) -> float:
        """Kelly-fractional position sizing, scaled by confidence."""
        max_pos = strategy.get("position_sizing", {}).get("max_position_pct", 0.20)
        kelly_f = strategy.get("position_sizing", {}).get("kelly_fraction", 0.25)
        size = max_pos * kelly_f * confidence
        return min(max_pos, max(0.01, size))

    def _evaluate_signals(self, price_data: dict, news_data: dict, strategy: dict) -> dict:
        """Returns signal evaluation with direction, confidence, and go/no-go."""
        signals_hit = 0
        direction = "long"
        confidence = 0.0
        reasons = []

        rsi = price_data.get("rsi", 50.0)
        rsi_cfg = strategy.get("entry", {}).get("indicators", {}).get("rsi", {})
        if rsi <= rsi_cfg.get("oversold", 30):
            signals_hit += 1
            direction = "long"
            confidence += 0.25
            reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi >= rsi_cfg.get("overbought", 70):
            signals_hit += 1
            direction = "short"
            confidence += 0.25
            reasons.append(f"RSI overbought ({rsi:.1f})")

        macd = price_data.get("macd", {})
        if macd.get("crossover"):
            signals_hit += 1
            confidence += 0.25
            reasons.append("MACD crossover")

        momentum = price_data.get("momentum_5d", 0.0)
        mom_threshold = strategy.get("entry", {}).get("indicators", {}).get("momentum", {}).get("min_pct", 0.005)
        if abs(momentum) > mom_threshold:
            signals_hit += 1
            confidence += 0.15
            direction = "long" if momentum > 0 else "short"
            reasons.append(f"Momentum {momentum:.2%}")

        # News sentiment — mandatory check
        sentiment = news_data.get("sentiment", {})
        news_score = sentiment.get("score", 0.0)
        news_conf = sentiment.get("confidence", 0.0)
        news_weight = strategy.get("entry", {}).get("news_sentiment_weight", 0.35)

        if abs(news_score) > 0.2 and news_conf >= self.goal.get("news_analysis", {}).get("sentiment_min_confidence", 0.6):
            signals_hit += 1
            confidence += news_weight * abs(news_score) * news_conf
            news_dir = "long" if news_score > 0 else "short"
            reasons.append(f"News sentiment {news_score:+.2f} (conf {news_conf:.2f}) → {news_dir}")
            if news_dir != direction:
                confidence *= 0.6

        if sentiment.get("high_impact"):
            confidence = min(0.95, confidence * 1.15)
            reasons.append("High-impact news detected")

        min_confluence = strategy.get("entry", {}).get("min_signal_confluence", 1)
        min_conf_threshold = strategy.get("entry", {}).get("min_confidence", 0.20)
        go = signals_hit >= min_confluence and confidence >= min_conf_threshold

        # Drawdown override check
        dd = self._drawdown_pct()
        soft_ceil = self.goal.get("max_drawdown", {}).get("soft_ceiling", 0.225)
        override_min_conf = self.goal.get("max_drawdown", {}).get("override_min_confidence", 0.85)
        if dd > soft_ceil:
            if confidence >= override_min_conf:
                reasons.append(f"Drawdown {dd:.1%} exceeds soft ceiling — overriding (confidence {confidence:.2f} >= {override_min_conf})")
            else:
                go = False
                reasons.append(f"Drawdown {dd:.1%} exceeds soft ceiling — trade blocked (confidence {confidence:.2f} < {override_min_conf})")

        return {
            "go": go,
            "direction": direction,
            "confidence": round(min(0.99, confidence), 3),
            "signals_hit": signals_hit,
            "reasons": reasons,
        }

    def _log_trade(self, trade: dict):
        with open(TRADES_FILE, "a") as f:
            f.write(json.dumps(trade) + "\n")
        self.total_trades += 1
        self.trades_since_reflection += 1

    async def _update_positions(self):
        """Check each open position — close if stop or target hit."""
        to_close = []
        for asset, pos in self.open_positions.items():
            try:
                pd = await _with_retry(lambda a=asset: price_adapter.fetch(a))
                current_price = pd.get("price", pos["entry_price"])
                entry = pos["entry_price"]
                direction = pos["direction"]

                change_pct = (current_price - entry) / entry if direction == "long" else (entry - current_price) / entry
                stop_loss = pos["stop_loss_pct"]
                take_profit = pos["take_profit_pct"]

                if change_pct <= -stop_loss:
                    to_close.append((asset, current_price, "stop_loss", change_pct))
                elif change_pct >= take_profit:
                    to_close.append((asset, current_price, "take_profit", change_pct))
            except Exception:
                pass

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
            console.print(
                f"[{'green' if pnl_pct > 0 else 'red'}]CLOSED {asset} {reason} "
                f"pnl={pnl_pct:+.2%} (${pnl_usd:+.0f})[/]"
            )

    async def _enter_trade(self, candidate: dict, price_data: dict, news_data: dict, signal: dict):
        asset = candidate["asset"]
        if asset in self.open_positions:
            return

        strategy = self.strategy
        price = price_data.get("price", 0.0)
        atr_pct = price_data.get("atr_pct", 0.005)
        confidence = signal["confidence"]

        stop_loss_pct = self._calculate_stop_loss(atr_pct, confidence, strategy)
        rr_ratio = strategy.get("take_profit", {}).get("base_rr_ratio", 2.5)
        news_boost = min(
            strategy.get("take_profit", {}).get("news_boost_max", 0.5),
            confidence * 0.5,
        )
        take_profit_pct = stop_loss_pct * (rr_ratio + news_boost)

        position_size_pct = self._calculate_position_size(confidence, strategy)
        size_usd = self.portfolio_balance * position_size_pct

        if size_usd < 10:
            return

        mode = os.getenv("HERMES_TRADING_MODE", "paper")

        trade = {
            "id": f"{asset}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "asset": asset,
            "direction": signal["direction"],
            "entry_price": price,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "stop_loss_pct": round(stop_loss_pct, 5),
            "take_profit_pct": round(take_profit_pct, 5),
            "size_usd": round(size_usd, 2),
            "position_size_pct": round(position_size_pct, 4),
            "confidence": confidence,
            "signals": signal["reasons"],
            "news_sentiment": news_data.get("sentiment", {}).get("score"),
            "news_headlines": news_data.get("headlines", [])[:3],
            "mode": mode,
            "strategy_version": strategy.get("version", "00"),
            "status": "open",
        }

        self.open_positions[asset] = trade

        # Log drawdown override justification if applicable
        dd = self._drawdown_pct()
        soft_ceil = self.goal.get("max_drawdown", {}).get("soft_ceiling", 0.225)
        if dd > soft_ceil:
            trade["drawdown_override_justification"] = (
                f"Portfolio drawdown {dd:.1%} exceeds soft ceiling {soft_ceil:.1%}. "
                f"Entered because confidence {confidence:.2f} >= override threshold. "
                f"Signals: {'; '.join(signal['reasons'])}"
            )

        console.print(
            f"[bold cyan]ENTER {asset} {signal['direction'].upper()} "
            f"@ {price:.5f} size=${size_usd:.0f} "
            f"SL={stop_loss_pct:.2%} TP={take_profit_pct:.2%} "
            f"conf={confidence:.2f}[/bold cyan]"
        )

    async def _tick(self):
        if self.circuit_breaker.open:
            return

        self._reload_strategy()

        # Update existing positions
        await _with_retry(self._update_positions, breaker=self.circuit_breaker)

        if not self._drawdown_ok():
            hard_ceil = self.goal.get("max_drawdown", {}).get("hard_ceiling", 0.40)
            console.print(f"[bold red]Hard drawdown ceiling {hard_ceil:.0%} hit — no new entries[/bold red]")
            self._write_heartbeat()
            return

        # Get macro context
        try:
            macro = await _with_retry(macro_adapter.fetch, breaker=self.circuit_breaker)
        except Exception:
            macro = {}

        # Scan universe
        try:
            universe = await _with_retry(
                lambda: universe_adapter.fetch(self.strategy),
                breaker=self.circuit_breaker,
            )
        except Exception as e:
            console.print(f"[yellow]Universe scan failed: {e}[/yellow]")
            self._write_heartbeat()
            return

        candidates = universe.get("candidates", [])
        if not candidates:
            self._write_heartbeat()
            return

        table = Table(title="Universe scan", show_header=True)
        table.add_column("Asset")
        table.add_column("Score", justify="right")
        table.add_column("RSI", justify="right")
        table.add_column("Mom", justify="right")
        table.add_column("News", justify="right")
        for c in candidates:
            table.add_row(
                c["asset"],
                f"{c['rank_score']:.3f}",
                f"{c.get('rsi', 0):.1f}",
                f"{c.get('momentum_5d', 0):+.2%}",
                f"{c.get('news_sentiment', 0):+.2f}",
            )
        console.print(table)

        # Evaluate each candidate
        for candidate in candidates:
            asset = candidate["asset"]
            if asset in self.open_positions:
                continue
            if len(self.open_positions) >= self.strategy.get("universe_selection", {}).get("max_candidates", 5):
                break

            try:
                price_data, news_data = await asyncio.gather(
                    _with_retry(lambda a=asset: price_adapter.fetch(a)),
                    _with_retry(lambda a=asset: news_adapter.fetch(a)),
                )
            except Exception as e:
                console.print(f"[yellow]{asset}: data fetch failed — {e}[/yellow]")
                continue

            signal = self._evaluate_signals(price_data, news_data, self.strategy)

            if signal["go"]:
                await self._enter_trade(candidate, price_data, news_data, signal)
            else:
                console.print(
                    f"[dim]{asset}: skip (conf={signal['confidence']:.2f}, "
                    f"signals={signal['signals_hit']}) — "
                    f"{'; '.join(signal['reasons'][:2])}[/dim]"
                )

        # Reflection check
        if self._should_reflect_today():
            console.print("[bold magenta]Reflection trigger: Mon/Wed/Fri 09:30 ET — running Hermes reflection[/bold magenta]")
            mode = "--hermes" if self._hermes_available() else "--fallback"
            subprocess.Popen(
                ["python", "-m", "hermes_trading.reflect", mode],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.last_reflection_date = datetime.now(ET).date().isoformat()
            self.trades_since_reflection = 0

        self._write_heartbeat()

    def _hermes_available(self) -> bool:
        import shutil
        return shutil.which("hermes") is not None

    async def run(self, single_tick: bool = False):
        # Restore state from Gist on startup (Koyeb restarts lose local files)
        restored = await gist_state.load(STATE_DIR)
        if restored:
            self._load_state()   # re-read now that files are refreshed from Gist

        console.print("[bold green]Trading loop started[/bold green]")
        console.print(f"  Mode:    {os.getenv('HERMES_TRADING_MODE', 'paper')}")
        console.print(f"  Balance: ${self.portfolio_balance:,.0f}")
        console.print(f"  Target:  +{self.goal.get('target_return_14d', 0.45)*100:.0f}% in {self.goal.get('timeframe_days', 14)}d")
        if single_tick:
            console.print("  Mode:    [bold yellow]single-tick (GitHub Actions)[/bold yellow]")

        while True:
            try:
                await self._tick()
            except Exception as e:
                console.print(f"[bold red]Tick error: {e}[/bold red]")
                self.circuit_breaker.record_failure()

            # Push state to Gist so a restart doesn't lose progress
            await gist_state.save(STATE_DIR)

            if single_tick:
                console.print("[dim]Single-tick complete — exiting.[/dim]")
                return

            await asyncio.sleep(60)
