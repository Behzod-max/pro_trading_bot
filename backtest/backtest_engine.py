from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import pandas as pd

from indicators.technical import TechnicalIndicators
from analysis.market_structure import MarketStructure
from analysis.trendline import TrendLineEngine
from analysis.liquidity import LiquidityEngine
from analysis.fibonacci import FibonacciEngine
from analysis.candle_confirmation import CandleConfirmationEngine
from analysis.signal_engine import SignalEngine


@dataclass
class BacktestTrade:
    entry_time: Any
    exit_time: Any
    direction: str
    entry: float
    stop_loss: float
    take_profit: float
    exit_price: float
    result: str
    pnl_r: float


class BacktestEngine:
    """
    Historical OHLC backtest for the existing signal stack.

    Design goals:
    - Uses only CLOSED candles for signal creation.
    - Enters on the NEXT candle open to reduce look-ahead bias.
    - Uses one TP target and one SL.
    - Does not send real orders.
    """

    def __init__(
        self,
        initial_balance: float = 150.0,
        risk_percent: float = 1.0,
        rr: float = 1.5,
        min_score: float = 60.0,
        warmup: int = 250,
        max_bars: Optional[int] = None,
    ):
        self.initial_balance = float(initial_balance)
        self.risk_percent = float(risk_percent)
        self.rr = float(rr)
        self.min_score = float(min_score)
        self.warmup = max(int(warmup), 50)
        self.max_bars = max_bars

        self.indicators = TechnicalIndicators()
        self.structure_engine = MarketStructure()
        self.trendline_engine = TrendLineEngine()
        self.liquidity_engine = LiquidityEngine()
        self.fibonacci_engine = FibonacciEngine()
        self.candle_engine = CandleConfirmationEngine()
        self.signal_engine = SignalEngine()

    @staticmethod
    def _num(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _norm_direction(value: Any) -> str:
        text = str(value or "NEUTRAL").upper().strip()
        if text in {"BUY", "LONG", "BULLISH", "UP"}:
            return "BUY"
        if text in {"SELL", "SHORT", "BEARISH", "DOWN"}:
            return "SELL"
        return "NEUTRAL"

    @staticmethod
    def _closed_frame(df: pd.DataFrame, end_index: int) -> pd.DataFrame:
        """
        end_index is the index of the latest CLOSED candle allowed for analysis.
        """
        return df.iloc[: end_index + 1].copy()

    def _build_signal_inputs(self, df: pd.DataFrame, end_index: int):
        """
        Builds the same analysis stack used by the live bot, but strictly
        from candles available up to end_index.
        """
        window = self._closed_frame(df, end_index)

        indicator_data = {"M5": self.indicators.add_all(window.copy())}

        m5 = indicator_data["M5"]
        if len(m5) < self.warmup:
            return None

        structure = self.structure_engine.analyze(m5)

        trendline = self.trendline_engine.analyze(
            m5,
            structure,
        )

        liquidity = self.liquidity_engine.analyze(m5)

        structure_trend = str(
            structure.get("trend", "ARALASH")
        ).upper()

        if structure_trend in {"KO_TARILISH", "UP", "UPTREND", "BULLISH"}:
            fib_direction = "BULLISH"
        elif structure_trend in {"PASAYISH", "DOWN", "DOWNTREND", "BEARISH"}:
            fib_direction = "BEARISH"
        else:
            fib_direction = None

        fibonacci = None
        if fib_direction:
            closed = m5.iloc[-1]
            fibonacci = self.fibonacci_engine.analyze_from_structure(
                current_price=self._num(closed.get("close")),
                structure_result=structure,
                direction=fib_direction,
            )

        candle = self.candle_engine.analyze_dict(m5)

        # Simple MTF proxy for historical testing: EMA trend on resampled
        # higher timeframes. This avoids future leakage while keeping the
        # test independent from a second data download.
        mtf_data = {"M5": m5}
        mtf_trends = {"M5": self._ema_trend(m5)}

        for tf, rule in (
            ("M15", "15min"),
            ("M30", "30min"),
            ("H1", "1h"),
            ("H4", "4h"),
            ("D1", "1D"),
        ):
            try:
                tmp = window.copy()
                if "time" not in tmp.columns:
                    break

                tmp["time"] = pd.to_datetime(tmp["time"])
                tmp = tmp.set_index("time")
                resampled = (
                    tmp[["open", "high", "low", "close", "tick_volume"]]
                    .resample(rule)
                    .agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "tick_volume": "sum",
                        }
                    )
                    .dropna()
                    .reset_index()
                )

                if len(resampled) >= 50:
                    calc = self.indicators.add_all(resampled)
                    mtf_data[tf] = calc
                    mtf_trends[tf] = self._ema_trend(calc)
            except Exception:
                continue

        structure_results = {"M5": structure}
        trendline_results = {"M5": trendline}
        liquidity_results = {"M5": liquidity}
        fibonacci_results = {"M5": fibonacci} if fibonacci else {}
        candle_results = {"M5": candle}

        signal = self.signal_engine.analyze(
            mtf_data=mtf_data,
            mtf_trends=mtf_trends,
            structure_results=structure_results,
            trendline_results=trendline_results,
            liquidity_results=liquidity_results,
            fibonacci_results=fibonacci_results,
            candle_results=candle_results,
        )

        return signal

    def _ema_trend(self, df: pd.DataFrame) -> str:
        if len(df) < 2:
            return "UNKNOWN"

        row = df.iloc[-1]
        close = self._num(row.get("close"))
        ema20 = self._num(row.get("ema_20"))
        ema50 = self._num(row.get("ema_50"))
        ema200 = self._num(row.get("ema_200"))

        if close > ema20 and ema20 > ema50 and ema50 > ema200:
            return "UPTREND"
        if close < ema20 and ema20 < ema50 and ema50 < ema200:
            return "DOWNTREND"
        if close > ema50:
            return "UPTREND"
        if close < ema50:
            return "DOWNTREND"
        return "RANGE"

    def _levels(
        self,
        df: pd.DataFrame,
        direction: str,
        entry: float,
    ):
        tail = df.tail(50)

        atr = 0.0
        if "atr_14" in tail.columns:
            atr = self._num(tail.iloc[-1].get("atr_14"))

        recent_low = self._num(tail["low"].min())
        recent_high = self._num(tail["high"].max())

        if direction == "BUY":
            stop = recent_low - max(atr * 0.20, 0.01)
            risk = entry - stop
            if risk <= 0:
                return None
            take_profit = entry + risk * self.rr
        elif direction == "SELL":
            stop = recent_high + max(atr * 0.20, 0.01)
            risk = stop - entry
            if risk <= 0:
                return None
            take_profit = entry - risk * self.rr
        else:
            return None

        return stop, take_profit, risk

    def _simulate_trade(
        self,
        df: pd.DataFrame,
        entry_index: int,
        direction: str,
        entry: float,
        stop: float,
        target: float,
    ):
        for i in range(entry_index, len(df)):
            bar = df.iloc[i]
            high = self._num(bar.get("high"))
            low = self._num(bar.get("low"))

            if direction == "BUY":
                hit_sl = low <= stop
                hit_tp = high >= target
            else:
                hit_sl = high >= stop
                hit_tp = low <= target

            # Conservative convention: if both are touched in the same
            # candle and no intrabar data exists, assume SL happened first.
            if hit_sl and hit_tp:
                return i, stop, "LOSS"
            if hit_sl:
                return i, stop, "LOSS"
            if hit_tp:
                return i, target, "WIN"

        return len(df) - 1, self._num(
            df.iloc[-1].get("close")
        ), "OPEN"

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df DataFrame bo'lishi kerak.")

        required = {"open", "high", "low", "close"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(
                f"Kerakli ustunlar yo'q: {sorted(missing)}"
            )

        data = df.copy().reset_index(drop=True)
        if self.max_bars and len(data) > self.max_bars:
            data = data.tail(self.max_bars).reset_index(drop=True)

        if len(data) < self.warmup + 10:
            raise ValueError(
                f"Backtest uchun kamida {self.warmup + 10} candle kerak."
            )

        trades: list[BacktestTrade] = []
        balance = self.initial_balance
        wins = 0
        losses = 0

        i = self.warmup
        while i < len(data) - 1:
            try:
                signal = self._build_signal_inputs(data, i)
            except Exception:
                signal = None

            if signal is None:
                i += 1
                continue

            direction = self._norm_direction(
                getattr(signal, "direction", "NEUTRAL")
            )
            score = self._num(
                getattr(signal, "score", 0)
            )
            status = str(
                getattr(signal, "status", "NO_SIGNAL")
            ).upper()

            candle_confirmed = True
            candle_results = getattr(
                signal,
                "warnings",
                [],
            ) or []

            if status == "NO_SIGNAL" or score < self.min_score:
                i += 1
                continue

            # Signal Engine already applies its confirmation gate.
            # Do not backtest an explicit conflict warning as a trade.
            if any(
                "Candle confirmation" in str(w)
                and "yo'q" in str(w).lower()
                for w in candle_results
            ):
                candle_confirmed = False

            if not candle_confirmed:
                i += 1
                continue

            entry_index = i + 1
            entry = self._num(
                data.iloc[entry_index]["open"]
            )

            levels = self._levels(
                data.iloc[: entry_index],
                direction,
                entry,
            )
            if not levels:
                i += 1
                continue

            stop, target, risk_distance = levels

            exit_index, exit_price, result = self._simulate_trade(
                data,
                entry_index,
                direction,
                entry,
                stop,
                target,
            )

            if result == "WIN":
                pnl_r = self.rr
                wins += 1
                risk_money = balance * self.risk_percent / 100.0
                balance += risk_money * pnl_r
            elif result == "LOSS":
                pnl_r = -1.0
                losses += 1
                risk_money = balance * self.risk_percent / 100.0
                balance -= risk_money
            else:
                pnl_r = 0.0

            trades.append(
                BacktestTrade(
                    entry_time=(
                        data.iloc[entry_index].get("time", entry_index)
                    ),
                    exit_time=(
                        data.iloc[exit_index].get("time", exit_index)
                    ),
                    direction=direction,
                    entry=round(entry, 5),
                    stop_loss=round(stop, 5),
                    take_profit=round(target, 5),
                    exit_price=round(exit_price, 5),
                    result=result,
                    pnl_r=round(pnl_r, 3),
                )
            )

            # One position at a time.
            i = max(exit_index + 1, i + 1)

        total = wins + losses
        win_rate = (wins / total * 100.0) if total else 0.0
        total_r = sum(t.pnl_r for t in trades)

        return {
            "initial_balance": round(self.initial_balance, 2),
            "final_balance": round(balance, 2),
            "net_profit": round(balance - self.initial_balance, 2),
            "risk_percent": round(self.risk_percent, 3),
            "rr": round(self.rr, 3),
            "min_score": round(self.min_score, 2),
            "trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 2),
            "total_r": round(total_r, 3),
            "trade_log": [asdict(t) for t in trades],
        }
