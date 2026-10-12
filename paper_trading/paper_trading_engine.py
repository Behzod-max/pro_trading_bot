from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class PaperPosition:
    direction: str
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    lot: float
    opened_at: Any
    remaining_lot: float
    tp1_hit: bool = False


class PaperTradingEngine:
    """
    Simulates trades without sending orders to MT5.

    TP1 closes 50% of the position.
    TP2 closes the remainder.
    SL closes everything that remains.

    The engine expects live tick prices from the caller.
    """

    def __init__(
        self,
        initial_balance: float = 150.0,
        risk_percent: float = 1.0,
        tp1_close_percent: float = 50.0,
    ):
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.risk_percent = float(risk_percent)
        self.tp1_close_percent = min(
            max(float(tp1_close_percent), 0.0),
            100.0,
        )

        self.position: Optional[PaperPosition] = None
        self.history: list[Dict[str, Any]] = []
        self.equity_curve: list[float] = [self.balance]

        self.total_wins = 0
        self.total_losses = 0
        self.total_trades = 0
        self.realized_r = 0.0
        self.peak_balance = self.balance
        self.max_drawdown = 0.0

    @staticmethod
    def _num(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _direction(value: Any) -> str:
        value = str(value or "NEUTRAL").upper().strip()
        if value in {"BUY", "LONG", "BULLISH"}:
            return "BUY"
        if value in {"SELL", "SHORT", "BEARISH"}:
            return "SELL"
        return "NEUTRAL"

    def _update_drawdown(self) -> None:
        self.peak_balance = max(self.peak_balance, self.balance)

        if self.peak_balance > 0:
            dd = (
                (self.peak_balance - self.balance)
                / self.peak_balance
                * 100.0
            )
            self.max_drawdown = max(self.max_drawdown, dd)

        self.equity_curve.append(self.balance)

    def open_position(
        self,
        direction: str,
        entry: float,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float,
        lot: float,
        opened_at: Any = None,
    ) -> bool:
        if self.position is not None:
            return False

        direction = self._direction(direction)
        entry = self._num(entry)
        stop_loss = self._num(stop_loss)
        tp1 = self._num(take_profit_1)
        tp2 = self._num(take_profit_2)
        lot = self._num(lot)

        if direction not in {"BUY", "SELL"}:
            return False

        if entry <= 0 or stop_loss <= 0 or tp1 <= 0 or tp2 <= 0:
            return False

        if lot <= 0:
            return False

        if direction == "BUY":
            if not (stop_loss < entry < tp1 < tp2):
                return False
        else:
            if not (stop_loss > entry > tp1 > tp2):
                return False

        self.position = PaperPosition(
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            lot=lot,
            opened_at=opened_at or datetime.now().isoformat(),
            remaining_lot=lot,
        )
        return True

    def _r_multiple(
        self,
        position: PaperPosition,
        price: float,
    ) -> float:
        risk_distance = abs(
            position.entry - position.stop_loss
        )

        if risk_distance <= 0:
            return 0.0

        if position.direction == "BUY":
            return (
                price - position.entry
            ) / risk_distance

        return (
            position.entry - price
        ) / risk_distance

    def _money_from_r(
        self,
        r_value: float,
    ) -> float:
        risk_money = (
            self.balance
            * self.risk_percent
            / 100.0
        )
        return risk_money * r_value

    def _record_close(
        self,
        reason: str,
        exit_price: float,
        realized_r: float,
        closed_lot: float,
        closed_at: Any,
    ) -> Dict[str, Any]:
        position = self.position
        assert position is not None

        pnl_money = self._money_from_r(realized_r)
        self.balance += pnl_money
        self.realized_r += realized_r

        result = "WIN" if realized_r > 0 else (
            "LOSS" if realized_r < 0 else "BREAKEVEN"
        )

        self.history.append(
            {
                "direction": position.direction,
                "entry": round(position.entry, 5),
                "exit": round(exit_price, 5),
                "stop_loss": round(position.stop_loss, 5),
                "take_profit_1": round(position.take_profit_1, 5),
                "take_profit_2": round(position.take_profit_2, 5),
                "lot": round(closed_lot, 8),
                "opened_at": position.opened_at,
                "closed_at": closed_at or datetime.now().isoformat(),
                "reason": reason,
                "result": result,
                "r": round(realized_r, 4),
                "pnl": round(pnl_money, 4),
            }
        )

        self._update_drawdown()
        return self.history[-1]

    def update(
        self,
        bid: float,
        ask: float,
        timestamp: Any = None,
    ) -> list[Dict[str, Any]]:
        """
        Process one live tick.

        BUY exits are checked against Bid.
        SELL exits are checked against Ask.
        """
        if self.position is None:
            return []

        bid = self._num(bid)
        ask = self._num(ask)
        closed: list[Dict[str, Any]] = []

        if bid <= 0 or ask <= 0:
            return closed

        position = self.position

        if position.direction == "BUY":
            price = bid

            # Conservative tick rule: SL first if TP and SL are both crossed.
            if price <= position.stop_loss:
                record = self._record_close(
                    "SL",
                    position.stop_loss,
                    -1.0,
                    position.remaining_lot,
                    timestamp,
                )
                closed.append(record)
                self.total_losses += 1
                self.total_trades += 1
                self.position = None
                return closed

            if not position.tp1_hit and price >= position.take_profit_1:
                close_fraction = self.tp1_close_percent / 100.0
                close_lot = position.remaining_lot * close_fraction

                if close_lot > 0:
                    tp1_r = self._r_multiple(
                        position,
                        position.take_profit_1,
                    )

                    record = self._record_close(
                        "TP1",
                        position.take_profit_1,
                        tp1_r * (close_lot / position.lot),
                        close_lot,
                        timestamp,
                    )
                    closed.append(record)

                    position.remaining_lot -= close_lot
                    position.tp1_hit = True

            if position is not None and position.remaining_lot > 1e-12:
                if price >= position.take_profit_2:
                    tp2_r = self._r_multiple(
                        position,
                        position.take_profit_2,
                    )

                    record = self._record_close(
                        "TP2",
                        position.take_profit_2,
                        tp2_r * (
                            position.remaining_lot / position.lot
                        ),
                        position.remaining_lot,
                        timestamp,
                    )
                    closed.append(record)

                    self.total_wins += 1
                    self.total_trades += 1
                    self.position = None

        else:
            price = ask

            if price >= position.stop_loss:
                record = self._record_close(
                    "SL",
                    position.stop_loss,
                    -1.0,
                    position.remaining_lot,
                    timestamp,
                )
                closed.append(record)
                self.total_losses += 1
                self.total_trades += 1
                self.position = None
                return closed

            if not position.tp1_hit and price <= position.take_profit_1:
                close_fraction = self.tp1_close_percent / 100.0
                close_lot = position.remaining_lot * close_fraction

                if close_lot > 0:
                    tp1_r = self._r_multiple(
                        position,
                        position.take_profit_1,
                    )

                    record = self._record_close(
                        "TP1",
                        position.take_profit_1,
                        tp1_r * (close_lot / position.lot),
                        close_lot,
                        timestamp,
                    )
                    closed.append(record)

                    position.remaining_lot -= close_lot
                    position.tp1_hit = True

            if position is not None and position.remaining_lot > 1e-12:
                if price <= position.take_profit_2:
                    tp2_r = self._r_multiple(
                        position,
                        position.take_profit_2,
                    )

                    record = self._record_close(
                        "TP2",
                        position.take_profit_2,
                        tp2_r * (
                            position.remaining_lot / position.lot
                        ),
                        position.remaining_lot,
                        timestamp,
                    )
                    closed.append(record)

                    self.total_wins += 1
                    self.total_trades += 1
                    self.position = None

        return closed

    def force_close(
        self,
        price: float,
        timestamp: Any = None,
        reason: str = "END_OF_TEST",
    ) -> Optional[Dict[str, Any]]:
        if self.position is None:
            return None

        position = self.position
        r_value = self._r_multiple(position, price)
        record = self._record_close(
            reason,
            price,
            r_value * (
                position.remaining_lot
                / position.lot
            ),
            position.remaining_lot,
            timestamp,
        )

        if r_value > 0:
            self.total_wins += 1
        elif r_value < 0:
            self.total_losses += 1

        self.total_trades += 1
        self.position = None
        return record

    def summary(self) -> Dict[str, Any]:
        closed_trades = self.total_wins + self.total_losses

        win_rate = (
            self.total_wins / closed_trades * 100.0
            if closed_trades
            else 0.0
        )

        return {
            "initial_balance": round(
                self.initial_balance,
                2,
            ),
            "balance": round(
                self.balance,
                2,
            ),
            "net_profit": round(
                self.balance - self.initial_balance,
                2,
            ),
            "risk_percent": self.risk_percent,
            "total_trades": self.total_trades,
            "wins": self.total_wins,
            "losses": self.total_losses,
            "win_rate": round(win_rate, 2),
            "realized_r": round(
                self.realized_r,
                4,
            ),
            "max_drawdown_percent": round(
                self.max_drawdown,
                2,
            ),
            "open_position": (
                asdict(self.position)
                if self.position
                else None
            ),
            "history": self.history[-20:],
        }
