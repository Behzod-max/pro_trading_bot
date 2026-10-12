from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List


@dataclass
class RiskResult:
    valid: bool
    direction: str
    balance: float
    risk_percent: float
    risk_amount: float
    entry: float
    stop_loss: float
    stop_distance: float
    tick_size: float
    tick_value: float
    raw_lot: float
    lot: float
    estimated_risk: float
    spread: float
    spread_percent: float
    warnings: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


class RiskManager:
    """
    Lot sizing and hard risk validation.

    Important:
    - Never rounds a lot UP when doing risk sizing.
    - If broker minimum lot would exceed the configured money risk,
      the result is INVALID instead of forcing an unsafe trade.
    - This module does not send orders.
    """

    def __init__(
        self,
        balance: float = 150.0,
        risk_percent: float = 1.0,
        min_lot: float = 0.01,
        max_lot: float = 50.0,
        lot_step: float = 0.01,
        max_spread: float = 1.50,
    ):
        self.balance = float(balance)
        self.risk_percent = float(risk_percent)
        self.min_lot = float(min_lot)
        self.max_lot = float(max_lot)
        self.lot_step = float(lot_step)
        self.max_spread = float(max_spread)

    def calculate_risk_amount(self) -> float:
        return max(self.balance * self.risk_percent / 100.0, 0.0)

    def calculate_lot(
        self,
        entry: float,
        stop_loss: float,
        tick_size: float,
        tick_value: float,
    ) -> float:
        distance = abs(float(entry) - float(stop_loss))

        if distance <= 0:
            return 0.0
        if tick_size <= 0 or tick_value <= 0:
            return 0.0

        loss_per_lot = (distance / tick_size) * tick_value
        if loss_per_lot <= 0:
            return 0.0

        return self.calculate_risk_amount() / loss_per_lot

    def normalize_lot(self, raw_lot: float) -> float:
        """
        Floor to broker lot step.
        Returns 0 when raw lot is below the broker minimum.
        """
        if raw_lot <= 0:
            return 0.0

        if self.lot_step <= 0:
            return 0.0

        lot = math.floor(
            (raw_lot + 1e-12) / self.lot_step
        ) * self.lot_step

        lot = min(lot, self.max_lot)

        if lot < self.min_lot:
            return 0.0

        return round(lot, 8)

    def calculate_estimated_risk(
        self,
        entry: float,
        stop_loss: float,
        lot: float,
        tick_size: float,
        tick_value: float,
    ) -> float:
        if lot <= 0:
            return 0.0

        distance = abs(float(entry) - float(stop_loss))
        if distance <= 0 or tick_size <= 0 or tick_value <= 0:
            return 0.0

        return (distance / tick_size) * tick_value * lot

    @staticmethod
    def _spread_percent(entry: float, spread: float) -> float:
        if entry <= 0:
            return 0.0
        return abs(spread) / entry * 100.0

    def analyze(
        self,
        direction: str,
        entry: float,
        stop_loss: float,
        tick_size: float,
        tick_value: float,
        spread: float = 0.0,
    ) -> RiskResult:
        direction = str(direction or "NEUTRAL").upper().strip()
        entry = float(entry or 0.0)
        stop_loss = float(stop_loss or 0.0)
        tick_size = float(tick_size or 0.0)
        tick_value = float(tick_value or 0.0)
        spread = max(float(spread or 0.0), 0.0)

        risk_amount = self.calculate_risk_amount()
        warnings: List[str] = []
        reasons: List[str] = []

        valid = True

        if direction not in {"BUY", "SELL"}:
            valid = False
            warnings.append("Yo'nalish BUY yoki SELL bo'lishi kerak.")

        if entry <= 0 or stop_loss <= 0:
            valid = False
            warnings.append("Entry/Stop Loss noto'g'ri.")

        if direction == "BUY" and stop_loss >= entry:
            valid = False
            warnings.append("BUY uchun Stop Loss Entry'dan past bo'lishi kerak.")

        if direction == "SELL" and stop_loss <= entry:
            valid = False
            warnings.append("SELL uchun Stop Loss Entry'dan yuqori bo'lishi kerak.")

        if tick_size <= 0:
            valid = False
            warnings.append("Tick Size brokerdan olinmadi.")

        if tick_value <= 0:
            valid = False
            warnings.append("Tick Value brokerdan olinmadi.")

        if spread > self.max_spread:
            valid = False
            warnings.append(
                f"Spread juda katta: {spread:.2f} > {self.max_spread:.2f}."
            )

        stop_distance = abs(entry - stop_loss)

        raw_lot = self.calculate_lot(
            entry,
            stop_loss,
            tick_size,
            tick_value,
        )

        lot = self.normalize_lot(raw_lot)

        estimated_risk = self.calculate_estimated_risk(
            entry=entry,
            stop_loss=stop_loss,
            lot=lot,
            tick_size=tick_size,
            tick_value=tick_value,
        )

        spread_percent = self._spread_percent(
            entry,
            spread,
        )

        if raw_lot <= 0:
            valid = False
            warnings.append(
                "Risk uchun hisoblangan lot 0 ga teng yoki juda kichik."
            )

        if 0 < raw_lot < self.min_lot:
            valid = False
            warnings.append(
                f"Hisoblangan lot {raw_lot:.6f}, "
                f"broker minimum loti {self.min_lot:.2f}. "
                "Bu balans va SL bilan 1% riskga mos savdo hajmi mavjud emas."
            )

        if lot > 0:
            reasons.append(
                f"Hisoblangan lot: {lot:.2f}."
            )
        else:
            reasons.append(
                "Broker minimum loti konfiguratsiyalangan riskdan oshib ketadi; "
                "savdo o'tkazilmaydi."
            )

        if estimated_risk > risk_amount + 1e-9:
            valid = False
            warnings.append(
                f"Taxminiy risk ${estimated_risk:.2f} "
                f"maksimal ${risk_amount:.2f} riskdan oshadi."
            )

        if lot > 0 and estimated_risk <= risk_amount + 1e-9:
            reasons.append(
                f"Taxminiy risk ${estimated_risk:.2f} "
                f"(maksimal ${risk_amount:.2f})."
            )

        if spread <= self.max_spread:
            reasons.append(
                f"Spread qabul qilinadigan diapazonda: {spread:.2f}."
            )

        return RiskResult(
            valid=bool(valid),
            direction=direction,
            balance=self.balance,
            risk_percent=self.risk_percent,
            risk_amount=risk_amount,
            entry=entry,
            stop_loss=stop_loss,
            stop_distance=stop_distance,
            tick_size=tick_size,
            tick_value=tick_value,
            raw_lot=raw_lot,
            lot=lot,
            estimated_risk=estimated_risk,
            spread=spread,
            spread_percent=spread_percent,
            warnings=warnings,
            reasons=reasons,
        )
