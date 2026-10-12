from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import math
import pandas as pd


@dataclass
class EntryExitResult:
    """
    Professional Entry / SL / TP natijasi.
    """

    direction: str

    entry: float
    stop_loss: float

    take_profit_1: float
    take_profit_2: float

    risk_distance: float
    reward_distance_1: float
    reward_distance_2: float

    risk_reward_1: float
    risk_reward_2: float

    atr: float

    support: Optional[float]
    resistance: Optional[float]

    swing_high: Optional[float]
    swing_low: Optional[float]

    valid: bool
    warnings: list[str]
    reasons: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "risk_distance": self.risk_distance,
            "reward_distance_1": self.reward_distance_1,
            "reward_distance_2": self.reward_distance_2,
            "risk_reward_1": self.risk_reward_1,
            "risk_reward_2": self.risk_reward_2,
            "atr": self.atr,
            "support": self.support,
            "resistance": self.resistance,
            "swing_high": self.swing_high,
            "swing_low": self.swing_low,
            "valid": self.valid,
            "warnings": self.warnings,
            "reasons": self.reasons,
        }


class EntryExitOptimizer:
    """
    GOLD/XAUUSD uchun Entry / Stop Loss / Take Profit optimizer.

    Asosiy maqsad:
        - SL'ni keragidan ortiq uzoqlashtirmaslik
        - strukturaviy SL ishlatish
        - ATR buffer qo'shish
        - TP1 va TP2 ni RR asosida hisoblash
        - mavjud support/resistance darajalarini hisobga olish
        - spreadni tekshirish
        - noto'g'ri Entry/SL/TP kombinatsiyalarini rad etish

    Muhim:
        Bu modul order ochmaydi.
        Faqat savdo parametrlarini hisoblaydi.
    """

    MIN_RR_1 = 1.50
    MIN_RR_2 = 2.00

    DEFAULT_ATR_MULTIPLIER = 1.20
    DEFAULT_TP1_RR = 1.50
    DEFAULT_TP2_RR = 2.50

    DEFAULT_SWING_LOOKBACK = 50

    def __init__(
        self,
        atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
        tp1_rr: float = DEFAULT_TP1_RR,
        tp2_rr: float = DEFAULT_TP2_RR,
        swing_lookback: int = DEFAULT_SWING_LOOKBACK,
        min_rr_1: float = MIN_RR_1,
        min_rr_2: float = MIN_RR_2,
    ) -> None:

        if atr_multiplier <= 0:
            raise ValueError("atr_multiplier 0 dan katta bo'lishi kerak.")

        if tp1_rr <= 0:
            raise ValueError("tp1_rr 0 dan katta bo'lishi kerak.")

        if tp2_rr <= 0:
            raise ValueError("tp2_rr 0 dan katta bo'lishi kerak.")

        if swing_lookback < 5:
            raise ValueError("swing_lookback kamida 5 bo'lishi kerak.")

        self.atr_multiplier = float(atr_multiplier)
        self.tp1_rr = float(tp1_rr)
        self.tp2_rr = float(tp2_rr)

        self.swing_lookback = int(swing_lookback)

        self.min_rr_1 = float(min_rr_1)
        self.min_rr_2 = float(min_rr_2)

    # =========================================================
    # GENERAL HELPERS
    # =========================================================

    @staticmethod
    def _normalize_direction(direction: Any) -> str:
        if direction is None:
            return "NEUTRAL"

        value = str(direction).strip().upper()

        if "BUY" in value:
            return "BUY"

        if "SELL" in value:
            return "SELL"

        return "NEUTRAL"

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None

            result = float(value)

            if not math.isfinite(result):
                return None

            return result

        except (TypeError, ValueError):
            return None

    @staticmethod
    def _round_price(price: float, digits: int = 2) -> float:
        return round(float(price), digits)

    # =========================================================
    # DATA VALIDATION
    # =========================================================

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df pandas DataFrame bo'lishi kerak.")

        if df.empty:
            raise ValueError("DataFrame bo'sh.")

        required = {"open", "high", "low", "close"}

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"DataFrame'da kerakli ustunlar yo'q: {sorted(missing)}"
            )

        if len(df) < 20:
            raise ValueError(
                "Entry/SL/TP hisoblash uchun kamida 20 ta candle kerak."
            )

    # =========================================================
    # LAST CLOSED CANDLE
    # =========================================================

    @staticmethod
    def _last_closed_candle(df: pd.DataFrame) -> pd.Series:
        """
        Signal hisoblashda hozirgi shakllanayotgan candle emas,
        oxirgi yopilgan candle ishlatiladi.
        """

        if len(df) < 2:
            raise ValueError("Yopilgan candle topilmadi.")

        return df.iloc[-2]

    # =========================================================
    # ATR
    # =========================================================

    @staticmethod
    def _calculate_atr(
        df: pd.DataFrame,
        period: int = 14,
    ) -> float:

        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")

        previous_close = close.shift(1)

        true_range = pd.concat(
            [
                high - low,
                (high - previous_close).abs(),
                (low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = true_range.rolling(
            window=period,
            min_periods=period,
        ).mean()

        value = atr.iloc[-2]

        if pd.isna(value):
            value = atr.dropna().iloc[-1]

        result = float(value)

        if result <= 0 or not math.isfinite(result):
            raise ValueError("ATR noto'g'ri hisoblandi.")

        return result

    # =========================================================
    # SWING LEVELS
    # =========================================================

    def _get_swing_levels(
        self,
        df: pd.DataFrame,
    ) -> tuple[Optional[float], Optional[float]]:

        closed_df = df.iloc[:-1]

        if closed_df.empty:
            return None, None

        lookback_df = closed_df.tail(self.swing_lookback)

        swing_high = self._safe_float(
            lookback_df["high"].max()
        )

        swing_low = self._safe_float(
            lookback_df["low"].min()
        )

        return swing_high, swing_low

    # =========================================================
    # SUPPORT / RESISTANCE
    # =========================================================

    def _get_support_resistance(
        self,
        df: pd.DataFrame,
    ) -> tuple[Optional[float], Optional[float]]:

        closed_df = df.iloc[:-1]

        if len(closed_df) < 10:
            return None, None

        recent = closed_df.tail(self.swing_lookback)

        support = self._safe_float(
            recent["low"].min()
        )

        resistance = self._safe_float(
            recent["high"].max()
        )

        return support, resistance

    # =========================================================
    # ENTRY
    # =========================================================

    def _calculate_entry(
        self,
        df: pd.DataFrame,
        direction: str,
        current_price: Optional[float] = None,
    ) -> float:

        if current_price is not None:
            price = self._safe_float(current_price)

            if price is not None and price > 0:
                return price

        candle = self._last_closed_candle(df)

        close = self._safe_float(candle["close"])

        if close is None or close <= 0:
            raise ValueError("Entry uchun yopilgan candle close topilmadi.")

        return close

    # =========================================================
    # STOP LOSS
    # =========================================================

    def _calculate_stop_loss(
        self,
        entry: float,
        direction: str,
        atr: float,
        swing_high: Optional[float],
        swing_low: Optional[float],
    ) -> float:

        buffer = atr * self.atr_multiplier

        if direction == "BUY":

            candidates = []

            if swing_low is not None and swing_low < entry:
                candidates.append(swing_low - buffer)

            if not candidates:
                return entry - buffer

            stop = max(candidates)

            if stop >= entry:
                stop = entry - buffer

            return stop

        if direction == "SELL":

            candidates = []

            if swing_high is not None and swing_high > entry:
                candidates.append(swing_high + buffer)

            if not candidates:
                return entry + buffer

            stop = min(candidates)

            if stop <= entry:
                stop = entry + buffer

            return stop

        raise ValueError("Direction BUY yoki SELL bo'lishi kerak.")

    # =========================================================
    # TAKE PROFITS
    # =========================================================

    def _calculate_take_profits(
        self,
        entry: float,
        stop_loss: float,
        direction: str,
    ) -> tuple[float, float]:

        risk_distance = abs(entry - stop_loss)

        if risk_distance <= 0:
            raise ValueError("Risk distance 0 bo'lishi mumkin emas.")

        reward_1 = risk_distance * self.tp1_rr
        reward_2 = risk_distance * self.tp2_rr

        if direction == "BUY":

            tp1 = entry + reward_1
            tp2 = entry + reward_2

        elif direction == "SELL":

            tp1 = entry - reward_1
            tp2 = entry - reward_2

        else:
            raise ValueError("Direction BUY yoki SELL bo'lishi kerak.")

        return tp1, tp2

    # =========================================================
    # RR
    # =========================================================

    @staticmethod
    def _calculate_rr(
        entry: float,
        stop_loss: float,
        take_profit: float,
    ) -> float:

        risk = abs(entry - stop_loss)

        if risk <= 0:
            return 0.0

        reward = abs(take_profit - entry)

        return reward / risk

    # =========================================================
    # VALIDATION
    # =========================================================

    def _validate_result(
        self,
        entry: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        direction: str,
        spread: Optional[float],
        atr: float,
    ) -> tuple[bool, list[str]]:

        warnings: list[str] = []

        valid = True

        # -----------------------------------------------------
        # PRICE ORDER
        # -----------------------------------------------------

        if direction == "BUY":

            if stop_loss >= entry:
                valid = False
                warnings.append(
                    "BUY uchun Stop Loss Entry'dan past bo'lishi kerak."
                )

            if tp1 <= entry:
                valid = False
                warnings.append(
                    "BUY uchun TP1 Entry'dan yuqorida bo'lishi kerak."
                )

            if tp2 <= tp1:
                valid = False
                warnings.append(
                    "BUY uchun TP2 TP1'dan yuqorida bo'lishi kerak."
                )

        elif direction == "SELL":

            if stop_loss <= entry:
                valid = False
                warnings.append(
                    "SELL uchun Stop Loss Entry'dan yuqorida bo'lishi kerak."
                )

            if tp1 >= entry:
                valid = False
                warnings.append(
                    "SELL uchun TP1 Entry'dan pastda bo'lishi kerak."
                )

            if tp2 >= tp1:
                valid = False
                warnings.append(
                    "SELL uchun TP2 TP1'dan pastda bo'lishi kerak."
                )

        # -----------------------------------------------------
        # RR
        # -----------------------------------------------------

        rr1 = self._calculate_rr(
            entry,
            stop_loss,
            tp1,
        )

        rr2 = self._calculate_rr(
            entry,
            stop_loss,
            tp2,
        )

        tolerance = 1e-9

        if rr1 + tolerance < self.min_rr_1:

            valid = False

            warnings.append(
                f"RR1 {rr1:.2f}, minimal talab {self.min_rr_1:.2f}."
            )

        if rr2 + tolerance < self.min_rr_2:

            valid = False

            warnings.append(
                f"RR2 {rr2:.2f}, minimal talab {self.min_rr_2:.2f}."
            )

        # -----------------------------------------------------
        # SPREAD
        # -----------------------------------------------------

        if spread is not None:

            spread_value = self._safe_float(spread)

            if spread_value is not None and spread_value > 0:

                # Spread ATR'ning juda katta qismini egallamasligi kerak.
                if atr > 0 and spread_value > atr * 0.20:

                    warnings.append(
                        "Spread ATR'ga nisbatan yuqori."
                    )

        # -----------------------------------------------------
        # ATR
        # -----------------------------------------------------

        if atr <= 0:

            valid = False

            warnings.append(
                "ATR noto'g'ri yoki 0."
            )

        return valid, warnings

    # =========================================================
    # MAIN ANALYSIS
    # =========================================================

    def analyze(
        self,
        df: pd.DataFrame,
        direction: str,
        current_price: Optional[float] = None,
        spread: Optional[float] = None,
        digits: int = 2,
    ) -> EntryExitResult:

        self._validate_dataframe(df)

        direction = self._normalize_direction(direction)

        if direction not in {"BUY", "SELL"}:
            raise ValueError(
                "EntryExitOptimizer faqat BUY yoki SELL uchun ishlaydi."
            )

        # -----------------------------------------------------
        # ATR
        # -----------------------------------------------------

        atr = self._calculate_atr(df)

        # -----------------------------------------------------
        # SWING
        # -----------------------------------------------------

        swing_high, swing_low = self._get_swing_levels(df)

        # -----------------------------------------------------
        # SUPPORT / RESISTANCE
        # -----------------------------------------------------

        support, resistance = self._get_support_resistance(df)

        # -----------------------------------------------------
        # ENTRY
        # -----------------------------------------------------

        entry = self._calculate_entry(
            df=df,
            direction=direction,
            current_price=current_price,
        )

        # -----------------------------------------------------
        # STOP LOSS
        # -----------------------------------------------------

        stop_loss = self._calculate_stop_loss(
            entry=entry,
            direction=direction,
            atr=atr,
            swing_high=swing_high,
            swing_low=swing_low,
        )

        # -----------------------------------------------------
        # TAKE PROFITS
        # -----------------------------------------------------

        tp1, tp2 = self._calculate_take_profits(
            entry=entry,
            stop_loss=stop_loss,
            direction=direction,
        )

        # -----------------------------------------------------
        # DISTANCES
        # -----------------------------------------------------

        risk_distance = abs(entry - stop_loss)

        reward_distance_1 = abs(tp1 - entry)
        reward_distance_2 = abs(tp2 - entry)

        # -----------------------------------------------------
        # RR
        # -----------------------------------------------------

        rr1 = self._calculate_rr(
            entry,
            stop_loss,
            tp1,
        )

        rr2 = self._calculate_rr(
            entry,
            stop_loss,
            tp2,
        )

        # -----------------------------------------------------
        # VALIDATION
        # -----------------------------------------------------

        valid, warnings = self._validate_result(
            entry=entry,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            direction=direction,
            spread=spread,
            atr=atr,
        )

        # -----------------------------------------------------
        # REASONS
        # -----------------------------------------------------

        reasons: list[str] = []

        reasons.append(
            f"{direction} yo'nalish uchun Entry oxirgi yopilgan "
            f"candle/current narx asosida hisoblandi."
        )

        reasons.append(
            f"ATR(14): {atr:.{digits}f}."
        )

        if direction == "BUY":

            if swing_low is not None:
                reasons.append(
                    f"BUY SL uchun swing low: "
                    f"{swing_low:.{digits}f}."
                )

            reasons.append(
                f"ATR buffer: "
                f"{atr * self.atr_multiplier:.{digits}f}."
            )

        else:

            if swing_high is not None:
                reasons.append(
                    f"SELL SL uchun swing high: "
                    f"{swing_high:.{digits}f}."
                )

            reasons.append(
                f"ATR buffer: "
                f"{atr * self.atr_multiplier:.{digits}f}."
            )

        reasons.append(
            f"TP1 RR: {self.tp1_rr:.2f}."
        )

        reasons.append(
            f"TP2 RR: {self.tp2_rr:.2f}."
        )

        # -----------------------------------------------------
        # RESULT
        # -----------------------------------------------------

        return EntryExitResult(
            direction=direction,

            entry=self._round_price(entry, digits),
            stop_loss=self._round_price(stop_loss, digits),

            take_profit_1=self._round_price(tp1, digits),
            take_profit_2=self._round_price(tp2, digits),

            risk_distance=self._round_price(
                risk_distance,
                digits,
            ),

            reward_distance_1=self._round_price(
                reward_distance_1,
                digits,
            ),

            reward_distance_2=self._round_price(
                reward_distance_2,
                digits,
            ),

            risk_reward_1=round(rr1, 2),
            risk_reward_2=round(rr2, 2),

            atr=self._round_price(
                atr,
                digits,
            ),

            support=(
                self._round_price(support, digits)
                if support is not None
                else None
            ),

            resistance=(
                self._round_price(resistance, digits)
                if resistance is not None
                else None
            ),

            swing_high=(
                self._round_price(swing_high, digits)
                if swing_high is not None
                else None
            ),

            swing_low=(
                self._round_price(swing_low, digits)
                if swing_low is not None
                else None
            ),

            valid=valid,

            warnings=warnings,

            reasons=reasons,
        )

    # =========================================================
    # SIMPLE DICT API
    # =========================================================

    def analyze_dict(
        self,
        df: pd.DataFrame,
        direction: str,
        current_price: Optional[float] = None,
        spread: Optional[float] = None,
        digits: int = 2,
    ) -> Dict[str, Any]:

        result = self.analyze(
            df=df,
            direction=direction,
            current_price=current_price,
            spread=spread,
            digits=digits,
        )

        return result.to_dict()