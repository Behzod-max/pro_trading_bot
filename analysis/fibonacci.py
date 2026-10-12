from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# ==========================================================
# FIBONACCI DATA MODEL
# ==========================================================

@dataclass(frozen=True)
class FibonacciLevel:
    name: str
    ratio: float
    price: float
    zone: str


@dataclass(frozen=True)
class FibonacciAnalysis:
    direction: str
    swing_high: float
    swing_low: float
    range_size: float
    current_price: float
    current_zone: str
    levels: dict[str, float]
    premium: bool
    discount: bool
    entry_zone: tuple[float, float]
    extension_1272: float
    extension_1618: float


# ==========================================================
# FIBONACCI ENGINE
# ==========================================================

class FibonacciEngine:
    """
    Professional Fibonacci analysis engine.

    Aniqlaydi:
        - Fibonacci Retracement
        - Premium / Discount
        - 0.236
        - 0.382
        - 0.500
        - 0.618
        - 0.786
        - 1.000
        - 1.272 extension
        - 1.618 extension
        - Joriy narxning Fibonacci zonasi

    Muhim:
        - Swing High / Swing Low Market Structure'dan olinadi.
        - Current candle emas, berilgan yopilgan candle narxi
          signal hisoblash uchun ishlatilishi mumkin.
    """

    # ------------------------------------------------------
    # STANDART FIBONACCI DARAJALAR
    # ------------------------------------------------------

    RETRACEMENT_LEVELS = {
        "0.236": 0.236,
        "0.382": 0.382,
        "0.500": 0.500,
        "0.618": 0.618,
        "0.786": 0.786,
        "1.000": 1.000,
    }

    EXTENSION_LEVELS = {
        "1.272": 1.272,
        "1.618": 1.618,
    }

    # ------------------------------------------------------
    # ENTRY ZONALAR
    # ------------------------------------------------------

    ENTRY_ZONE_LOW = 0.618
    ENTRY_ZONE_HIGH = 0.786

    def __init__(
        self,
        tolerance: float = 0.001,
    ):
        if tolerance < 0:
            raise ValueError(
                "tolerance manfiy bo'lmasligi kerak."
            )

        self.tolerance = tolerance

    # ======================================================
    # VALIDATION
    # ======================================================

    @staticmethod
    def _validate_price(price: float, name: str) -> float:

        try:
            value = float(price)
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} son bo'lishi kerak."
            )

        if pd.isna(value):
            raise ValueError(
                f"{name} NaN bo'lishi mumkin emas."
            )

        if value <= 0:
            raise ValueError(
                f"{name} 0 dan katta bo'lishi kerak."
            )

        return value

    # ======================================================
    # FIBONACCI CALCULATION
    # ======================================================

    @staticmethod
    def calculate_levels(
        swing_high: float,
        swing_low: float,
        direction: str,
    ) -> dict[str, float]:

        swing_high = float(swing_high)
        swing_low = float(swing_low)

        if swing_high <= swing_low:
            raise ValueError(
                "Swing High Swing Low'dan katta bo'lishi kerak."
            )

        direction = direction.upper()

        if direction not in {
            "BULLISH",
            "BEARISH",
        }:
            raise ValueError(
                "direction faqat BULLISH yoki BEARISH bo'lishi kerak."
            )

        price_range = swing_high - swing_low

        levels: dict[str, float] = {}

        # --------------------------------------------------
        # BULLISH
        #
        # Swing Low → Swing High
        # --------------------------------------------------

        if direction == "BULLISH":

            for name, ratio in FibonacciEngine.RETRACEMENT_LEVELS.items():

                price = (
                    swing_high
                    - price_range * ratio
                )

                levels[name] = price

            levels["1.272"] = (
                swing_high
                + price_range * 0.272
            )

            levels["1.618"] = (
                swing_high
                + price_range * 0.618
            )

        # --------------------------------------------------
        # BEARISH
        #
        # Swing High → Swing Low
        # --------------------------------------------------

        else:

            for name, ratio in FibonacciEngine.RETRACEMENT_LEVELS.items():

                price = (
                    swing_low
                    + price_range * ratio
                )

                levels[name] = price

            levels["1.272"] = (
                swing_low
                - price_range * 0.272
            )

            levels["1.618"] = (
                swing_low
                - price_range * 0.618
            )

        return levels

    # ======================================================
    # CURRENT PRICE ZONE
    # ======================================================

    def detect_zone(
        self,
        current_price: float,
        levels: dict[str, float],
        direction: str,
    ) -> str:

        current_price = self._validate_price(
            current_price,
            "current_price",
        )

        direction = direction.upper()

        if direction not in {
            "BULLISH",
            "BEARISH",
        }:
            return "NOANIQ"

        f236 = levels["0.236"]
        f382 = levels["0.382"]
        f500 = levels["0.500"]
        f618 = levels["0.618"]
        f786 = levels["0.786"]
        f1000 = levels["1.000"]

        # --------------------------------------------------
        # BULLISH
        # --------------------------------------------------

        if direction == "BULLISH":

            if current_price >= f236:
                return "PREMIUM"

            if f382 <= current_price < f236:
                return "0.236-0.382"

            if f500 <= current_price < f382:
                return "0.382-0.500"

            if f618 <= current_price < f500:
                return "0.500-0.618"

            if f786 <= current_price < f618:
                return "GOLDEN_ZONE"

            if f1000 <= current_price < f786:
                return "0.786-1.000"

            return "DISCOUNT"

        # --------------------------------------------------
        # BEARISH
        # --------------------------------------------------

        if current_price <= f236:
            return "PREMIUM"

        if f236 < current_price <= f382:
            return "0.236-0.382"

        if f382 < current_price <= f500:
            return "0.382-0.500"

        if f500 < current_price <= f618:
            return "0.500-0.618"

        if f618 < current_price <= f786:
            return "GOLDEN_ZONE"

        if f786 < current_price <= f1000:
            return "0.786-1.000"

        return "DISCOUNT"

    # ======================================================
    # PREMIUM / DISCOUNT
    # ======================================================

    def premium_discount(
        self,
        current_price: float,
        swing_high: float,
        swing_low: float,
    ) -> tuple[bool, bool]:

        current_price = self._validate_price(
            current_price,
            "current_price",
        )

        swing_high = self._validate_price(
            swing_high,
            "swing_high",
        )

        swing_low = self._validate_price(
            swing_low,
            "swing_low",
        )

        if swing_high <= swing_low:
            raise ValueError(
                "Swing High Swing Low'dan katta bo'lishi kerak."
            )

        equilibrium = (
            swing_low
            + (swing_high - swing_low) * 0.5
        )

        premium = current_price > equilibrium
        discount = current_price < equilibrium

        return premium, discount

    # ======================================================
    # ENTRY ZONE
    # ======================================================

    @staticmethod
    def get_entry_zone(
        levels: dict[str, float],
        direction: str,
    ) -> tuple[float, float]:

        direction = direction.upper()

        f618 = levels["0.618"]
        f786 = levels["0.786"]

        if direction == "BULLISH":
            return (
                min(f618, f786),
                max(f618, f786),
            )

        if direction == "BEARISH":
            return (
                min(f618, f786),
                max(f618, f786),
            )

        raise ValueError(
            "direction faqat BULLISH yoki BEARISH."
        )

    # ======================================================
    # PRICE NEAR FIB LEVEL
    # ======================================================

    @staticmethod
    def nearest_level(
        current_price: float,
        levels: dict[str, float],
    ) -> tuple[str, float, float]:

        current_price = float(current_price)

        nearest_name = ""
        nearest_price = 0.0
        nearest_distance = float("inf")

        for name, price in levels.items():

            distance = abs(
                current_price - price
            )

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_name = name
                nearest_price = price

        return (
            nearest_name,
            nearest_price,
            nearest_distance,
        )

    # ======================================================
    # MAIN ANALYSIS
    # ======================================================

    def analyze(
        self,
        current_price: float,
        swing_high: float,
        swing_low: float,
        direction: str,
    ) -> dict:

        current_price = self._validate_price(
            current_price,
            "current_price",
        )

        swing_high = self._validate_price(
            swing_high,
            "swing_high",
        )

        swing_low = self._validate_price(
            swing_low,
            "swing_low",
        )

        direction = direction.upper()

        if swing_high <= swing_low:
            raise ValueError(
                "Swing High Swing Low'dan katta bo'lishi kerak."
            )

        if direction not in {
            "BULLISH",
            "BEARISH",
        }:
            raise ValueError(
                "direction faqat BULLISH yoki BEARISH."
            )

        price_range = (
            swing_high - swing_low
        )

        levels = self.calculate_levels(
            swing_high=swing_high,
            swing_low=swing_low,
            direction=direction,
        )

        zone = self.detect_zone(
            current_price=current_price,
            levels=levels,
            direction=direction,
        )

        premium, discount = self.premium_discount(
            current_price=current_price,
            swing_high=swing_high,
            swing_low=swing_low,
        )

        entry_zone = self.get_entry_zone(
            levels=levels,
            direction=direction,
        )

        nearest = self.nearest_level(
            current_price=current_price,
            levels=levels,
        )

        return {
            "direction": direction,

            "swing_high": swing_high,
            "swing_low": swing_low,
            "range_size": price_range,

            "current_price": current_price,

            "levels": levels,

            "zone": zone,

            "premium": premium,
            "discount": discount,

            "entry_zone": entry_zone,

            "nearest_level": nearest[0],
            "nearest_level_price": nearest[1],
            "nearest_level_distance": nearest[2],

            "extension_1272": levels["1.272"],
            "extension_1618": levels["1.618"],
        }

    # ======================================================
    # DATAFRAME HELPER
    # ======================================================

    @staticmethod
    def get_last_closed_price(
        df: pd.DataFrame,
    ) -> float:

        required = {
            "close",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"DataFrame'da kerakli ustun yo'q: {missing}"
            )

        if len(df) < 2:
            raise ValueError(
                "Oxirgi yopilgan candle olish uchun "
                "kamida 2 ta candle kerak."
            )

        # -1 = current/forming candle
        # -2 = last closed candle
        return float(
            df.iloc[-2]["close"]
        )

    # ======================================================
    # MARKET STRUCTURE RESULT BILAN ISHLASH
    # ======================================================

    def analyze_from_structure(
        self,
        current_price: float,
        structure_result: dict,
        direction: Optional[str] = None,
    ) -> dict:

        if not isinstance(
            structure_result,
            dict,
        ):
            raise ValueError(
                "structure_result dict bo'lishi kerak."
            )

        swing_highs = structure_result.get(
            "swing_highs",
            [],
        )

        swing_lows = structure_result.get(
            "swing_lows",
            [],
        )

        if not swing_highs:
            raise ValueError(
                "Market Structure'dan Swing High topilmadi."
            )

        if not swing_lows:
            raise ValueError(
                "Market Structure'dan Swing Low topilmadi."
            )

        last_high = swing_highs[-1]
        last_low = swing_lows[-1]

        swing_high = float(
            last_high.price
        )

        swing_low = float(
            last_low.price
        )

        if direction is None:
            structure_trend = structure_result.get(
                "trend",
                "ARALASH",
            )

            if structure_trend == "KO_TARILISH":
                direction = "BULLISH"

            elif structure_trend == "PASAYISH":
                direction = "BEARISH"

            else:
                direction = "BULLISH"

        return self.analyze(
            current_price=current_price,
            swing_high=swing_high,
            swing_low=swing_low,
            direction=direction,
        )


# ==========================================================
# DEFAULT INSTANCE
# ==========================================================

fibonacci_engine = FibonacciEngine()