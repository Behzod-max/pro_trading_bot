from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# ==========================================================
# DATA MODELS
# ==========================================================

@dataclass(frozen=True)
class SwingPoint:
    index: int
    time: object
    price: float
    kind: str


@dataclass(frozen=True)
class StructureEvent:
    index: int
    time: object
    price: float
    event: str
    direction: str
    broken_level: float


# ==========================================================
# MARKET STRUCTURE ENGINE
# ==========================================================

class MarketStructure:
    """
    Professional Market Structure Engine.

    Aniqlaydi:
        - Swing High
        - Swing Low
        - HH
        - HL
        - LH
        - LL
        - BOS
        - CHoCH
        - Trend

    Muhim:
        - Faqat tasdiqlangan swinglar ishlatiladi.
        - Break candle CLOSE orqali aniqlanadi.
        - ATR asosidagi minimal break filtri mavjud.
        - Bir xil swing level qayta-qayta signal bermaydi.
    """

    def __init__(
        self,
        left: int = 3,
        right: int = 3,
        atr_period: int = 14,
        break_atr_multiplier: float = 0.10,
    ):
        if left < 1 or right < 1:
            raise ValueError(
                "left va right kamida 1 bo'lishi kerak."
            )

        if atr_period < 1:
            raise ValueError(
                "atr_period kamida 1 bo'lishi kerak."
            )

        if break_atr_multiplier < 0:
            raise ValueError(
                "break_atr_multiplier manfiy bo'lmasligi kerak."
            )

        self.left = left
        self.right = right
        self.atr_period = atr_period
        self.break_atr_multiplier = break_atr_multiplier

    # ==========================================================
    # DATA VALIDATION
    # ==========================================================

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        required = {
            "time",
            "high",
            "low",
            "close",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Market structure uchun kerakli ustunlar mavjud emas: {missing}"
            )

        if df.empty:
            raise ValueError(
                "Market structure uchun DataFrame bo'sh."
            )

    # ==========================================================
    # ATR
    # ==========================================================

    def _calculate_atr(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

        previous_close = df["close"].shift(1)

        true_range = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - previous_close).abs(),
                (df["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        return true_range.ewm(
            alpha=1 / self.atr_period,
            adjust=False,
        ).mean()

    # ==========================================================
    # SWING HIGH / LOW
    # ==========================================================

    def find_swings(
        self,
        df: pd.DataFrame,
    ) -> list[SwingPoint]:

        self._validate_dataframe(df)

        swings: list[SwingPoint] = []

        if len(df) <= self.left + self.right:
            return swings

        for i in range(
            self.left,
            len(df) - self.right,
        ):

            current_high = float(
                df.iloc[i]["high"]
            )

            current_low = float(
                df.iloc[i]["low"]
            )

            left_highs = df.iloc[
                i - self.left:i
            ]["high"]

            right_highs = df.iloc[
                i + 1:i + self.right + 1
            ]["high"]

            left_lows = df.iloc[
                i - self.left:i
            ]["low"]

            right_lows = df.iloc[
                i + 1:i + self.right + 1
            ]["low"]

            # --------------------------------------------------
            # SWING HIGH
            # --------------------------------------------------

            is_swing_high = (
                current_high > left_highs.max()
                and current_high >= right_highs.max()
            )

            # --------------------------------------------------
            # SWING LOW
            # --------------------------------------------------

            is_swing_low = (
                current_low < left_lows.min()
                and current_low <= right_lows.min()
            )

            if is_swing_high:
                swings.append(
                    SwingPoint(
                        index=i,
                        time=df.iloc[i]["time"],
                        price=current_high,
                        kind="SWING_HIGH",
                    )
                )

            if is_swing_low:
                swings.append(
                    SwingPoint(
                        index=i,
                        time=df.iloc[i]["time"],
                        price=current_low,
                        kind="SWING_LOW",
                    )
                )

        swings.sort(
            key=lambda swing: swing.index
        )

        return swings

    # ==========================================================
    # SWING HIGH / LOW SEPARATION
    # ==========================================================

    @staticmethod
    def split_swings(
        swings: list[SwingPoint],
    ) -> tuple[list[SwingPoint], list[SwingPoint]]:

        swing_highs = [
            swing
            for swing in swings
            if swing.kind in {
                "SWING_HIGH",
                "HIGH",
                "HH",
                "LH",
            }
        ]

        swing_lows = [
            swing
            for swing in swings
            if swing.kind in {
                "SWING_LOW",
                "LOW",
                "HL",
                "LL",
            }
        ]

        return swing_highs, swing_lows

    # ==========================================================
    # HH / HL / LH / LL
    # ==========================================================

    def classify_swings(
        self,
        swings: list[SwingPoint],
    ) -> list[SwingPoint]:

        result: list[SwingPoint] = []

        previous_high: Optional[float] = None
        previous_low: Optional[float] = None

        for swing in swings:

            # --------------------------------------------------
            # HIGH
            # --------------------------------------------------

            if swing.kind == "SWING_HIGH":

                if previous_high is None:
                    label = "HIGH"

                elif swing.price > previous_high:
                    label = "HH"

                else:
                    label = "LH"

                previous_high = swing.price

            # --------------------------------------------------
            # LOW
            # --------------------------------------------------

            else:

                if previous_low is None:
                    label = "LOW"

                elif swing.price > previous_low:
                    label = "HL"

                else:
                    label = "LL"

                previous_low = swing.price

            result.append(
                SwingPoint(
                    index=swing.index,
                    time=swing.time,
                    price=swing.price,
                    kind=label,
                )
            )

        return result

    # ==========================================================
    # TREND
    # ==========================================================

    def detect_trend(
        self,
        classified: list[SwingPoint],
    ) -> str:

        highs = [
            swing
            for swing in classified
            if swing.kind in {"HH", "LH"}
        ]

        lows = [
            swing
            for swing in classified
            if swing.kind in {"HL", "LL"}
        ]

        if len(highs) < 2 or len(lows) < 2:
            return "NOANIQ"

        recent_highs = highs[-3:]
        recent_lows = lows[-3:]

        bullish_score = 0
        bearish_score = 0

        for swing in recent_highs:

            if swing.kind == "HH":
                bullish_score += 1

            elif swing.kind == "LH":
                bearish_score += 1

        for swing in recent_lows:

            if swing.kind == "HL":
                bullish_score += 1

            elif swing.kind == "LL":
                bearish_score += 1

        if bullish_score >= bearish_score + 2:
            return "KO_TARILISH"

        if bearish_score >= bullish_score + 2:
            return "PASAYISH"

        return "ARALASH"

    # ==========================================================
    # LAST CONFIRMED HIGH
    # ==========================================================

    @staticmethod
    def _last_confirmed_high(
        swings: list[SwingPoint],
    ) -> Optional[SwingPoint]:

        for swing in reversed(swings):

            if swing.kind in {
                "HIGH",
                "HH",
                "LH",
            }:
                return swing

        return None

    # ==========================================================
    # LAST CONFIRMED LOW
    # ==========================================================

    @staticmethod
    def _last_confirmed_low(
        swings: list[SwingPoint],
    ) -> Optional[SwingPoint]:

        for swing in reversed(swings):

            if swing.kind in {
                "LOW",
                "HL",
                "LL",
            }:
                return swing

        return None

    # ==========================================================
    # BOS / CHoCH
    # ==========================================================

    def detect_structure_events(
        self,
        df: pd.DataFrame,
        classified: list[SwingPoint],
    ) -> list[StructureEvent]:

        events: list[StructureEvent] = []

        if len(classified) < 4:
            return events

        atr = self._calculate_atr(df)

        current_trend = "ARALASH"

        broken_high_indices: set[int] = set()
        broken_low_indices: set[int] = set()

        for i in range(len(df)):

            # Faqat candle'dan oldingi tasdiqlangan swinglar.
            confirmed_swings = [
                swing
                for swing in classified
                if swing.index < i
            ]

            if len(confirmed_swings) < 2:
                continue

            last_high = self._last_confirmed_high(
                confirmed_swings
            )

            last_low = self._last_confirmed_low(
                confirmed_swings
            )

            close = float(
                df.iloc[i]["close"]
            )

            current_atr = float(
                atr.iloc[i]
            )

            if pd.isna(current_atr):
                continue

            minimum_break = (
                current_atr
                * self.break_atr_multiplier
            )

            # ==================================================
            # BULLISH BREAK
            # ==================================================

            if last_high is not None:

                high_break_distance = (
                    close - last_high.price
                )

                valid_high_break = (
                    high_break_distance
                    >= minimum_break
                )

                not_broken_before = (
                    last_high.index
                    not in broken_high_indices
                )

                if (
                    valid_high_break
                    and not_broken_before
                ):

                    if current_trend == "PASAYISH":
                        event_name = "CHoCH"
                    else:
                        event_name = "BOS"

                    events.append(
                        StructureEvent(
                            index=i,
                            time=df.iloc[i]["time"],
                            price=close,
                            event=event_name,
                            direction="BULLISH",
                            broken_level=last_high.price,
                        )
                    )

                    current_trend = "KO_TARILISH"

                    broken_high_indices.add(
                        last_high.index
                    )

            # ==================================================
            # BEARISH BREAK
            # ==================================================

            if last_low is not None:

                low_break_distance = (
                    last_low.price - close
                )

                valid_low_break = (
                    low_break_distance
                    >= minimum_break
                )

                not_broken_before = (
                    last_low.index
                    not in broken_low_indices
                )

                if (
                    valid_low_break
                    and not_broken_before
                ):

                    if current_trend == "KO_TARILISH":
                        event_name = "CHoCH"
                    else:
                        event_name = "BOS"

                    events.append(
                        StructureEvent(
                            index=i,
                            time=df.iloc[i]["time"],
                            price=close,
                            event=event_name,
                            direction="BEARISH",
                            broken_level=last_low.price,
                        )
                    )

                    current_trend = "PASAYISH"

                    broken_low_indices.add(
                        last_low.index
                    )

        return events

    # ==========================================================
    # ASOSIY TAHLIL
    # ==========================================================

    def analyze(
        self,
        df: pd.DataFrame,
    ) -> dict:

        self._validate_dataframe(df)

        minimum_rows = (
            self.left
            + self.right
            + 20
        )

        if len(df) < minimum_rows:
            raise ValueError(
                "Market structure uchun "
                "shamlar yetarli emas."
            )

        # ------------------------------------------------------
        # 1. RAW SWINGS
        # ------------------------------------------------------

        raw_swings = self.find_swings(df)

        # ------------------------------------------------------
        # 2. CLASSIFICATION
        # ------------------------------------------------------

        classified = self.classify_swings(
            raw_swings
        )

        # ------------------------------------------------------
        # 3. SWING HIGH / LOW
        # ------------------------------------------------------

        swing_highs, swing_lows = self.split_swings(
            classified
        )

        # ------------------------------------------------------
        # 4. TREND
        # ------------------------------------------------------

        trend = self.detect_trend(
            classified
        )

        # ------------------------------------------------------
        # 5. BOS / CHoCH
        # ------------------------------------------------------

        events = self.detect_structure_events(
            df,
            classified,
        )

        # ------------------------------------------------------
        # 6. LAST SWINGS
        # ------------------------------------------------------

        last_swing_high = (
            swing_highs[-1]
            if swing_highs
            else None
        )

        last_swing_low = (
            swing_lows[-1]
            if swing_lows
            else None
        )

        # ------------------------------------------------------
        # 7. RESULT
        # ------------------------------------------------------

        return {
            # Asosiy
            "swings": classified,

            # Alohida swinglar
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,

            # Raw swinglar ham saqlanadi
            "raw_swings": raw_swings,

            # Trend
            "trend": trend,

            # BOS / CHoCH
            "events": events,

            # Oxirgi swinglar
            "last_swing_high": last_swing_high,
            "last_swing_low": last_swing_low,

            # Statistikalar
            "swing_high_count": len(swing_highs),
            "swing_low_count": len(swing_lows),
            "event_count": len(events),
        }