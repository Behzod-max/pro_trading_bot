from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class TrendLine:
    kind: str
    start_index: int
    end_index: int
    start_price: float
    end_price: float
    slope: float
    touches: int = 0
    strength: str = "ZAIF"


@dataclass
class TrendLineEvent:
    index: int
    time: Any
    price: float
    event: str
    direction: str
    line_price: float


# ============================================================
# TRENDLINE ENGINE
# ============================================================

class TrendLineEngine:

    def __init__(
        self,
        atr_period: int = 14,
        touch_tolerance_atr: float = 0.20,
        breakout_atr: float = 0.20,
        min_touches: int = 2,
    ):
        self.atr_period = atr_period
        self.touch_tolerance_atr = touch_tolerance_atr
        self.breakout_atr = breakout_atr
        self.min_touches = min_touches

    # ========================================================
    # SAFE HELPERS
    # ========================================================

    @staticmethod
    def _get(obj: Any, key: str, default=None):
        """
        Object / dict / string formatlarini xavfsiz o'qish.
        """

        if obj is None:
            return default

        if isinstance(obj, dict):
            return obj.get(key, default)

        return getattr(obj, key, default)

    @staticmethod
    def _float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_kind(swing: Any) -> Optional[str]:
        """
        MarketStructure swing obyektining turini aniqlaydi.

        Qo'llab-quvvatlanadi:
            kind
            label
            type
            swing_type
        """

        if isinstance(swing, str):
            text = swing.upper().strip()

            if text in {
                "HH",
                "HL",
                "LH",
                "LL",
                "HIGH",
                "LOW",
            }:
                return text

            return None

        for key in (
            "kind",
            "label",
            "type",
            "swing_type",
        ):
            value = TrendLineEngine._get(
                swing,
                key,
                None,
            )

            if value is not None:

                text = str(value).upper().strip()

                if text in {
                    "HH",
                    "HL",
                    "LH",
                    "LL",
                    "HIGH",
                    "LOW",
                }:
                    return text

        return None

    @staticmethod
    def _get_price(swing: Any) -> Optional[float]:

        for key in (
            "price",
            "level",
            "value",
        ):

            value = TrendLineEngine._get(
                swing,
                key,
                None,
            )

            if value is not None:

                try:
                    return float(value)
                except (TypeError, ValueError):
                    pass

        return None

    @staticmethod
    def _get_index(swing: Any, fallback: int) -> int:

        for key in (
            "index",
            "bar_index",
            "candle_index",
            "i",
        ):

            value = TrendLineEngine._get(
                swing,
                key,
                None,
            )

            if value is not None:

                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass

        return fallback

    @staticmethod
    def _get_time(swing: Any, default=None):

        for key in (
            "time",
            "timestamp",
            "date",
        ):

            value = TrendLineEngine._get(
                swing,
                key,
                None,
            )

            if value is not None:
                return value

        return default

    # ========================================================
    # STRUCTURE EXTRACTION
    # ========================================================

    def _extract_swings(
        self,
        structure_result: Any,
    ) -> list:
        """
        MarketStructure natijasidan swinglarni chiqaradi.

        Dict:
            {"swings": [...]}

        yoki:
            {"swing_points": [...]}

        yoki to'g'ridan-to'g'ri list
        formatlarini qo'llab-quvvatlaydi.
        """

        if structure_result is None:
            return []

        if isinstance(structure_result, list):
            return structure_result

        if isinstance(structure_result, tuple):
            return list(structure_result)

        if isinstance(structure_result, dict):

            for key in (
                "swings",
                "swing_points",
                "structure_points",
            ):

                value = structure_result.get(
                    key,
                    None,
                )

                if value is not None:

                    if isinstance(value, dict):
                        return list(value.values())

                    if isinstance(value, (list, tuple)):
                        return list(value)

            return []

        # Object format
        for key in (
            "swings",
            "swing_points",
            "structure_points",
        ):

            value = getattr(
                structure_result,
                key,
                None,
            )

            if value is not None:

                if isinstance(value, dict):
                    return list(value.values())

                if isinstance(value, (list, tuple)):
                    return list(value)

        return []

    # ========================================================
    # ATR
    # ========================================================

    def _get_atr(
        self,
        df: pd.DataFrame,
        index: int,
    ) -> float:

        if df is None or df.empty:
            return 0.0

        if "atr_14" in df.columns:

            value = self._float(
                df.iloc[index]["atr_14"],
                0.0,
            )

            if value > 0:
                return value

        if "atr" in df.columns:

            value = self._float(
                df.iloc[index]["atr"],
                0.0,
            )

            if value > 0:
                return value

        # ATR mavjud bo'lmasa, fallback
        start = max(
            0,
            index - self.atr_period + 1,
        )

        sample = df.iloc[start:index + 1]

        if sample.empty:
            return 0.0

        ranges = (
            sample["high"]
            - sample["low"]
        )

        value = self._float(
            ranges.mean(),
            0.0,
        )

        return value

    # ========================================================
    # LINE PRICE
    # ========================================================

    @staticmethod
    def _line_price(
        start_index: int,
        start_price: float,
        slope: float,
        index: int,
    ) -> float:

        return (
            start_price
            + slope * (
                index - start_index
            )
        )

    # ========================================================
    # TOUCH COUNT
    # ========================================================

    def _count_touches(
        self,
        df: pd.DataFrame,
        start_index: int,
        end_index: int,
        start_price: float,
        slope: float,
    ) -> int:

        if df is None or df.empty:
            return 0

        touches = 0

        first = max(
            0,
            start_index,
        )

        last = min(
            len(df) - 1,
            len(df) - 1,
        )

        for i in range(first, last + 1):

            line_price = self._line_price(
                start_index,
                start_price,
                slope,
                i,
            )

            atr = self._get_atr(
                df,
                i,
            )

            if atr <= 0:
                atr = abs(
                    df.iloc[i]["high"]
                    - df.iloc[i]["low"]
                )

            if atr <= 0:
                continue

            tolerance = (
                atr
                * self.touch_tolerance_atr
            )

            high = self._float(
                df.iloc[i]["high"]
            )

            low = self._float(
                df.iloc[i]["low"]
            )

            # Candle linega tegsa
            if (
                low
                <= line_price + tolerance
                and
                high
                >= line_price - tolerance
            ):
                touches += 1

        return touches

    # ========================================================
    # STRENGTH
    # ========================================================

    @staticmethod
    def _strength(touches: int) -> str:

        if touches >= 5:
            return "JUDA KUCHLI"

        if touches >= 4:
            return "KUCHLI"

        if touches >= 3:
            return "O'RTACHA"

        if touches >= 2:
            return "ZAIF"

        return "YAROQSIZ"

    # ========================================================
    # BULLISH TRENDLINE
    # ========================================================

    def find_bullish(
        self,
        df: pd.DataFrame,
        structure_result: Any,
    ) -> list[TrendLine]:

        raw_swings = self._extract_swings(
            structure_result
        )

        lows = []

        for fallback_index, swing in enumerate(
            raw_swings
        ):

            kind = self._normalize_kind(
                swing
            )

            if kind not in {
                "HL",
                "LL",
                "LOW",
            }:
                continue

            price = self._get_price(
                swing
            )

            if price is None:
                continue

            index = self._get_index(
                swing,
                fallback_index,
            )

            lows.append(
                {
                    "index": index,
                    "price": price,
                    "kind": kind,
                    "time": self._get_time(
                        swing
                    ),
                }
            )

        # Vaqt/index bo'yicha
        lows.sort(
            key=lambda x: x["index"]
        )

        lines = []

        # Oxirgi swinglar orasidan trendline
        for i in range(
            max(0, len(lows) - 12),
            len(lows),
        ):

            for j in range(
                i + 1,
                len(lows),
            ):

                first = lows[i]
                second = lows[j]

                if (
                    second["index"]
                    <= first["index"]
                ):
                    continue

                price_difference = (
                    second["price"]
                    - first["price"]
                )

                index_difference = (
                    second["index"]
                    - first["index"]
                )

                if index_difference <= 0:
                    continue

                slope = (
                    price_difference
                    / index_difference
                )

                # Bullish trendline odatda
                # yuqoriga yo'nalgan bo'ladi.
                if slope <= 0:
                    continue

                touches = self._count_touches(
                    df,
                    first["index"],
                    second["index"],
                    first["price"],
                    slope,
                )

                if touches < self.min_touches:
                    continue

                strength = self._strength(
                    touches
                )

                lines.append(
                    TrendLine(
                        kind="BULLISH",
                        start_index=first["index"],
                        end_index=second["index"],
                        start_price=first["price"],
                        end_price=second["price"],
                        slope=slope,
                        touches=touches,
                        strength=strength,
                    )
                )

        return self._deduplicate_lines(
            lines
        )

    # ========================================================
    # BEARISH TRENDLINE
    # ========================================================

    def find_bearish(
        self,
        df: pd.DataFrame,
        structure_result: Any,
    ) -> list[TrendLine]:

        raw_swings = self._extract_swings(
            structure_result
        )

        highs = []

        for fallback_index, swing in enumerate(
            raw_swings
        ):

            kind = self._normalize_kind(
                swing
            )

            if kind not in {
                "LH",
                "HH",
                "HIGH",
            }:
                continue

            price = self._get_price(
                swing
            )

            if price is None:
                continue

            index = self._get_index(
                swing,
                fallback_index,
            )

            highs.append(
                {
                    "index": index,
                    "price": price,
                    "kind": kind,
                    "time": self._get_time(
                        swing
                    ),
                }
            )

        highs.sort(
            key=lambda x: x["index"]
        )

        lines = []

        for i in range(
            max(0, len(highs) - 12),
            len(highs),
        ):

            for j in range(
                i + 1,
                len(highs),
            ):

                first = highs[i]
                second = highs[j]

                if (
                    second["index"]
                    <= first["index"]
                ):
                    continue

                price_difference = (
                    second["price"]
                    - first["price"]
                )

                index_difference = (
                    second["index"]
                    - first["index"]
                )

                if index_difference <= 0:
                    continue

                slope = (
                    price_difference
                    / index_difference
                )

                # Bearish trendline pastga yo'naladi.
                if slope >= 0:
                    continue

                touches = self._count_touches(
                    df,
                    first["index"],
                    second["index"],
                    first["price"],
                    slope,
                )

                if touches < self.min_touches:
                    continue

                strength = self._strength(
                    touches
                )

                lines.append(
                    TrendLine(
                        kind="BEARISH",
                        start_index=first["index"],
                        end_index=second["index"],
                        start_price=first["price"],
                        end_price=second["price"],
                        slope=slope,
                        touches=touches,
                        strength=strength,
                    )
                )

        return self._deduplicate_lines(
            lines
        )

    # ========================================================
    # DEDUPLICATE
    # ========================================================

    @staticmethod
    def _deduplicate_lines(
        lines: list[TrendLine],
    ) -> list[TrendLine]:

        if not lines:
            return []

        result = []

        for line in lines:

            duplicate = False

            for existing in result:

                if (
                    line.kind
                    != existing.kind
                ):
                    continue

                price_diff = abs(
                    line.end_price
                    - existing.end_price
                )

                if price_diff < 0.50:

                    if (
                        line.touches
                        <= existing.touches
                    ):
                        duplicate = True
                        break

            if not duplicate:
                result.append(line)

        # Eng kuchli trendlinelar tepada
        result.sort(
            key=lambda x: (
                x.touches,
                x.strength,
            ),
            reverse=True,
        )

        return result[:10]

    # ========================================================
    # BREAKOUT
    # ========================================================

    def detect_breakouts(
        self,
        df: pd.DataFrame,
        lines: list[TrendLine],
    ) -> list[TrendLineEvent]:

        if df is None or df.empty:
            return []

        events = []

        for line in lines:

            start = max(
                line.end_index + 1,
                1,
            )

            for i in range(
                start,
                len(df),
            ):

                candle = df.iloc[i]
                previous = df.iloc[i - 1]

                line_price = self._line_price(
                    line.start_index,
                    line.start_price,
                    line.slope,
                    i,
                )

                previous_line = self._line_price(
                    line.start_index,
                    line.start_price,
                    line.slope,
                    i - 1,
                )

                atr = self._get_atr(
                    df,
                    i,
                )

                if atr <= 0:
                    continue

                threshold = (
                    atr
                    * self.breakout_atr
                )

                close = self._float(
                    candle["close"]
                )

                previous_close = self._float(
                    previous["close"]
                )

                # --------------------------------------------
                # BULLISH LINE BREAKDOWN
                # --------------------------------------------

                if line.kind == "BULLISH":

                    if (
                        previous_close
                        >= previous_line
                        and
                        close
                        < line_price - threshold
                    ):

                        events.append(
                            TrendLineEvent(
                                index=i,
                                time=candle["time"],
                                price=close,
                                event="BREAKOUT",
                                direction="BEARISH",
                                line_price=line_price,
                            )
                        )

                        break

                # --------------------------------------------
                # BEARISH LINE BREAKOUT
                # --------------------------------------------

                elif line.kind == "BEARISH":

                    if (
                        previous_close
                        <= previous_line
                        and
                        close
                        > line_price + threshold
                    ):

                        events.append(
                            TrendLineEvent(
                                index=i,
                                time=candle["time"],
                                price=close,
                                event="BREAKOUT",
                                direction="BULLISH",
                                line_price=line_price,
                            )
                        )

                        break

        return events

    # ========================================================
    # RETEST
    # ========================================================

    def detect_retests(
        self,
        df: pd.DataFrame,
        lines: list[TrendLine],
        breakouts: list[TrendLineEvent],
    ) -> list[TrendLineEvent]:

        if df is None or df.empty:
            return []

        retests = []

        for breakout in breakouts:

            breakout_index = breakout.index

            # Keyingi 20 candle ichida retest qidiramiz.
            end_index = min(
                len(df),
                breakout_index + 21,
            )

            for i in range(
                breakout_index + 1,
                end_index,
            ):

                candle = df.iloc[i]

                for line in lines:

                    line_price = self._line_price(
                        line.start_index,
                        line.start_price,
                        line.slope,
                        i,
                    )

                    atr = self._get_atr(
                        df,
                        i,
                    )

                    if atr <= 0:
                        continue

                    tolerance = (
                        atr
                        * self.touch_tolerance_atr
                    )

                    high = self._float(
                        candle["high"]
                    )

                    low = self._float(
                        candle["low"]
                    )

                    close = self._float(
                        candle["close"]
                    )

                    touched = (
                        low
                        <= line_price + tolerance
                        and
                        high
                        >= line_price - tolerance
                    )

                    if not touched:
                        continue

                    # ----------------------------------------
                    # BREAKOUT BULLISH
                    # ----------------------------------------

                    if (
                        breakout.direction
                        == "BULLISH"
                    ):

                        if close > line_price:

                            retests.append(
                                TrendLineEvent(
                                    index=i,
                                    time=candle["time"],
                                    price=close,
                                    event="RETEST",
                                    direction="BULLISH",
                                    line_price=line_price,
                                )
                            )

                            break

                    # ----------------------------------------
                    # BREAKOUT BEARISH
                    # ----------------------------------------

                    elif (
                        breakout.direction
                        == "BEARISH"
                    ):

                        if close < line_price:

                            retests.append(
                                TrendLineEvent(
                                    index=i,
                                    time=candle["time"],
                                    price=close,
                                    event="RETEST",
                                    direction="BEARISH",
                                    line_price=line_price,
                                )
                            )

                            break

        return retests

    # ========================================================
    # ANALYZE
    # ========================================================

    def analyze(
        self,
        df: pd.DataFrame,
        structure_result: Any,
    ) -> dict:

        if df is None or df.empty:

            return {
                "bullish_lines": [],
                "bearish_lines": [],
                "breakouts": [],
                "retests": [],
            }

        bullish = self.find_bullish(
            df,
            structure_result,
        )

        bearish = self.find_bearish(
            df,
            structure_result,
        )

        all_lines = (
            bullish
            + bearish
        )

        breakouts = self.detect_breakouts(
            df,
            all_lines,
        )

        retests = self.detect_retests(
            df,
            all_lines,
            breakouts,
        )

        return {
            "bullish_lines": bullish,
            "bearish_lines": bearish,
            "breakouts": breakouts,
            "retests": retests,
        }