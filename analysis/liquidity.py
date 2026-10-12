from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class LiquidityLevel:
    """
    Liquidity pool / level.
    """

    kind: str
    price: float
    index: int
    time: object
    strength: str
    touches: int


@dataclass
class LiquiditySweep:
    """
    Liquidity sweep hodisasi.
    """

    index: int
    time: object
    direction: str

    liquidity_price: float
    sweep_price: float
    close_price: float

    distance: float
    confirmed: bool

    strength: str


# ============================================================
# LIQUIDITY ENGINE
# ============================================================

class LiquidityEngine:

    def __init__(
        self,
        lookback: int = 100,
        swing_window: int = 3,
        equal_tolerance_atr: float = 0.20,
        sweep_min_atr: float = 0.05,
        sweep_max_atr: float = 0.80,
        confirmation_bars: int = 2,
    ):

        self.lookback = lookback
        self.swing_window = swing_window

        self.equal_tolerance_atr = (
            equal_tolerance_atr
        )

        self.sweep_min_atr = (
            sweep_min_atr
        )

        self.sweep_max_atr = (
            sweep_max_atr
        )

        self.confirmation_bars = (
            confirmation_bars
        )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _float(
        value,
        default: float = 0.0,
    ) -> float:

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):

            return default

    def _atr(
        self,
        df: pd.DataFrame,
        index: int,
    ) -> float:

        if "atr_14" in df.columns:

            value = self._float(
                df.iloc[index]["atr_14"]
            )

            if value > 0:
                return value

        if "atr" in df.columns:

            value = self._float(
                df.iloc[index]["atr"]
            )

            if value > 0:
                return value

        start = max(
            0,
            index - 13,
        )

        sample = df.iloc[
            start:index + 1
        ]

        if sample.empty:
            return 0.0

        return self._float(
            (
                sample["high"]
                - sample["low"]
            ).mean()
        )

    # ========================================================
    # SWING HIGHS
    # ========================================================

    def find_swing_highs(
        self,
        df: pd.DataFrame,
    ) -> list[dict]:

        if df is None or df.empty:
            return []

        window = self.swing_window

        highs = []

        start = window
        end = len(df) - window

        for i in range(
            start,
            end,
        ):

            current_high = self._float(
                df.iloc[i]["high"]
            )

            left = df.iloc[
                i - window:i
            ]["high"]

            right = df.iloc[
                i + 1:i + window + 1
            ]["high"]

            if (
                current_high >= left.max()
                and
                current_high >= right.max()
            ):

                highs.append(
                    {
                        "index": i,
                        "price": current_high,
                        "time": df.iloc[i]["time"],
                    }
                )

        return highs

    # ========================================================
    # SWING LOWS
    # ========================================================

    def find_swing_lows(
        self,
        df: pd.DataFrame,
    ) -> list[dict]:

        if df is None or df.empty:
            return []

        window = self.swing_window

        lows = []

        start = window
        end = len(df) - window

        for i in range(
            start,
            end,
        ):

            current_low = self._float(
                df.iloc[i]["low"]
            )

            left = df.iloc[
                i - window:i
            ]["low"]

            right = df.iloc[
                i + 1:i + window + 1
            ]["low"]

            if (
                current_low <= left.min()
                and
                current_low <= right.min()
            ):

                lows.append(
                    {
                        "index": i,
                        "price": current_low,
                        "time": df.iloc[i]["time"],
                    }
                )

        return lows

    # ========================================================
    # EQUAL HIGHS
    # ========================================================

    def find_equal_highs(
        self,
        df: pd.DataFrame,
        swing_highs: list[dict],
    ) -> list[LiquidityLevel]:

        result = []

        if not swing_highs:
            return result

        start_index = max(
            0,
            len(df) - self.lookback,
        )

        recent = [
            x
            for x in swing_highs
            if x["index"] >= start_index
        ]

        used = set()

        for i in range(
            len(recent)
        ):

            if i in used:
                continue

            first = recent[i]

            cluster = [
                first
            ]

            for j in range(
                i + 1,
                len(recent),
            ):

                if j in used:
                    continue

                second = recent[j]

                avg_price = (
                    first["price"]
                    + second["price"]
                ) / 2

                atr = self._atr(
                    df,
                    second["index"],
                )

                if atr <= 0:
                    continue

                tolerance = (
                    atr
                    * self.equal_tolerance_atr
                )

                if abs(
                    first["price"]
                    - second["price"]
                ) <= tolerance:

                    cluster.append(
                        second
                    )

                    used.add(j)

            if len(cluster) < 2:
                continue

            prices = [
                x["price"]
                for x in cluster
            ]

            level_price = sum(
                prices
            ) / len(prices)

            latest = cluster[-1]

            touches = len(cluster)

            if touches >= 4:
                strength = "JUDA KUCHLI"

            elif touches == 3:
                strength = "KUCHLI"

            else:
                strength = "O'RTACHA"

            result.append(
                LiquidityLevel(
                    kind="EQUAL_HIGH",
                    price=level_price,
                    index=latest["index"],
                    time=latest["time"],
                    strength=strength,
                    touches=touches,
                )
            )

        return result

    # ========================================================
    # EQUAL LOWS
    # ========================================================

    def find_equal_lows(
        self,
        df: pd.DataFrame,
        swing_lows: list[dict],
    ) -> list[LiquidityLevel]:

        result = []

        if not swing_lows:
            return result

        start_index = max(
            0,
            len(df) - self.lookback,
        )

        recent = [
            x
            for x in swing_lows
            if x["index"] >= start_index
        ]

        used = set()

        for i in range(
            len(recent)
        ):

            if i in used:
                continue

            first = recent[i]

            cluster = [
                first
            ]

            for j in range(
                i + 1,
                len(recent),
            ):

                if j in used:
                    continue

                second = recent[j]

                atr = self._atr(
                    df,
                    second["index"],
                )

                if atr <= 0:
                    continue

                tolerance = (
                    atr
                    * self.equal_tolerance_atr
                )

                if abs(
                    first["price"]
                    - second["price"]
                ) <= tolerance:

                    cluster.append(
                        second
                    )

                    used.add(j)

            if len(cluster) < 2:
                continue

            prices = [
                x["price"]
                for x in cluster
            ]

            level_price = sum(
                prices
            ) / len(prices)

            latest = cluster[-1]

            touches = len(cluster)

            if touches >= 4:
                strength = "JUDA KUCHLI"

            elif touches == 3:
                strength = "KUCHLI"

            else:
                strength = "O'RTACHA"

            result.append(
                LiquidityLevel(
                    kind="EQUAL_LOW",
                    price=level_price,
                    index=latest["index"],
                    time=latest["time"],
                    strength=strength,
                    touches=touches,
                )
            )

        return result

    # ========================================================
    # PREVIOUS HIGH / LOW
    # ========================================================

    def previous_levels(
        self,
        df: pd.DataFrame,
    ) -> list[LiquidityLevel]:

        if df is None or len(df) < 3:
            return []

        current_index = len(df) - 1

        levels = []

        # Previous candle
        prev = df.iloc[
            current_index - 1
        ]

        levels.append(
            LiquidityLevel(
                kind="PREVIOUS_HIGH",
                price=self._float(
                    prev["high"]
                ),
                index=current_index - 1,
                time=prev["time"],
                strength="O'RTACHA",
                touches=1,
            )
        )

        levels.append(
            LiquidityLevel(
                kind="PREVIOUS_LOW",
                price=self._float(
                    prev["low"]
                ),
                index=current_index - 1,
                time=prev["time"],
                strength="O'RTACHA",
                touches=1,
            )
        )

        # Previous 5 candles high / low
        start = max(
            0,
            current_index - 6,
        )

        recent = df.iloc[
            start:current_index
        ]

        if not recent.empty:

            levels.append(
                LiquidityLevel(
                    kind="RECENT_HIGH",
                    price=self._float(
                        recent["high"].max()
                    ),
                    index=int(
                        recent["high"].idxmax()
                    ),
                    time=df.loc[
                        recent["high"].idxmax(),
                        "time",
                    ],
                    strength="KUCHLI",
                    touches=1,
                )
            )

            levels.append(
                LiquidityLevel(
                    kind="RECENT_LOW",
                    price=self._float(
                        recent["low"].min()
                    ),
                    index=int(
                        recent["low"].idxmin()
                    ),
                    time=df.loc[
                        recent["low"].idxmin(),
                        "time",
                    ],
                    strength="KUCHLI",
                    touches=1,
                )
            )

        return levels

    # ========================================================
    # LIQUIDITY SWEEP
    # ========================================================

    def detect_sweeps(
        self,
        df: pd.DataFrame,
        levels: list[LiquidityLevel],
    ) -> list[LiquiditySweep]:

        if df is None or df.empty:
            return []

        sweeps = []

        # Faqat oxirgi lookback
        # candlelarni tekshiramiz.
        start_index = max(
            1,
            len(df) - self.lookback,
        )

        for i in range(
            start_index,
            len(df),
        ):

            candle = df.iloc[i]

            high = self._float(
                candle["high"]
            )

            low = self._float(
                candle["low"]
            )

            close = self._float(
                candle["close"]
            )

            atr = self._atr(
                df,
                i,
            )

            if atr <= 0:
                continue

            min_distance = (
                atr
                * self.sweep_min_atr
            )

            max_distance = (
                atr
                * self.sweep_max_atr
            )

            for level in levels:

                liquidity_price = (
                    level.price
                )

                # ==========================================
                # SELL-SIDE LIQUIDITY SWEEP
                # ==========================================

                if level.kind in {
                    "EQUAL_LOW",
                    "PREVIOUS_LOW",
                    "RECENT_LOW",
                }:

                    sweep_distance = (
                        liquidity_price
                        - low
                    )

                    # Low liquidity ostiga o'tdi
                    if (
                        sweep_distance
                        >= min_distance
                        and
                        sweep_distance
                        <= max_distance
                    ):

                        # Candle yana liquidity
                        # ustida yopilgan bo'lishi kerak.
                        if close > liquidity_price:

                            confirmed = (
                                self._confirm_bullish_reclaim(
                                    df,
                                    i,
                                    liquidity_price,
                                )
                            )

                            if confirmed:

                                sweeps.append(
                                    LiquiditySweep(
                                        index=i,
                                        time=candle["time"],
                                        direction="BULLISH",
                                        liquidity_price=liquidity_price,
                                        sweep_price=low,
                                        close_price=close,
                                        distance=sweep_distance,
                                        confirmed=True,
                                        strength=level.strength,
                                    )
                                )

                # ==========================================
                # BUY-SIDE LIQUIDITY SWEEP
                # ==========================================

                if level.kind in {
                    "EQUAL_HIGH",
                    "PREVIOUS_HIGH",
                    "RECENT_HIGH",
                }:

                    sweep_distance = (
                        high
                        - liquidity_price
                    )

                    # High liquidity ustiga o'tdi
                    if (
                        sweep_distance
                        >= min_distance
                        and
                        sweep_distance
                        <= max_distance
                    ):

                        # Candle yana liquidity
                        # ostida yopilishi kerak.
                        if close < liquidity_price:

                            confirmed = (
                                self._confirm_bearish_rejection(
                                    df,
                                    i,
                                    liquidity_price,
                                )
                            )

                            if confirmed:

                                sweeps.append(
                                    LiquiditySweep(
                                        index=i,
                                        time=candle["time"],
                                        direction="BEARISH",
                                        liquidity_price=liquidity_price,
                                        sweep_price=high,
                                        close_price=close,
                                        distance=sweep_distance,
                                        confirmed=True,
                                        strength=level.strength,
                                    )
                                )

        return self._deduplicate_sweeps(
            sweeps
        )

    # ========================================================
    # BULLISH RECLAIM
    # ========================================================

    def _confirm_bullish_reclaim(
        self,
        df: pd.DataFrame,
        index: int,
        level: float,
    ) -> bool:

        if index >= len(df) - 1:
            return False

        end = min(
            len(df),
            index
            + self.confirmation_bars
            + 1,
        )

        for i in range(
            index + 1,
            end,
        ):

            close = self._float(
                df.iloc[i]["close"]
            )

            if close > level:
                return True

        return False

    # ========================================================
    # BEARISH REJECTION
    # ========================================================

    def _confirm_bearish_rejection(
        self,
        df: pd.DataFrame,
        index: int,
        level: float,
    ) -> bool:

        if index >= len(df) - 1:
            return False

        end = min(
            len(df),
            index
            + self.confirmation_bars
            + 1,
        )

        for i in range(
            index + 1,
            end,
        ):

            close = self._float(
                df.iloc[i]["close"]
            )

            if close < level:
                return True

        return False

    # ========================================================
    # DEDUPLICATE SWEEPS
    # ========================================================

    @staticmethod
    def _deduplicate_sweeps(
        sweeps: list[LiquiditySweep],
    ) -> list[LiquiditySweep]:

        if not sweeps:
            return []

        result = []

        seen = set()

        for sweep in sweeps:

            key = (
                sweep.index,
                sweep.direction,
                round(
                    sweep.liquidity_price,
                    2,
                ),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(sweep)

        return result

    # ========================================================
    # ANALYZE
    # ========================================================

    def analyze(
        self,
        df: pd.DataFrame,
    ) -> dict:

        if df is None or df.empty:

            return {
                "swing_highs": [],
                "swing_lows": [],
                "equal_highs": [],
                "equal_lows": [],
                "previous_levels": [],
                "liquidity_levels": [],
                "sweeps": [],
                "latest_sweep": None,
            }

        swing_highs = self.find_swing_highs(
            df
        )

        swing_lows = self.find_swing_lows(
            df
        )

        equal_highs = self.find_equal_highs(
            df,
            swing_highs,
        )

        equal_lows = self.find_equal_lows(
            df,
            swing_lows,
        )

        previous = self.previous_levels(
            df
        )

        liquidity_levels = (
            equal_highs
            + equal_lows
            + previous
        )

        sweeps = self.detect_sweeps(
            df,
            liquidity_levels,
        )

        sweeps.sort(
            key=lambda x: x.index
        )

        latest_sweep = (
            sweeps[-1]
            if sweeps
            else None
        )

        return {
            "swing_highs": swing_highs,
            "swing_lows": swing_lows,
            "equal_highs": equal_highs,
            "equal_lows": equal_lows,
            "previous_levels": previous,
            "liquidity_levels": liquidity_levels,
            "sweeps": sweeps,
            "latest_sweep": latest_sweep,
        }