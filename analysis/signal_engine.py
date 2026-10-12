from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math


# ============================================================
# SIGNAL RESULT
# ============================================================

@dataclass
class SignalResult:
    direction: str = "NEUTRAL"
    score: float = 0.0
    status: str = "NO_SIGNAL"

    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None

    risk_reward_1: Optional[float] = None
    risk_reward_2: Optional[float] = None

    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    components: Dict[str, float] = field(default_factory=dict)


# ============================================================
# SIGNAL ENGINE
# ============================================================

class SignalEngine:
    """
    Professional weighted signal engine.

    Weights:
        MTF                 25
        Market Structure    20
        Liquidity           15
        Fibonacci           15
        Trendline           10
        Candle Confirmation 15
        --------------------------------
        TOTAL              100

    Important:
        - Only closed candle is used for signal calculations.
        - pandas DataFrame/Series is never evaluated with `or`/`and`.
        - Main.py can pass M5 results wrapped as {"M5": result}.
    """

    WEIGHTS = {
        "mtf": 25.0,
        "structure": 20.0,
        "liquidity": 15.0,
        "fibonacci": 15.0,
        "trendline": 10.0,
        "candle": 15.0,
    }

    EARLY_THRESHOLD = 60.0
    CONFIRMED_THRESHOLD = 75.0
    HIGH_CONFIDENCE_THRESHOLD = 85.0

    MIN_RR_1 = 1.50
    MIN_RR_2 = 2.00

    # --------------------------------------------------------
    # SAFE HELPERS
    # --------------------------------------------------------

    @staticmethod
    def _is_number(value: Any) -> bool:
        if value is None or isinstance(value, bool):
            return False

        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _float(value: Any, default: Optional[float] = None):
        if not SignalEngine._is_number(value):
            return default

        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip().upper()
        return str(value).strip().upper()

    @staticmethod
    def _get(obj: Any, key: str, default=None):
        if obj is None:
            return default

        if isinstance(obj, dict):
            return obj.get(key, default)

        try:
            return getattr(obj, key, default)
        except Exception:
            return default

    @staticmethod
    def _first(obj: Any, keys: Tuple[str, ...], default=None):
        """
        Returns first value which is not None.

        Deliberately does NOT use:
            value or default

        because value may be a pandas DataFrame/Series.
        """
        for key in keys:
            value = SignalEngine._get(obj, key, None)
            if value is not None:
                return value
        return default

    @staticmethod
    def _list(value: Any) -> list:
        if value is None:
            return []

        if isinstance(value, (list, tuple)):
            return list(value)

        return []

    @staticmethod
    def direction(value: Any) -> str:
        """
        Normalize direction without ever evaluating DataFrame/Series
        as a boolean.
        """

        if value is None:
            return "NEUTRAL"

        # String
        if isinstance(value, str):
            text = value.strip().upper()

        # Numeric
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                number = float(value)
            except (TypeError, ValueError):
                return "NEUTRAL"

            if number > 0:
                return "BUY"
            if number < 0:
                return "SELL"
            return "NEUTRAL"

        # pandas / objects
        else:
            try:
                import pandas as pd
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    return "NEUTRAL"
            except Exception:
                pass

            text = ""

            for attr in ("direction", "trend", "bias", "side"):
                try:
                    candidate = getattr(value, attr, None)
                except Exception:
                    candidate = None

                if isinstance(candidate, str):
                    text = candidate.strip().upper()
                    break

            if not text:
                return "NEUTRAL"

        aliases = {
            "BUY": "BUY",
            "LONG": "BUY",
            "BULL": "BUY",
            "BULLISH": "BUY",
            "UP": "BUY",
            "UPTREND": "BUY",
            "O'SISH": "BUY",
            "OSISH": "BUY",
            "KO_TARILISH": "BUY",
            "KOTARILISH": "BUY",

            "SELL": "SELL",
            "SHORT": "SELL",
            "BEAR": "SELL",
            "BEARISH": "SELL",
            "DOWN": "SELL",
            "DOWNTREND": "SELL",
            "PASAYISH": "SELL",

            "NEUTRAL": "NEUTRAL",
            "MIXED": "NEUTRAL",
            "ARALASH": "NEUTRAL",
            "NONE": "NEUTRAL",
            "UNKNOWN": "NEUTRAL",
            "NO_SIGNAL": "NEUTRAL",
        }

        return aliases.get(text, "NEUTRAL")

    # --------------------------------------------------------
    # RESULT EXTRACTION
    # --------------------------------------------------------

    def _extract_m5(self, value: Any):
        """
        Main.py stores component results as:
            {"M5": result}

        This extracts M5 safely.
        """
        if isinstance(value, dict):
            result = value.get("M5")

            if result is not None:
                return result

            result = value.get("m5")

            if result is not None:
                return result

            return value

        return value

    # ========================================================
    # MTF — 25
    # ========================================================

    def score_mtf(
        self,
        mtf_trends: Any,
    ) -> Tuple[float, str, List[str]]:

        reasons: List[str] = []

        if not isinstance(mtf_trends, dict):
            return 0.0, "NEUTRAL", reasons

        timeframes = [
            "M1",
            "M5",
            "M15",
            "M30",
            "H1",
            "H4",
            "D1",
        ]

        # Higher timeframes have more weight.
        tf_weights = {
            "M1": 1.0,
            "M5": 2.0,
            "M15": 3.0,
            "M30": 3.0,
            "H1": 4.0,
            "H4": 5.0,
            "D1": 3.0,
        }

        buy = 0.0
        sell = 0.0
        total = 0.0

        for tf in timeframes:
            value = mtf_trends.get(tf)

            if value is None:
                value = mtf_trends.get(tf.lower())

            if value is None:
                continue

            direction = self.direction(value)
            weight = tf_weights[tf]

            if direction == "BUY":
                buy += weight
                total += weight

            elif direction == "SELL":
                sell += weight
                total += weight

        if total <= 0:
            reasons.append("MTF ma'lumotlari yetarli emas.")
            return 0.0, "NEUTRAL", reasons

        if buy > sell:
            direction = "BUY"
            dominant = buy
        elif sell > buy:
            direction = "SELL"
            dominant = sell
        else:
            direction = "NEUTRAL"
            dominant = 0.0

        # Dominance ratio converted to 25 points.
        score = (dominant / total) * self.WEIGHTS["mtf"]

        reasons.append(
            f"MTF: BUY={buy:.1f}, SELL={sell:.1f}."
        )

        if direction != "NEUTRAL":
            dominance = dominant / total

            if dominance >= 0.70:
                reasons.append(
                    "Ko'p timeframe bir yo'nalishga kuchli moslashgan."
                )
            elif dominance >= 0.55:
                reasons.append(
                    "MTF yo'nalishida o'rtacha ustunlik mavjud."
                )
            else:
                reasons.append(
                    "MTF yo'nalishi aralash, ustunlik kuchsiz."
                )

        return min(score, 25.0), direction, reasons

    # ========================================================
    # MARKET STRUCTURE — 20
    # ========================================================

    def score_structure(
        self,
        structure_result: Any,
    ) -> Tuple[float, str, List[str]]:

        reasons: List[str] = []

        if structure_result is None:
            return 0.0, "NEUTRAL", reasons

        trend_value = self._first(
            structure_result,
            ("trend", "direction", "bias"),
            None,
        )

        direction = self.direction(trend_value)

        score = 0.0

        if direction in ("BUY", "SELL"):
            score += 8.0
            reasons.append(
                f"Structure bias: {direction}."
            )
        else:
            reasons.append(
                "Structure bias: NEUTRAL."
            )

        events = self._first(
            structure_result,
            ("events", "structure_events"),
            [],
        )

        events = self._list(events)

        latest = events[-1] if events else None

        event_name = self._first(
            latest,
            ("kind", "label", "type", "event", "name"),
            latest if isinstance(latest, str) else "",
        )

        event_name = self._text(event_name)

        event_direction = self.direction(
            self._first(
                latest,
                ("direction", "bias", "side"),
                None,
            )
        )

        if event_name == "BOS":
            score += 7.0
            reasons.append("Market Structure: BOS aniqlangan.")

            if event_direction != "NEUTRAL" and event_direction != direction:
                score -= 2.0
                reasons.append(
                    "BOS yo'nalishi structure bias bilan qarama-qarshi."
                )

        elif event_name == "CHOCH":
            score += 6.0
            reasons.append("Market Structure: CHoCH aniqlangan.")

        elif event_name:
            score += 2.0
            reasons.append(
                f"Market Structure: oxirgi event = {event_name}."
            )

        else:
            reasons.append(
                "Market Structure: oxirgi event topilmadi."
            )

        swing_highs = self._list(
            self._first(
                structure_result,
                ("swing_highs",),
                [],
            )
        )

        swing_lows = self._list(
            self._first(
                structure_result,
                ("swing_lows",),
                [],
            )
        )

        if swing_highs:
            score += 2.5

        if swing_lows:
            score += 2.5

        return min(max(score, 0.0), 20.0), direction, reasons

    # ========================================================
    # LIQUIDITY — 15
    # ========================================================

    def score_liquidity(
        self,
        liquidity_result: Any,
    ) -> Tuple[float, str, List[str]]:

        reasons: List[str] = []

        liquidity_result = self._extract_m5(liquidity_result)

        if liquidity_result is None:
            return 0.0, "NEUTRAL", reasons

        sweeps = self._list(
            self._first(
                liquidity_result,
                ("sweeps", "liquidity_sweeps"),
                [],
            )
        )

        latest_sweep = self._first(
            liquidity_result,
            ("latest_sweep",),
            None,
        )

        # If engine did not provide latest_sweep, use latest sweep.
        if latest_sweep is None and sweeps:
            latest_sweep = sweeps[-1]

        if latest_sweep is None:
            reasons.append(
                "Liquidity: sweep topilmadi."
            )
            return 0.0, "NEUTRAL", reasons

        direction = self.direction(
            self._first(
                latest_sweep,
                ("direction", "bias", "side"),
                None,
            )
        )

        score = 7.0

        confirmed = self._first(
            latest_sweep,
            ("confirmed", "is_confirmed"),
            False,
        )

        if confirmed is True:
            score += 4.0
            reasons.append(
                "Liquidity sweep tasdiqlangan."
            )
        else:
            reasons.append(
                "Liquidity sweep mavjud, lekin tasdiqlanmagan."
            )

        strength = self._float(
            self._first(
                latest_sweep,
                ("strength",),
                None,
            ),
            0.0,
        )

        if strength >= 70:
            score += 4.0
        elif strength >= 40:
            score += 2.0
        else:
            score += 1.0

        score = min(score, 15.0)

        if direction == "BUY":
            reasons.append(
                "Liquidity sweep BUY yo'nalishini ko'rsatmoqda."
            )
        elif direction == "SELL":
            reasons.append(
                "Liquidity sweep SELL yo'nalishini ko'rsatmoqda."
            )
        else:
            reasons.append(
                "Liquidity sweep yo'nalishi noaniq."
            )

        return score, direction, reasons

    # ========================================================
    # FIBONACCI — 15
    # ========================================================

    def score_fibonacci(
        self,
        fibonacci_result: Any,
    ) -> Tuple[float, str, List[str]]:

        reasons: List[str] = []

        fibonacci_result = self._extract_m5(fibonacci_result)

        if fibonacci_result is None:
            return 0.0, "NEUTRAL", reasons

        direction = self.direction(
            self._first(
                fibonacci_result,
                ("direction", "trend", "bias"),
                None,
            )
        )

        score = 0.0

        if direction in ("BUY", "SELL"):
            score += 5.0

        zone = self._text(
            self._first(
                fibonacci_result,
                ("zone",),
                "",
            )
        )

        nearest_level = self._first(
            fibonacci_result,
            ("nearest_level",),
            None,
        )

        nearest_price = self._float(
            self._first(
                fibonacci_result,
                ("nearest_level_price",),
                None,
            ),
            None,
        )

        entry_zone = self._first(
            fibonacci_result,
            ("entry_zone",),
            None,
        )

        # Golden Zone
        if zone == "GOLDEN_ZONE":
            score += 6.0
            reasons.append(
                "Fibonacci: GOLDEN ZONE aniqlangan."
            )

        elif zone:
            score += 2.0
            reasons.append(
                f"Fibonacci zona: {zone}."
            )

        # Nearest level
        if nearest_level is not None and nearest_price is not None:
            score += 2.0
            reasons.append(
                f"Fibonacci yaqin daraja: "
                f"{nearest_level} @ {nearest_price:.2f}."
            )

        # Valid entry zone
        if (
            isinstance(entry_zone, (list, tuple))
            and len(entry_zone) >= 2
        ):
            score += 2.0
            reasons.append(
                "Fibonacci entry zona mavjud."
            )

        score = min(score, 15.0)

        if direction == "BUY":
            reasons.append(
                "Fibonacci BUY setupni qo'llab-quvvatlamoqda."
            )
        elif direction == "SELL":
            reasons.append(
                "Fibonacci SELL setupni qo'llab-quvvatlamoqda."
            )

        return score, direction, reasons

    # ========================================================
    # TRENDLINE — 10
    # ========================================================

    def score_trendline(
        self,
        trendline_result: Any,
    ) -> Tuple[float, str, List[str]]:

        reasons: List[str] = []

        trendline_result = self._extract_m5(trendline_result)

        if trendline_result is None:
            return 0.0, "NEUTRAL", reasons

        breakouts = self._list(
            self._first(
                trendline_result,
                ("breakouts", "breakout_events"),
                [],
            )
        )

        retests = self._list(
            self._first(
                trendline_result,
                ("retests", "retest_events"),
                [],
            )
        )

        events = self._list(
            self._first(
                trendline_result,
                ("events", "trendline_events"),
                [],
            )
        )

        # ----------------------------------------------------
        # Direction from latest event
        # ----------------------------------------------------

        direction = "NEUTRAL"

        latest_event = events[-1] if events else None

        if latest_event is not None:

            direction = self.direction(
                self._first(
                    latest_event,
                    ("direction", "bias", "side"),
                    None,
                )
            )

            event_name = self._text(
                self._first(
                    latest_event,
                    ("event", "type", "name", "kind"),
                    "",
                )
            )

            if event_name:
                reasons.append(
                    f"Trendline event: {event_name}."
                )

        # ----------------------------------------------------
        # If no event direction, inspect breakout
        # ----------------------------------------------------

        if direction == "NEUTRAL" and breakouts:

            for breakout in reversed(breakouts):

                candidate = self.direction(
                    self._first(
                        breakout,
                        ("direction", "bias", "side"),
                        None,
                    )
                )

                if candidate != "NEUTRAL":
                    direction = candidate
                    break

        # ----------------------------------------------------
        # Scoring
        # ----------------------------------------------------

        score = 0.0

        if breakouts:
            score += 5.0
            reasons.append(
                f"Trendline breakout: {len(breakouts)}."
            )

        if retests:
            score += 3.0
            reasons.append(
                f"Trendline retest: {len(retests)}."
            )

        # Event exists = additional evidence.
        if latest_event is not None:
            score += 2.0

        score = min(score, 10.0)

        if direction == "BUY":
            reasons.append(
                "Trendline BUY yo'nalishini qo'llab-quvvatlamoqda."
            )
        elif direction == "SELL":
            reasons.append(
                "Trendline SELL yo'nalishini qo'llab-quvvatlamoqda."
            )

        return score, direction, reasons

    # ========================================================
    # CANDLE — 15
    # ========================================================

    def score_candle(
        self,
        candle_result: Any,
    ) -> Tuple[float, str, List[str]]:

        reasons: List[str] = []

        candle_result = self._extract_m5(candle_result)

        if candle_result is None:
            return 0.0, "NEUTRAL", reasons

        direction = self.direction(
            self._first(
                candle_result,
                ("direction", "bias", "side"),
                None,
            )
        )

        raw_score = self._float(
            self._first(
                candle_result,
                ("score", "confidence"),
                0.0,
            ),
            0.0,
        )

        raw_score = max(0.0, min(raw_score, 100.0))

        confirmed = self._first(
            candle_result,
            ("confirmed", "is_confirmed"),
            False,
        )

        pattern = self._first(
            candle_result,
            ("pattern", "name"),
            "NO_PATTERN",
        )

        # Existing CandleConfirmation score is 0-100.
        score = (raw_score / 100.0) * 10.0

        if confirmed is True:
            score += 5.0

        if pattern and self._text(pattern) != "NO_PATTERN":
            reasons.append(
                f"Candle pattern: {pattern}."
            )
        else:
            reasons.append(
                "Candle pattern topilmadi."
            )

        if confirmed is True:
            reasons.append(
                "Candle confirmation tasdiqlangan."
            )
        else:
            reasons.append(
                "Candle confirmation hali tasdiqlanmagan."
            )

        return min(score, 15.0), direction, reasons

    # ========================================================
    # ENTRY / SL / TP
    # ========================================================

    def calculate_levels(
        self,
        row: Any,
        direction: str,
        structure_result: Any = None,
        fibonacci_result: Any = None,
    ):
        """
        Conservative level calculation.

        BUY:
            SL below latest swing low / ATR fallback
            TP1 = 1.5R
            TP2 = 2.5R

        SELL:
            SL above latest swing high / ATR fallback
            TP1 = 1.5R
            TP2 = 2.5R
        """

        if row is None:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
            )

        close = self._float(
            row.get("close") if hasattr(row, "get") else None,
            None,
        )

        high = self._float(
            row.get("high") if hasattr(row, "get") else None,
            None,
        )

        low = self._float(
            row.get("low") if hasattr(row, "get") else None,
            None,
        )

        atr = self._float(
            row.get("atr_14") if hasattr(row, "get") else None,
            None,
        )

        if close is None:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
            )

        # ATR fallback.
        if atr is None or atr <= 0:
            if high is not None and low is not None:
                atr = max(high - low, 0.50)
            else:
                atr = 1.00

        # ----------------------------------------------------
        # Extract latest swing prices.
        # ----------------------------------------------------

        swing_high = None
        swing_low = None

        if structure_result is not None:

            highs = self._list(
                self._first(
                    structure_result,
                    ("swing_highs",),
                    [],
                )
            )

            lows = self._list(
                self._first(
                    structure_result,
                    ("swing_lows",),
                    [],
                )
            )

            if highs:
                swing_high = self._float(
                    self._first(
                        highs[-1],
                        ("price", "value"),
                        None,
                    ),
                    None,
                )

            if lows:
                swing_low = self._float(
                    self._first(
                        lows[-1],
                        ("price", "value"),
                        None,
                    ),
                    None,
                )

        # ----------------------------------------------------
        # BUY
        # ----------------------------------------------------

        if direction == "BUY":

            if swing_low is not None and swing_low < close:
                stop_loss = swing_low - (atr * 0.20)
            else:
                stop_loss = close - (atr * 1.20)

            risk = close - stop_loss

            if risk <= 0:
                risk = atr
                stop_loss = close - risk

            entry = close
            tp1 = entry + (risk * 1.50)
            tp2 = entry + (risk * 2.50)

        # ----------------------------------------------------
        # SELL
        # ----------------------------------------------------

        elif direction == "SELL":

            if swing_high is not None and swing_high > close:
                stop_loss = swing_high + (atr * 0.20)
            else:
                stop_loss = close + (atr * 1.20)

            risk = stop_loss - close

            if risk <= 0:
                risk = atr
                stop_loss = close + risk

            entry = close
            tp1 = entry - (risk * 1.50)
            tp2 = entry - (risk * 2.50)

        else:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
            )

        risk = abs(entry - stop_loss)

        if risk <= 0:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
            )

        rr1 = abs(tp1 - entry) / risk
        rr2 = abs(tp2 - entry) / risk

        return (
            entry,
            stop_loss,
            tp1,
            tp2,
            rr1,
            rr2,
        )

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        row=None,
        mtf_trends=None,
        structure_result=None,
        structure_events=None,
        liquidity_result=None,
        trendline_result=None,
        fibonacci_result=None,
        candle_result=None,
        df=None,
    ) -> SignalResult:

        result = SignalResult()

        # ----------------------------------------------------
        # 1. MTF
        # ----------------------------------------------------

        mtf_score, mtf_direction, mtf_reasons = self.score_mtf(
            mtf_trends
        )

        result.components["mtf"] = round(mtf_score, 2)
        result.reasons.extend(mtf_reasons)

        # ----------------------------------------------------
        # 2. MARKET STRUCTURE
        # ----------------------------------------------------

        structure_score, structure_direction, structure_reasons = (
            self.score_structure(structure_result)
        )

        result.components["structure"] = round(structure_score, 2)
        result.reasons.extend(structure_reasons)

        # ----------------------------------------------------
        # 3. LIQUIDITY
        # ----------------------------------------------------

        liquidity_score, liquidity_direction, liquidity_reasons = (
            self.score_liquidity(liquidity_result)
        )

        result.components["liquidity"] = round(liquidity_score, 2)
        result.reasons.extend(liquidity_reasons)

        # ----------------------------------------------------
        # 4. FIBONACCI
        # ----------------------------------------------------

        fibonacci_score, fibonacci_direction, fibonacci_reasons = (
            self.score_fibonacci(fibonacci_result)
        )

        result.components["fibonacci"] = round(fibonacci_score, 2)
        result.reasons.extend(fibonacci_reasons)

        # ----------------------------------------------------
        # 5. TRENDLINE
        # ----------------------------------------------------

        trendline_score, trendline_direction, trendline_reasons = (
            self.score_trendline(trendline_result)
        )

        result.components["trendline"] = round(trendline_score, 2)
        result.reasons.extend(trendline_reasons)

        # ----------------------------------------------------
        # 6. CANDLE
        # ----------------------------------------------------

        candle_score, candle_direction, candle_reasons = (
            self.score_candle(candle_result)
        )

        result.components["candle"] = round(candle_score, 2)
        result.reasons.extend(candle_reasons)

        # ----------------------------------------------------
        # TOTAL
        # ----------------------------------------------------

        result.score = round(
            min(
                sum(result.components.values()),
                100.0,
            ),
            2,
        )

        # ----------------------------------------------------
        # DIRECTION VOTING
        # ----------------------------------------------------

        votes = {
            "BUY": 0.0,
            "SELL": 0.0,
        }

        component_directions = (
            ("mtf", mtf_direction),
            ("structure", structure_direction),
            ("liquidity", liquidity_direction),
            ("fibonacci", fibonacci_direction),
            ("trendline", trendline_direction),
            ("candle", candle_direction),
        )

        for name, direction in component_directions:

            points = result.components.get(name, 0.0)

            if direction == "BUY":
                votes["BUY"] += points

            elif direction == "SELL":
                votes["SELL"] += points

        if votes["BUY"] > votes["SELL"]:
            result.direction = "BUY"

        elif votes["SELL"] > votes["BUY"]:
            result.direction = "SELL"

        else:
            result.direction = "NEUTRAL"

        # ----------------------------------------------------
        # CONFLICT WARNING
        # ----------------------------------------------------

        buy_components = [
            name
            for name, direction in component_directions
            if direction == "BUY"
        ]

        sell_components = [
            name
            for name, direction in component_directions
            if direction == "SELL"
        ]

        if buy_components and sell_components:
            result.warnings.append(
                "BUY/SELL komponentlari o'rtasida qarama-qarshilik mavjud."
            )

            result.warnings.append(
                "Tasdiqlangan signal uchun barcha timeframe va "
                "confirmationlar bir yo'nalishda bo'lishi afzal."
            )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if result.direction == "NEUTRAL":
            result.status = "NO_SIGNAL"
        elif result.score >= self.HIGH_CONFIDENCE_THRESHOLD:
            result.status = "HIGH_CONFIDENCE"
        elif result.score >= self.CONFIRMED_THRESHOLD:
            result.status = "CONFIRMED"
        elif result.score >= self.EARLY_THRESHOLD:
            result.status = "EARLY"
        else:
            result.status = "WATCH"

        # ----------------------------------------------------
        # LAST CLOSED CANDLE
        # ----------------------------------------------------

        if row is None and df is not None:

            try:
                if len(df) >= 2:
                    row = df.iloc[-2]
            except Exception:
                row = None

        # ----------------------------------------------------
        # LEVELS
        # ----------------------------------------------------

        if result.direction in ("BUY", "SELL"):

            (
                entry,
                stop_loss,
                tp1,
                tp2,
                rr1,
                rr2,
            ) = self.calculate_levels(
                row=row,
                direction=result.direction,
                structure_result=structure_result,
                fibonacci_result=fibonacci_result,
            )

            result.entry = entry
            result.stop_loss = stop_loss
            result.take_profit_1 = tp1
            result.take_profit_2 = tp2
            result.risk_reward_1 = rr1
            result.risk_reward_2 = rr2

            if rr1 is not None and rr1 < self.MIN_RR_1:
                result.warnings.append(
                    f"RR1={rr1:.2f} minimal {self.MIN_RR_1:.2f} dan past."
                )

            if rr2 is not None and rr2 < self.MIN_RR_2:
                result.warnings.append(
                    f"RR2={rr2:.2f} minimal {self.MIN_RR_2:.2f} dan past."
                )

        # ----------------------------------------------------
        # FINAL CONFIRMATION GATE
        # ----------------------------------------------------
        candle_confirmed = self._first(
            candle_result,
            ("confirmed", "is_confirmed"),
            False,
        ) is True

        rr_valid = (
            result.risk_reward_1 is not None
            and result.risk_reward_2 is not None
            and result.risk_reward_1 >= self.MIN_RR_1
            and result.risk_reward_2 >= self.MIN_RR_2
        )

        has_conflict = bool(buy_components and sell_components)

        if result.direction == "NEUTRAL":
            result.status = "NO_SIGNAL"
        elif (
            result.score >= self.HIGH_CONFIDENCE_THRESHOLD
            and candle_confirmed
            and rr_valid
            and not has_conflict
        ):
            result.status = "HIGH_CONFIDENCE"
        elif (
            result.score >= self.CONFIRMED_THRESHOLD
            and candle_confirmed
            and rr_valid
            and not has_conflict
        ):
            result.status = "CONFIRMED"
        elif result.score >= self.EARLY_THRESHOLD:
            result.status = "EARLY"
        else:
            result.status = "WATCH"

        if result.status in {"CONFIRMED", "HIGH_CONFIDENCE"}:
            result.reasons.append(
                "Professional confirmation gate passed: "
                "score + candle + RR + conflict checks."
            )
        else:
            if not candle_confirmed:
                result.warnings.append(
                    "Candle confirmation mavjud emas: tasdiqlangan signal bloklandi."
                )
            if not rr_valid and result.direction != "NEUTRAL":
                result.warnings.append(
                    "Risk/Reward talabi bajarilmadi: tasdiqlangan signal bloklandi."
                )
            if has_conflict:
                result.warnings.append(
                    "Qarama-qarshi komponentlar mavjud: tasdiqlangan signal bloklandi."
                )

        # ----------------------------------------------------
        # FINAL REASONS
        # ----------------------------------------------------

        result.reasons.append(
            f"Yakuniy score: {result.score:.2f}/100."
        )

        result.reasons.append(
            f"Yakuniy yo'nalish: {result.direction}."
        )

        result.reasons.append(
            f"Signal holati: {result.status}."
        )

        return result

    # ========================================================
    # ANALYZE — MAIN.PY INTEGRATION
    # ========================================================

    def analyze(
        self,
        mtf_data=None,
        mtf_trends=None,
        structure_results=None,
        trendline_results=None,
        liquidity_results=None,
        fibonacci_results=None,
        candle_results=None,
        df=None,
        **kwargs,
    ):
        """
        Compatible with current main.py.

        main.py sends:
            mtf_data=indicator_data
            mtf_trends=mtf_trends
            structure_results=structure_results
            trendline_results=trendline_results
            liquidity_results=liquidity_results
            fibonacci_results=fibonacci_results
            candle_results=candle_confirmation_results

        All M5 component dictionaries are unwrapped here.
        """

        # ----------------------------------------------------
        # M5 DATAFRAME
        # ----------------------------------------------------

        m5 = df

        if m5 is None and isinstance(mtf_data, dict):

            candidate = mtf_data.get("M5")

            if candidate is not None:
                m5 = candidate

            else:
                candidate = mtf_data.get("m5")

                if candidate is not None:
                    m5 = candidate

        # ----------------------------------------------------
        # M5 STRUCTURE
        # ----------------------------------------------------

        structure_result = self._extract_m5(
            structure_results
        )

        # ----------------------------------------------------
        # M5 TRENDLINE
        # ----------------------------------------------------

        trendline_result = self._extract_m5(
            trendline_results
        )

        # ----------------------------------------------------
        # M5 LIQUIDITY
        # ----------------------------------------------------

        liquidity_result = self._extract_m5(
            liquidity_results
        )

        # ----------------------------------------------------
        # M5 FIBONACCI
        # ----------------------------------------------------

        fibonacci_result = self._extract_m5(
            fibonacci_results
        )

        # ----------------------------------------------------
        # M5 CANDLE
        # ----------------------------------------------------

        candle_result = self._extract_m5(
            candle_results
        )

        # ----------------------------------------------------
        # MTF
        # ----------------------------------------------------

        if isinstance(mtf_trends, dict):
            trends = mtf_trends
        else:
            trends = {}

        # ----------------------------------------------------
        # LAST CLOSED CANDLE
        # ----------------------------------------------------

        row = None

        if m5 is not None:

            try:
                if len(m5) >= 2:
                    row = m5.iloc[-2]
            except Exception:
                row = None

        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        return self.generate(
            row=row,
            mtf_trends=trends,
            structure_result=structure_result,
            liquidity_result=liquidity_result,
            trendline_result=trendline_result,
            fibonacci_result=fibonacci_result,
            candle_result=candle_result,
            df=m5,
        )
