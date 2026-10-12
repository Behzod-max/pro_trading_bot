from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd


# ============================================================
# TIMEFRAME CONFIG
# ============================================================

@dataclass(frozen=True)
class TimeframeConfig:
    """
    Timeframe konfiguratsiyasi.
    """

    name: str
    mt5_timeframe: int


# ============================================================
# MARKET DATA
# ============================================================

class MarketData:
    """
    MT5 dan GOLD uchun real multi-timeframe
    OHLCV ma'lumotlarini oladi.

    Timeframe:

        M1  -> precision
        M5  -> entry
        M15 -> setup
        M30 -> confirmation
        H1  -> trend
        H4  -> major structure
        D1  -> macro trend
    """

    # ========================================================
    # TIMEFRAMES
    # ========================================================

    TIMEFRAMES = {

        "M1": TimeframeConfig(
            "M1",
            mt5.TIMEFRAME_M1
        ),

        "M5": TimeframeConfig(
            "M5",
            mt5.TIMEFRAME_M5
        ),

        "M15": TimeframeConfig(
            "M15",
            mt5.TIMEFRAME_M15
        ),

        "M30": TimeframeConfig(
            "M30",
            mt5.TIMEFRAME_M30
        ),

        "H1": TimeframeConfig(
            "H1",
            mt5.TIMEFRAME_H1
        ),

        "H4": TimeframeConfig(
            "H4",
            mt5.TIMEFRAME_H4
        ),

        "D1": TimeframeConfig(
            "D1",
            mt5.TIMEFRAME_D1
        ),
    }

    # ========================================================
    # INIT
    # ========================================================

    def __init__(self, symbol: str = "GOLD"):

        self.symbol = symbol

    # ========================================================
    # SYMBOL
    # ========================================================

    def ensure_symbol(self) -> bool:
        """
        GOLD Market Watch ichida mavjudligini
        va tanlanganligini tekshiradi.
        """

        if not self.symbol:

            print(
                "❌ Symbol nomi bo'sh."
            )

            return False

        info = mt5.symbol_info(
            self.symbol
        )

        if info is None:

            print(
                f"❌ Symbol topilmadi: {self.symbol}"
            )

            print(
                f"MT5 error: {mt5.last_error()}"
            )

            return False

        # Agar Market Watch'da ko'rinmasa,
        # avtomatik qo'shishga harakat qilamiz.
        if not info.visible:

            selected = mt5.symbol_select(
                self.symbol,
                True
            )

            if not selected:

                print(
                    "❌ Symbolni Market Watch "
                    f"ga qo'shib bo'lmadi: {self.symbol}"
                )

                print(
                    f"MT5 error: {mt5.last_error()}"
                )

                return False

        return True

    # ========================================================
    # CANDLES
    # ========================================================

    def get_candles(
        self,
        timeframe: str,
        count: int = 500,
    ) -> Optional[pd.DataFrame]:
        """
        Bitta timeframe uchun candle ma'lumotlarini oladi.

        Masalan:

            get_candles("M5", 500)
        """

        # ----------------------------------------------------
        # TIMEFRAME
        # ----------------------------------------------------

        if not isinstance(
            timeframe,
            str
        ):

            raise TypeError(
                "timeframe string bo'lishi kerak. "
                "Masalan: 'M5'"
            )

        timeframe = timeframe.upper().strip()

        if timeframe not in self.TIMEFRAMES:

            raise ValueError(
                f"Noto'g'ri timeframe: {timeframe}. "
                f"Mavjud: {list(self.TIMEFRAMES.keys())}"
            )

        # ----------------------------------------------------
        # COUNT
        # ----------------------------------------------------

        if not isinstance(
            count,
            int
        ):

            raise TypeError(
                "count integer bo'lishi kerak."
            )

        if count < 10:

            raise ValueError(
                "count kamida 10 bo'lishi kerak."
            )

        # ----------------------------------------------------
        # SYMBOL
        # ----------------------------------------------------

        if not self.ensure_symbol():

            return None

        config = self.TIMEFRAMES[
            timeframe
        ]

        # ----------------------------------------------------
        # MT5 REQUEST
        # ----------------------------------------------------

        rates = mt5.copy_rates_from_pos(
            self.symbol,
            config.mt5_timeframe,
            0,
            count
        )

        if rates is None:

            error = mt5.last_error()

            print(
                f"❌ {self.symbol} {timeframe} "
                f"candle olinmadi."
            )

            print(
                f"MT5 error: {error}"
            )

            return None

        if len(rates) == 0:

            print(
                f"❌ {self.symbol} {timeframe} "
                "uchun candle mavjud emas."
            )

            return None

        # ----------------------------------------------------
        # DATAFRAME
        # ----------------------------------------------------

        df = pd.DataFrame(
            rates
        )

        if df.empty:

            return None

        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        df["time"] = pd.to_datetime(
            df["time"],
            unit="s",
            utc=True
        )

        # ----------------------------------------------------
        # REQUIRED COLUMNS
        # ----------------------------------------------------

        required_columns = [
            "time",
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
        ]

        for column in required_columns:

            if column not in df.columns:

                df[column] = 0

        df = df[
            required_columns
        ].copy()

        # ----------------------------------------------------
        # NUMERIC CLEANUP
        # ----------------------------------------------------

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
        ]

        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        # ----------------------------------------------------
        # SORT
        # ----------------------------------------------------

        df = (
            df
            .sort_values("time")
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # REMOVE INVALID CANDLES
        # ----------------------------------------------------

        df = (
            df
            .dropna(
                subset=[
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                ]
            )
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        df.attrs["symbol"] = self.symbol

        df.attrs["timeframe"] = timeframe

        df.attrs["mt5_timeframe"] = (
            config.mt5_timeframe
        )

        return df

    # ========================================================
    # MULTI TIMEFRAME
    # ========================================================

    def get_multi_timeframe(
        self,
        count: int = 500,
        timeframes: Optional[list[str]] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Bir nechta timeframe uchun
        candle DataFrame qaytaradi.
        """

        if timeframes is None:

            timeframes = [
                "M1",
                "M5",
                "M15",
                "M30",
                "H1",
                "H4",
                "D1",
            ]

        result: dict[
            str,
            pd.DataFrame
        ] = {}

        for timeframe in timeframes:

            try:

                df = self.get_candles(
                    timeframe=timeframe,
                    count=count
                )

                if (
                    df is not None
                    and not df.empty
                ):

                    result[
                        timeframe
                    ] = df

            except Exception as e:

                print(
                    f"❌ {timeframe}: "
                    f"{e}"
                )

        return result

    # ========================================================
    # LATEST CANDLE
    # ========================================================

    def get_latest_candle(
        self,
        timeframe: str,
    ) -> Optional[pd.Series]:
        """
        Timeframe bo'yicha eng oxirgi candle.
        """

        df = self.get_candles(
            timeframe=timeframe,
            count=10
        )

        if df is None or df.empty:

            return None

        return df.iloc[-1]

    # ========================================================
    # BID / ASK
    # ========================================================

    def get_tick(self):
        """
        GOLD ning real-time Bid / Ask ma'lumotlari.
        """

        if not self.ensure_symbol():

            return None

        tick = mt5.symbol_info_tick(
            self.symbol
        )

        if tick is None:

            print(
                f"❌ {self.symbol} tick olinmadi."
            )

            print(
                f"MT5 error: {mt5.last_error()}"
            )

            return None

        return tick

    # ========================================================
    # PRICE
    # ========================================================

    def get_price(
        self
    ) -> Optional[dict]:
        """
        Bid / Ask / Spread.
        """

        tick = self.get_tick()

        if tick is None:

            return None

        bid = float(
            tick.bid
        )

        ask = float(
            tick.ask
        )

        spread = ask - bid

        return {
            "symbol": self.symbol,
            "bid": bid,
            "ask": ask,
            "spread": spread,
        }

    # ========================================================
    # SYMBOL INFO
    # ========================================================

    def get_symbol_info(self):
        """
        Broker GOLD parametrlarini qaytaradi.
        """

        if not self.ensure_symbol():

            return None

        return mt5.symbol_info(
            self.symbol
        )

    # ========================================================
    # DATA VALIDATION
    # ========================================================

    @staticmethod
    def validate_dataframe(
        df: Optional[pd.DataFrame],
        minimum_rows: int = 100,
    ) -> bool:
        """
        Candle ma'lumotlari analiz uchun
        yetarliligini tekshiradi.
        """

        if df is None:

            return False

        if df.empty:

            return False

        if len(df) < minimum_rows:

            return False

        required = {
            "time",
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
        }

        if not required.issubset(
            df.columns
        ):

            return False

        if df["close"].isna().any():

            return False

        if df["high"].isna().any():

            return False

        if df["low"].isna().any():

            return False

        return True

    # ========================================================
    # DEBUG / SUMMARY
    # ========================================================

    def timeframe_summary(
        self,
        timeframes: Optional[list[str]] = None,
        count: int = 100,
    ) -> dict:
        """
        Har bir timeframe bo'yicha
        qisqa diagnostika.
        """

        if timeframes is None:

            timeframes = [
                "M1",
                "M5",
                "M15",
                "M30",
                "H1",
                "H4",
                "D1",
            ]

        summary = {}

        for timeframe in timeframes:

            try:

                df = self.get_candles(
                    timeframe=timeframe,
                    count=count
                )

            except Exception as e:

                summary[
                    timeframe
                ] = {
                    "status": "ERROR",
                    "rows": 0,
                    "error": str(e),
                }

                continue

            if df is None or df.empty:

                summary[
                    timeframe
                ] = {
                    "status": "ERROR",
                    "rows": 0,
                }

                continue

            first_time = (
                df.iloc[0]["time"]
            )

            last_time = (
                df.iloc[-1]["time"]
            )

            last_close = float(
                df.iloc[-1]["close"]
            )

            summary[
                timeframe
            ] = {
                "status": "OK",
                "rows": len(df),
                "first_time": first_time,
                "last_time": last_time,
                "last_close": last_close,
            }

        return summary