from __future__ import annotations

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """
    PRO TRADING BOT uchun texnik indikatorlar.

    Mavjud:
        EMA 20 / 50 / 200
        SMA 20 / 50 / 200
        EMA 20/50 kesishishi
        RSI 14
        MACD
        ATR 14
        ADX 14
        Bollinger Bands
        Stochastic
        Volume
    """

    # ==========================================================
    # EMA
    # ==========================================================

    @staticmethod
    def ema(
        df: pd.DataFrame,
        period: int,
        column: str = "close",
    ) -> pd.Series:

        return df[column].ewm(
            span=period,
            adjust=False,
        ).mean()

    # ==========================================================
    # SMA
    # ==========================================================

    @staticmethod
    def sma(
        df: pd.DataFrame,
        period: int,
        column: str = "close",
    ) -> pd.Series:

        return df[column].rolling(
            window=period,
            min_periods=period,
        ).mean()

    # ==========================================================
    # EMA CROSSOVER
    # ==========================================================

    @staticmethod
    def ma_crossover(
        df: pd.DataFrame,
        fast_period: int = 20,
        slow_period: int = 50,
    ) -> pd.DataFrame:

        fast = df["close"].ewm(
            span=fast_period,
            adjust=False,
        ).mean()

        slow = df["close"].ewm(
            span=slow_period,
            adjust=False,
        ).mean()

        previous_fast = fast.shift(1)
        previous_slow = slow.shift(1)

        bullish_cross = (
            (previous_fast <= previous_slow)
            & (fast > slow)
        )

        bearish_cross = (
            (previous_fast >= previous_slow)
            & (fast < slow)
        )

        return pd.DataFrame(
            {
                "ma_fast": fast,
                "ma_slow": slow,
                "ma_bullish_cross": bullish_cross,
                "ma_bearish_cross": bearish_cross,
            },
            index=df.index,
        )

    # ==========================================================
    # RSI
    # ==========================================================

    @staticmethod
    def rsi(
        df: pd.DataFrame,
        period: int = 14,
        column: str = "close",
    ) -> pd.Series:

        delta = df[column].diff()

        gain = delta.clip(
            lower=0
        )

        loss = -delta.clip(
            upper=0
        )

        average_gain = gain.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()

        average_loss = loss.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()

        rs = (
            average_gain
            / average_loss.replace(
                0,
                np.nan,
            )
        )

        rsi = 100 - (
            100 / (1 + rs)
        )

        return rsi.fillna(50)

    # ==========================================================
    # MACD
    # ==========================================================

    @staticmethod
    def macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str = "close",
    ) -> pd.DataFrame:

        fast_ema = df[column].ewm(
            span=fast,
            adjust=False,
        ).mean()

        slow_ema = df[column].ewm(
            span=slow,
            adjust=False,
        ).mean()

        macd_line = (
            fast_ema - slow_ema
        )

        signal_line = macd_line.ewm(
            span=signal,
            adjust=False,
        ).mean()

        histogram = (
            macd_line - signal_line
        )

        return pd.DataFrame(
            {
                "macd": macd_line,
                "macd_signal": signal_line,
                "macd_histogram": histogram,
            },
            index=df.index,
        )

    # ==========================================================
    # ATR
    # ==========================================================

    @staticmethod
    def atr(
        df: pd.DataFrame,
        period: int = 14,
    ) -> pd.Series:

        previous_close = df[
            "close"
        ].shift(1)

        true_range = pd.concat(
            [
                df["high"] - df["low"],

                (
                    df["high"]
                    - previous_close
                ).abs(),

                (
                    df["low"]
                    - previous_close
                ).abs(),
            ],
            axis=1,
        ).max(axis=1)

        return true_range.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()

    # ==========================================================
    # ADX
    # ==========================================================

    @staticmethod
    def adx(
        df: pd.DataFrame,
        period: int = 14,
    ) -> pd.DataFrame:

        high = df["high"]
        low = df["low"]
        close = df["close"]

        up_move = high.diff()

        down_move = -low.diff()

        plus_dm = pd.Series(
            np.where(
                (
                    (up_move > down_move)
                    & (up_move > 0)
                ),
                up_move,
                0.0,
            ),
            index=df.index,
        )

        minus_dm = pd.Series(
            np.where(
                (
                    (down_move > up_move)
                    & (down_move > 0)
                ),
                down_move,
                0.0,
            ),
            index=df.index,
        )

        previous_close = close.shift(1)

        true_range = pd.concat(
            [
                high - low,

                (
                    high
                    - previous_close
                ).abs(),

                (
                    low
                    - previous_close
                ).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = true_range.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()

        plus_dm_average = plus_dm.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()

        minus_dm_average = minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()

        plus_di = (
            100
            * plus_dm_average
            / atr.replace(
                0,
                np.nan,
            )
        )

        minus_di = (
            100
            * minus_dm_average
            / atr.replace(
                0,
                np.nan,
            )
        )

        denominator = (
            plus_di + minus_di
        ).replace(
            0,
            np.nan,
        )

        dx = (
            100
            * (plus_di - minus_di).abs()
            / denominator
        )

        adx = dx.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()

        return pd.DataFrame(
            {
                "plus_di": plus_di,
                "minus_di": minus_di,
                "adx": adx,
            },
            index=df.index,
        )

    # ==========================================================
    # BOLLINGER BANDS
    # ==========================================================

    @staticmethod
    def bollinger(
        df: pd.DataFrame,
        period: int = 20,
        deviation: float = 2.0,
        column: str = "close",
    ) -> pd.DataFrame:

        middle = df[column].rolling(
            window=period,
            min_periods=period,
        ).mean()

        std = df[column].rolling(
            window=period,
            min_periods=period,
        ).std()

        upper = (
            middle
            + deviation * std
        )

        lower = (
            middle
            - deviation * std
        )

        return pd.DataFrame(
            {
                "bb_upper": upper,
                "bb_middle": middle,
                "bb_lower": lower,
            },
            index=df.index,
        )

    # ==========================================================
    # STOCHASTIC
    # ==========================================================

    @staticmethod
    def stochastic(
        df: pd.DataFrame,
        period: int = 14,
        smooth: int = 3,
    ) -> pd.DataFrame:

        lowest_low = df[
            "low"
        ].rolling(
            window=period,
            min_periods=period,
        ).min()

        highest_high = df[
            "high"
        ].rolling(
            window=period,
            min_periods=period,
        ).max()

        denominator = (
            highest_high
            - lowest_low
        ).replace(
            0,
            np.nan,
        )

        k = (
            100
            * (
                df["close"]
                - lowest_low
            )
            / denominator
        )

        d = k.rolling(
            window=smooth,
            min_periods=smooth,
        ).mean()

        return pd.DataFrame(
            {
                "stoch_k": k,
                "stoch_d": d,
            },
            index=df.index,
        )

    # ==========================================================
    # VOLUME
    # ==========================================================

    @staticmethod
    def volume_analysis(
        df: pd.DataFrame,
        period: int = 20,
    ) -> pd.DataFrame:

        volume = df["tick_volume"]

        average_volume = volume.rolling(
            window=period,
            min_periods=period,
        ).mean()

        volume_ratio = (
            volume
            / average_volume.replace(
                0,
                np.nan,
            )
        )

        high_volume = (
            volume_ratio >= 1.5
        )

        low_volume = (
            volume_ratio <= 0.7
        )

        return pd.DataFrame(
            {
                "volume_average": average_volume,
                "volume_ratio": volume_ratio,
                "high_volume": high_volume,
                "low_volume": low_volume,
            },
            index=df.index,
        )

    # ==========================================================
    # HAMMASINI HISOBLASH
    # ==========================================================

    @staticmethod
    def add_all(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        result = df.copy()

        # ==================================================
        # EMA
        # ==================================================

        result["ema_20"] = (
            TechnicalIndicators.ema(
                result,
                20,
            )
        )

        result["ema_50"] = (
            TechnicalIndicators.ema(
                result,
                50,
            )
        )

        result["ema_200"] = (
            TechnicalIndicators.ema(
                result,
                200,
            )
        )

        # ==================================================
        # SMA
        # ==================================================

        result["sma_20"] = (
            TechnicalIndicators.sma(
                result,
                20,
            )
        )

        result["sma_50"] = (
            TechnicalIndicators.sma(
                result,
                50,
            )
        )

        result["sma_200"] = (
            TechnicalIndicators.sma(
                result,
                200,
            )
        )

        # ==================================================
        # EMA 20 / 50 CROSSOVER
        # ==================================================

        crossover = (
            TechnicalIndicators.ma_crossover(
                result,
                fast_period=20,
                slow_period=50,
            )
        )

        result = result.join(
            crossover
        )

        # ==================================================
        # RSI
        # ==================================================

        result["rsi_14"] = (
            TechnicalIndicators.rsi(
                result,
                14,
            )
        )

        # ==================================================
        # MACD
        # ==================================================

        macd = (
            TechnicalIndicators.macd(
                result,
                fast=12,
                slow=26,
                signal=9,
            )
        )

        result = result.join(
            macd
        )

        # ==================================================
        # ATR
        # ==================================================

        result["atr_14"] = (
            TechnicalIndicators.atr(
                result,
                14,
            )
        )

        # ==================================================
        # ADX
        # ==================================================

        adx = (
            TechnicalIndicators.adx(
                result,
                14,
            )
        )

        result = result.join(
            adx
        )

        # ==================================================
        # BOLLINGER
        # ==================================================

        bollinger = (
            TechnicalIndicators.bollinger(
                result,
                period=20,
                deviation=2.0,
            )
        )

        result = result.join(
            bollinger
        )

        # ==================================================
        # STOCHASTIC
        # ==================================================

        stochastic = (
            TechnicalIndicators.stochastic(
                result,
                period=14,
                smooth=3,
            )
        )

        result = result.join(
            stochastic
        )

        # ==================================================
        # VOLUME
        # ==================================================

        volume = (
            TechnicalIndicators.volume_analysis(
                result,
                period=20,
            )
        )

        result = result.join(
            volume
        )

        return result