
from __future__ import annotations

import os
import re
import sys
import io
import contextlib
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd

from config.settings import settings
from core.market_data import MarketData
from core.mt5_client import MT5Client
from indicators.technical import TechnicalIndicators
from analysis.market_structure import MarketStructure
from analysis.trendline import TrendLineEngine
from analysis.liquidity import LiquidityEngine
from analysis.fibonacci import FibonacciEngine
from analysis.candle_confirmation import CandleConfirmationEngine
from analysis.signal_engine import SignalEngine
from risk.entry_exit_optimizer import EntryExitOptimizer
from risk.risk_manager import RiskManager
from analysis.ai_analyst import AIAnalyst
from backtest.backtest_engine import BacktestEngine
from paper_trading.paper_trading_engine import PaperTradingEngine


# ============================================================
# YORDAMCHI FUNKSIYALAR
# ============================================================

def ajratgich(belgi="=", uzunlik=70):
    print(belgi * uzunlik)


def xavfsiz_son(qiymat, default=0.0):
    try:
        if qiymat is None:
            return default

        qiymat = float(qiymat)

        if pd.isna(qiymat):
            return default

        return qiymat

    except (TypeError, ValueError):
        return default


def yonalishni_normallashtir(value):
    if value is None:
        return "NEUTRAL"

    text = str(value).upper().strip()

    mapping = {
        "BUY": "BUY",
        "SELL": "SELL",
        "LONG": "BUY",
        "SHORT": "SELL",
        "BULLISH": "BUY",
        "BEARISH": "SELL",
        "UP": "BUY",
        "DOWN": "SELL",
        "NEUTRAL": "NEUTRAL",
        "NONE": "NEUTRAL",
        "UNKNOWN": "NEUTRAL",
    }

    return mapping.get(text, "NEUTRAL")


def yonalish_uz(value):
    value = yonalishni_normallashtir(value)

    if value == "BUY":
        return "🟢 BUY — SOTIB OLISH"

    if value == "SELL":
        return "🔴 SELL — SOTISH"

    return "⚪ NEUTRAL — YO'NALISH NOANIQ"


def holat_uz(value):
    if value is None:
        return "NO SIGNAL"

    text = str(value).upper().strip()

    mapping = {
        "NO_SIGNAL": "NO SIGNAL",
        "NO SIGNAL": "NO SIGNAL",
        "WATCH": "KUZATISH",
        "EARLY": "ERTA SIGNAL",
        "CONFIRMED": "TASDIQLANGAN SIGNAL",
        "HIGH_CONFIDENCE": "YUQORI ISHONCHLI SIGNAL",
        "HIGH-CONFIDENCE": "YUQORI ISHONCHLI SIGNAL",
    }

    return mapping.get(text, text)


def trend_uz(value):
    if value is None:
        return "NOANIQ"

    text = str(value).upper().strip()

    mapping = {
        "UPTREND": "O'SISH TRENDI",
        "DOWNTREND": "PASAYISH TRENDI",
        "RANGE": "RANGE / YONALISH YO'Q",
        "SIDEWAYS": "YONALISH YO'Q",
        "BULLISH": "O'SISH TRENDI",
        "BEARISH": "PASAYISH TRENDI",
        "UP": "O'SISH",
        "DOWN": "PASAYISH",
        "NEUTRAL": "NEUTRAL",
        "UNKNOWN": "NOANIQ",
    }

    return mapping.get(text, text)


def oxirgi_yopilgan_candle(df):
    """
    [-1] = hozir shakllanayotgan candle
    [-2] = oxirgi yopilgan candle
    """

    if df is None or len(df) < 2:
        return None

    return df.iloc[-2]


def qiymat_olish(obj, key, default=None):
    """
    Dict yoki objectdan qiymat olish.
    """

    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


# ============================================================
# MTF TREND
# ============================================================

def mtf_trendni_aniqlash(
    indicator_data,
):
    """
    Multi-timeframe trendni EMA asosida aniqlaydi.
    """

    natija = {}

    for timeframe, df in indicator_data.items():

        try:

            if df is None or df.empty:
                natija[timeframe] = "UNKNOWN"
                continue

            row = (
                df.iloc[-2]
                if len(df) >= 2
                else df.iloc[-1]
            )

            close = xavfsiz_son(
                row.get("close")
            )

            ema20 = xavfsiz_son(
                row.get("ema_20")
            )

            ema50 = xavfsiz_son(
                row.get("ema_50")
            )

            ema200 = xavfsiz_son(
                row.get("ema_200")
            )

            if (
                close > ema20
                and ema20 > ema50
                and ema50 > ema200
            ):

                natija[timeframe] = "UPTREND"

            elif (
                close < ema20
                and ema20 < ema50
                and ema50 < ema200
            ):

                natija[timeframe] = "DOWNTREND"

            elif close > ema50:

                natija[timeframe] = "UPTREND"

            elif close < ema50:

                natija[timeframe] = "DOWNTREND"

            else:

                natija[timeframe] = "RANGE"

        except Exception:

            natija[timeframe] = "UNKNOWN"

    return natija


# ============================================================
# SIGNAL CHIQARISH
# ============================================================

def signalni_chiqarish(
    signal_result,
    model_balance,
    risk_percent,
    current_price,
):

    print()
    ajratgich()
    print("🎯 SIGNAL ENGINE")
    ajratgich()

    if signal_result is None:

        print(
            "❌ Signal natijasi mavjud emas."
        )

        return

    direction = yonalishni_normallashtir(
        qiymat_olish(
            signal_result,
            "direction"
        )
    )

    score = xavfsiz_son(
        qiymat_olish(
            signal_result,
            "score",
            0
        )
    )

    status = qiymat_olish(
        signal_result,
        "status",
        "NO_SIGNAL"
    )

    entry = qiymat_olish(
        signal_result,
        "entry"
    )

    stop_loss = qiymat_olish(
        signal_result,
        "stop_loss"
    )

    tp1 = qiymat_olish(
        signal_result,
        "take_profit_1"
    )

    tp2 = qiymat_olish(
        signal_result,
        "take_profit_2"
    )

    rr1 = xavfsiz_son(
        qiymat_olish(
            signal_result,
            "risk_reward_1",
            0
        )
    )

    rr2 = xavfsiz_son(
        qiymat_olish(
            signal_result,
            "risk_reward_2",
            0
        )
    )

    print(
        f"Yo'nalish : {yonalish_uz(direction)}"
    )

    print(
        f"Ball      : {score:.1f}/100"
    )

    print(
        f"Holat     : {holat_uz(status)}"
    )

    print()
    print("💰 SAVDO PARAMETRLARI")

    if entry is not None:
        print(
            f"Entry     : {xavfsiz_son(entry):.2f}"
        )
    else:
        print("Entry     : -")

    if stop_loss is not None:
        print(
            f"Stop Loss : {xavfsiz_son(stop_loss):.2f}"
        )
    else:
        print("Stop Loss : -")

    if tp1 is not None:
        print(
            f"TP1       : {xavfsiz_son(tp1):.2f}"
        )
    else:
        print("TP1       : -")

    if tp2 is not None:
        print(
            f"TP2       : {xavfsiz_son(tp2):.2f}"
        )
    else:
        print("TP2       : -")

    print(
        f"RR1       : {rr1:.2f}"
    )

    print(
        f"RR2       : {rr2:.2f}"
    )

    # ========================================================
    # RISK
    # ========================================================

    print()
    print("💵 RISK")

    risk_amount = (
        model_balance
        * risk_percent
        / 100
    )

    print(
        f"Trading hisob balansi : ${model_balance:.2f}"
    )

    print(
        f"Risk         : {risk_percent:.2f}%"
    )

    print(
        f"1 ta savdo riski : ${risk_amount:.2f}"
    )

    if current_price:

        print(
            f"Joriy narx  : {current_price:.2f}"
        )

    # ========================================================
    # SABABLAR
    # ========================================================

    reasons = qiymat_olish(
        signal_result,
        "reasons",
        []
    )

    if reasons:

        print()
        print("🧠 SIGNAL SABABLARI")

        for reason in reasons:

            print(
                f"  • {reason}"
            )

    # ========================================================
    # OGOHLANTIRISHLAR
    # ========================================================

    warnings = qiymat_olish(
        signal_result,
        "warnings",
        []
    )

    if warnings:

        print()
        print("⚠️ OGOHLANTIRISHLAR")

        for warning in warnings:

            print(
                f"  • {warning}"
            )

    # ========================================================
    # COMPONENTS
    # ========================================================

    components = qiymat_olish(
        signal_result,
        "components"
    )

    if components:

        print()
        print("📊 SIGNAL KOMPONENTLARI")

        if isinstance(
            components,
            dict
        ):

            for key, value in components.items():

                if isinstance(
                    value,
                    (int, float)
                ):

                    print(
                        f"  {key:<25}: {value:.2f}"
                    )

                else:

                    print(
                        f"  {key:<25}: {value}"
                    )



def qisqa_signalni_chiqarish(raw_output: str) -> None:
    lines = raw_output.splitlines()

    status = "NO_SIGNAL"
    direction = "NEUTRAL"
    score = None

    final_idx = -1
    for i, line in enumerate(lines):
        if "📌 YAKUNIY HOLAT" in line:
            final_idx = i

    search_slice = lines[final_idx:] if final_idx >= 0 else lines[-120:]

    for line in search_slice:
        if line.strip().startswith("Holat"):
            value = line.split(":", 1)[1].strip().upper()
            if "YUQORI ISHONCHLI" in value:
                status = "HIGH_CONFIDENCE"
            elif "TASDIQLANGAN" in value:
                status = "CONFIRMED"
            elif "ERTA SIGNAL" in value:
                status = "EARLY"
            else:
                status = "NO_SIGNAL"
        elif line.strip().startswith("Yo'nalish"):
            if "BUY" in line:
                direction = "BUY"
            elif "SELL" in line:
                direction = "SELL"
        elif line.strip().startswith("Ball"):
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", line)
            if m:
                score = float(m.group(1))

    risk_valid = any(
        "Holat           : ✅ RISK VALID" in line
        or "Holat           : ✅ VALID" in line
        for line in lines
    )

    actionable = status in {"CONFIRMED", "HIGH_CONFIDENCE"} and risk_valid

    print()
    ajratgich()
    print("🎯 XAUUSD — ANIQ SAVDO SIGNALI")
    ajratgich()

    if not actionable:
        if direction == "NEUTRAL":
            print("⚪ Hozircha aniq signal yo'q.")
        else:
            print(f"Yo'nalish : {yonalish_uz(direction)}")
            if score is not None:
                print(f"Ball      : {score:.1f}/100")
            print("Holat     : ⏳ TASDIQ KUTILMOQDA")
            print()
            print("❌ Hozircha savdo signali berilmaydi.")
        print()
        ajratgich()
        return

    block_start = None
    for i, line in enumerate(lines):
        if "💹 REAL MARKET ENTRY" in line:
            block_start = i
            break

    print(f"Yo'nalish : {yonalish_uz(direction)}")
    if score is not None:
        print(f"Ball      : {score:.1f}/100")
    print(f"Holat     : {holat_uz(status)}")
    print()

    if block_start is not None:
        for line in lines[block_start + 1:]:
            if line.startswith("=" * 10):
                break
            if any(
                key in line
                for key in (
                    "Analiz Entry",
                    "Market Entry",
                    "Stop Loss",
                    "TP1",
                    "TP2",
                )
            ):
                print(line.strip())

    print()
    print("✅ RISK TASDIQLANGAN")
    for line in lines:
        if "Final Lot" in line or "Maksimal risk" in line:
            print(line.strip())

    print()
    ajratgich()



# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HEADER
    # ========================================================

    print()

    ajratgich()

    print(
        "🚀 PRO TRADING BOT — GOLD / XAUUSD"
    )

    ajratgich()

    print(
        f"⏱ Vaqt: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    mode = os.getenv(
        "MODE",
        "analysis"
    ).lower()

    print(
        f"⚙️ Ishlash rejimi: {mode.upper()}"
    )

    # ========================================================
    # SETTINGS
    # ========================================================

    # Risk foizi ENV orqali boshqariladi.
    risk_percent = xavfsiz_son(
        os.getenv(
            "RISK_PERCENT",
            getattr(settings, "RISK_PERCENT", 1.0),
        ),
        getattr(settings, "RISK_PERCENT", 1.0),
    )

    # Model balans endi asosiy rejimda MT5 trading hisobidan avtomatik olinadi.
    # MT5 account_info ishlamasa, xavfsiz fallback sifatida ENV/settings ishlatiladi.
    model_balance = 0.0
    balance_source = "MT5"

    live_trading = bool(
        getattr(
            settings,
            "LIVE_TRADING",
            False
        )
    )

    print()

    ajratgich()

    print(
        "⚙️ BOT SOZLAMALARI"
    )

    ajratgich()

    print(
        f"Model balans : ${model_balance:.2f}"
    )

    print(
        f"Risk         : {risk_percent:.2f}%"
    )

    risk_amount = (
        model_balance
        * risk_percent
        / 100
    )

    print(
        f"1 ta savdo riski : ${risk_amount:.2f}"
    )

    print(
        "Haqiqiy savdo : "
        + (
            "YOQILGAN"
            if live_trading
            else "O'CHIRILGAN"
        )
    )

    # ========================================================
    # MT5 CLIENT
    # ========================================================

    print()

    ajratgich()

    print(
        "🔌 MT5 BILAN ULANISH"
    )

    ajratgich()

    client = MT5Client()

    if not client.connect():

        print(
            "❌ MT5 bilan ulanish amalga oshmadi."
        )

        return

    print(
        "✅ MT5 muvaffaqiyatli ulandi."
    )

    # ========================================================
    # TERMINAL
    # ========================================================

    try:

        terminal = mt5.terminal_info()

        if terminal:

            print()

            print(
                f"Terminal nomi : {terminal.name}"
            )

            print(
                f"Terminal yo'li: {terminal.path}"
            )

    except Exception as e:

        print(
            f"⚠️ Terminal ma'lumoti xatosi: {e}"
        )

    # ========================================================
    # ACCOUNT
    # ========================================================

    try:

        account = mt5.account_info()

        if account:

            print()
            print("💳 MT5 HISOBI")

            print(
                f"Login   : {account.login}"
            )

            print(
                f"Balans  : ${account.balance:.2f}"
            )

            print(
                f"Equity  : ${account.equity:.2f}"
            )

            print(
                f"Valyuta : {account.currency}"
            )

            # ====================================================
            # AVTOMATIK BALANS
            # ====================================================
            account_balance = xavfsiz_son(
                getattr(account, "balance", 0.0),
                0.0,
            )

            if account_balance >= 0:
                model_balance = account_balance
                balance_source = "MT5 ACCOUNT BALANCE"
            else:
                model_balance = xavfsiz_son(
                    os.getenv(
                        "MODEL_BALANCE",
                        getattr(settings, "MODEL_BALANCE", 0.0),
                    ),
                    0.0,
                )
                balance_source = "ENV / SETTINGS FALLBACK"

            print()
            print(
                "✅ Strategiya balansi MT5 trading hisobidan "
                "avtomatik olindi."
            )
            print(
                f"Balans manbasi : {balance_source}"
            )

        else:
            model_balance = xavfsiz_son(
                os.getenv(
                    "MODEL_BALANCE",
                    getattr(settings, "MODEL_BALANCE", 0.0),
                ),
                0.0,
            )
            balance_source = "ENV / SETTINGS FALLBACK"

    except Exception as e:

        model_balance = xavfsiz_son(
            os.getenv(
                "MODEL_BALANCE",
                getattr(settings, "MODEL_BALANCE", 0.0),
            ),
            0.0,
        )
        balance_source = "ENV / SETTINGS FALLBACK"

        print(
            f"⚠️ Account ma'lumot xatosi: {e}"
        )

    # ========================================================
    # SYMBOL
    # ========================================================

    symbol = getattr(
        settings,
        "SYMBOL",
        "GOLD"
    )

    print()

    ajratgich()

    print(
        "💰 GOLD SYMBOL"
    )

    ajratgich()

    # MarketData symbolni o'zi saqlaydi.
    market_data = MarketData(
        symbol=symbol
    )

    info = market_data.get_symbol_info()

    if info is None:

        print(
            f"❌ {symbol} symbol topilmadi."
        )

        client.shutdown()

        return

    print(
        f"Symbol       : {info.name}"
    )

    print(
        f"Raqamlar      : {info.digits}"
    )

    print(
        f"Point         : {info.point}"
    )

    print(
        f"Minimal lot   : {info.volume_min}"
    )

    print(
        f"Maksimal lot  : {info.volume_max}"
    )

    print(
        f"Lot qadami    : {info.volume_step}"
    )

    # ========================================================
    # REAL-TIME PRICE
    # ========================================================

    print()

    ajratgich()

    print(
        "📈 REAL VAQT NARXI"
    )

    ajratgich()

    current_price = 0.0

    try:

        tick = market_data.get_tick()

    except Exception as e:

        tick = None

        print(
            f"❌ Tick xatosi: {e}"
        )

    if tick:

        bid = xavfsiz_son(
            getattr(
                tick,
                "bid",
                0
            )
        )

        ask = xavfsiz_son(
            getattr(
                tick,
                "ask",
                0
            )
        )

        spread = ask - bid

        if bid > 0 and ask > 0:

            current_price = (
                bid + ask
            ) / 2

        else:

            current_price = bid

        print(
            f"Bid    : {bid:.2f}"
        )

        print(
            f"Ask    : {ask:.2f}"
        )

        print(
            f"Spread : {spread:.2f}"
        )

    else:

        print(
            "❌ Real-time tick olinmadi."
        )

        try:

            price_data = (
                market_data.get_price()
            )

            if price_data:

                bid = xavfsiz_son(
                    price_data.get(
                        "bid"
                    )
                )

                ask = xavfsiz_son(
                    price_data.get(
                        "ask"
                    )
                )

                spread = xavfsiz_son(
                    price_data.get(
                        "spread"
                    )
                )

                current_price = (
                    (bid + ask) / 2
                    if bid > 0 and ask > 0
                    else bid
                )

                print()
                print(
                    "ℹ️ Narx get_price() orqali olindi."
                )

                print(
                    f"Bid    : {bid:.2f}"
                )

                print(
                    f"Ask    : {ask:.2f}"
                )

                print(
                    f"Spread : {spread:.2f}"
                )

        except Exception as e:

            print(
                f"⚠️ Alternativ narx olishda xato: {e}"
            )

    # ========================================================
    # TIMEFRAMES
    # ========================================================

    timeframe_map = {

        "M1": "M1",
        "M5": "M5",
        "M15": "M15",
        "M30": "M30",
        "H1": "H1",
        "H4": "H4",
        "D1": "D1",

    }

    # ========================================================
    # MARKET DATA
    # ========================================================

    print()

    ajratgich()

    print(
        "📊 KO'P TIMEFRAME BOZOR MA'LUMOTLARI"
    )

    ajratgich()

    mtf_data = {}

    for name, timeframe in timeframe_map.items():

        try:

            # MUHIM:
            # symbol bu yerda berilmaydi.
            # MarketData ichida self.symbol mavjud.

            df = market_data.get_candles(
                timeframe=timeframe,
                count=500
            )

            if df is None or df.empty:

                print(
                    f"❌ {name}: "
                    "ma'lumot mavjud emas."
                )

                continue

            mtf_data[name] = df

            last_closed = (
                df.iloc[-2]
                if len(df) >= 2
                else df.iloc[-1]
            )

            print(
                f"✅ {name}: "
                f"{len(df)} candle | "
                f"Close: "
                f"{xavfsiz_son(last_closed['close']):.2f}"
            )

        except Exception as e:

            print(
                f"❌ {name}: "
                f"ma'lumot olishda xato: {e}"
            )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()

    ajratgich()

    print(
        "🔎 MA'LUMOTLARNI TEKSHIRISH"
    )

    ajratgich()

    valid_count = 0

    for timeframe, df in mtf_data.items():

        try:

            valid = (
                market_data.validate_dataframe(
                    df
                )
            )

            if valid:

                valid_count += 1

                print(
                    f"✅ {timeframe}: YAROQLI"
                )

            else:

                print(
                    f"⚠️ {timeframe}: YAROQSIZ"
                )

        except Exception as e:

            print(
                f"⚠️ {timeframe}: "
                f"tekshiruv xatosi: {e}"
            )

    print()

    print(
        f"Natija: "
        f"{valid_count}/{len(timeframe_map)} "
        f"timeframe yaroqli."
    )

    if valid_count == 0:

        print()
        print(
            "❌ Bozor ma'lumotlari olinmadi."
        )

        client.shutdown()

        return

    # ========================================================
    # TECHNICAL INDICATORS
    # ========================================================

    print()

    ajratgich()

    print(
        "📐 TEXNIK INDIKATORLAR"
    )

    ajratgich()

    indicators = TechnicalIndicators()

    indicator_data = {}

    for timeframe, df in mtf_data.items():

        try:

            result = indicators.add_all(
                df.copy()
            )

            indicator_data[
                timeframe
            ] = result

            row = (
                result.iloc[-2]
                if len(result) >= 2
                else result.iloc[-1]
            )

            print(
                f"✅ {timeframe} | "
                f"EMA20: "
                f"{xavfsiz_son(row.get('ema_20')):.2f} | "
                f"EMA50: "
                f"{xavfsiz_son(row.get('ema_50')):.2f} | "
                f"EMA200: "
                f"{xavfsiz_son(row.get('ema_200')):.2f} | "
                f"RSI: "
                f"{xavfsiz_son(row.get('rsi_14')):.2f} | "
                f"ADX: "
                f"{xavfsiz_son(row.get('adx')):.2f}"
            )

        except Exception as e:

            print(
                f"❌ {timeframe}: "
                f"indikator xatosi: {e}"
            )

    # ========================================================
    # MTF TREND
    # ========================================================

    print()

    ajratgich()

    print(
        "🧭 MULTI-TIMEFRAME TREND"
    )

    ajratgich()

    mtf_trends = mtf_trendni_aniqlash(
        indicator_data
    )

    for timeframe in timeframe_map:

        trend = mtf_trends.get(
            timeframe,
            "UNKNOWN"
        )

        print(
            f"{timeframe:<4} : "
            f"{trend_uz(trend)}"
        )

    # ========================================================
    # MARKET STRUCTURE
    # ========================================================

    print()

    ajratgich()

    print(
        "🏗 MARKET STRUCTURE"
    )

    ajratgich()

    structure_engine = MarketStructure()

    structure_results = {}

    for timeframe, df in indicator_data.items():

        try:

            result = structure_engine.analyze(
                df
            )

            structure_results[
                timeframe
            ] = result

            trend = result.get(
                "trend",
                result.get(
                    "direction",
                    "UNKNOWN"
                )
            )

            swing_highs = result.get(
                "swing_highs",
                []
            )

            swing_lows = result.get(
                "swing_lows",
                []
            )

            events = result.get(
                "events",
                result.get(
                    "structure_events",
                    []
                )
            )

            print(
                f"✅ {timeframe} | "
                f"Trend: {trend_uz(trend)} | "
                f"Swing High: {len(swing_highs)} | "
                f"Swing Low: {len(swing_lows)} | "
                f"Events: {len(events)}"
            )

            if events:

                last_event = events[-1]

                event_name = (
                    qiymat_olish(
                        last_event,
                        "kind"
                    )
                    or qiymat_olish(
                        last_event,
                        "label"
                    )
                    or qiymat_olish(
                        last_event,
                        "type"
                    )
                    or qiymat_olish(
                        last_event,
                        "event"
                    )
                    or "UNKNOWN"
                )

                print(
                    f"    Oxirgi event: "
                    f"{event_name}"
                )

        except Exception as e:

            print(
                f"❌ {timeframe}: "
                f"market structure xatosi: {e}"
            )

    # ========================================================
    # TRENDLINE
    # ========================================================

    print()

    ajratgich()

    print(
        "📏 TRENDLINE ANALIZI"
    )

    ajratgich()

    trendline_engine = TrendLineEngine()

    trendline_results = {}

    if "M5" in indicator_data:

        try:

            m5_structure = (
                structure_results.get(
                    "M5",
                    {}
                )
            )

            m5_df = indicator_data[
                "M5"
            ]

            trendline_result = (
                trendline_engine.analyze(
                    m5_df,
                    m5_structure
                )
            )

            trendline_results[
                "M5"
            ] = trendline_result

            bullish_lines = (
                trendline_result.get(
                    "bullish_lines",
                    []
                )
            )

            bearish_lines = (
                trendline_result.get(
                    "bearish_lines",
                    []
                )
            )

            breakouts = (
                trendline_result.get(
                    "breakouts",
                    []
                )
            )

            retests = (
                trendline_result.get(
                    "retests",
                    []
                )
            )

            print(
                f"✅ M5 | "
                f"Bullish lines: "
                f"{len(bullish_lines)} | "
                f"Bearish lines: "
                f"{len(bearish_lines)} | "
                f"Breakout: "
                f"{len(breakouts)} | "
                f"Retest: "
                f"{len(retests)}"
            )

        except Exception as e:

            print(
                f"❌ M5 trendline xatosi: {e}"
            )

    # ========================================================
    # LIQUIDITY
    # ========================================================

    print()

    ajratgich()

    print(
        "💧 LIQUIDITY ANALIZI"
    )

    ajratgich()

    liquidity_engine = LiquidityEngine()

    liquidity_results = {}

    if "M5" in indicator_data:

        try:

            m5_df = indicator_data[
                "M5"
            ]

            liquidity_result = (
                liquidity_engine.analyze(
                    m5_df
                )
            )

            liquidity_results[
                "M5"
            ] = liquidity_result

            swing_highs = (
                liquidity_result.get(
                    "swing_highs",
                    []
                )
            )

            swing_lows = (
                liquidity_result.get(
                    "swing_lows",
                    []
                )
            )

            equal_highs = (
                liquidity_result.get(
                    "equal_highs",
                    []
                )
            )

            equal_lows = (
                liquidity_result.get(
                    "equal_lows",
                    []
                )
            )

            liquidity_levels = (
                liquidity_result.get(
                    "liquidity_levels",
                    []
                )
            )

            sweeps = (
                liquidity_result.get(
                    "sweeps",
                    []
                )
            )

            latest_sweep = (
                liquidity_result.get(
                    "latest_sweep"
                )
            )

            print(
                f"✅ M5 | "
                f"Swing High: "
                f"{len(swing_highs)} | "
                f"Swing Low: "
                f"{len(swing_lows)} | "
                f"Equal High: "
                f"{len(equal_highs)} | "
                f"Equal Low: "
                f"{len(equal_lows)} | "
                f"Liquidity: "
                f"{len(liquidity_levels)} | "
                f"Sweep: "
                f"{len(sweeps)}"
            )

            if latest_sweep:

                sweep_direction = (
                    qiymat_olish(
                        latest_sweep,
                        "direction",
                        "UNKNOWN"
                    )
                )

                print(
                    "    Oxirgi liquidity sweep: "
                    f"{yonalish_uz(sweep_direction)}"
                )

        except Exception as e:

            print(
                f"❌ M5 liquidity xatosi: {e}"
            )

    # ========================================================
    # FIBONACCI
    # ========================================================

    print()
    ajratgich()
    print("📐 FIBONACCI ANALIZI")
    ajratgich()

    fibonacci_results = {}

    if "M5" in indicator_data:
        try:
            fibonacci_engine = FibonacciEngine()
            m5_structure = structure_results.get("M5", {})
            m5_closed = oxirgi_yopilgan_candle(indicator_data["M5"])

            if m5_closed is None:
                raise ValueError("M5 oxirgi yopilgan candle topilmadi.")

            signal_price = xavfsiz_son(m5_closed.get("close"))
            structure_trend = m5_structure.get("trend", "ARALASH")

            if structure_trend == "KO_TARILISH":
                fib_direction = "BULLISH"
            elif structure_trend == "PASAYISH":
                fib_direction = "BEARISH"
            else:
                m5_mtf_trend = mtf_trends.get("M5", "UNKNOWN")
                if m5_mtf_trend == "UPTREND":
                    fib_direction = "BULLISH"
                elif m5_mtf_trend == "DOWNTREND":
                    fib_direction = "BEARISH"
                else:
                    fib_direction = None

            if fib_direction is None:
                print("⚪ Fibonacci yo'nalishi noaniq.")
            else:
                fibonacci_result = fibonacci_engine.analyze_from_structure(
                    current_price=signal_price,
                    structure_result=m5_structure,
                    direction=fib_direction,
                )

                fibonacci_results["M5"] = fibonacci_result
                direction = fibonacci_result["direction"]
                levels = fibonacci_result["levels"]

                print(f"Yo'nalish       : {yonalish_uz(direction)}")
                print(f"Swing High      : {fibonacci_result['swing_high']:.2f}")
                print(f"Swing Low       : {fibonacci_result['swing_low']:.2f}")
                print(f"Range           : {fibonacci_result['range_size']:.2f}")
                print()
                print(f"0.236           : {levels['0.236']:.2f}")
                print(f"0.382           : {levels['0.382']:.2f}")
                print(f"0.500           : {levels['0.500']:.2f}")
                print(f"0.618           : {levels['0.618']:.2f}")
                print(f"0.786           : {levels['0.786']:.2f}")
                print(f"1.000           : {levels['1.000']:.2f}")
                print(f"1.272           : {levels['1.272']:.2f}")
                print(f"1.618           : {levels['1.618']:.2f}")
                print()
                print(f"Joriy narx      : {signal_price:.2f}")
                print(f"Zona            : {fibonacci_result['zone']}")
                premium_text = "HA" if fibonacci_result["premium"] else "YO'Q"
                print(f"Premium         : {premium_text}")
                discount_text = "HA" if fibonacci_result["discount"] else "YO'Q"
                print(f"Discount        : {discount_text}")
                print()
                entry_zone = fibonacci_result["entry_zone"]
                print(f"Entry zona      : {entry_zone[0]:.2f} — {entry_zone[1]:.2f}")
                print(
                    f"Eng yaqin daraja: {fibonacci_result['nearest_level']} "
                    f"@ {fibonacci_result['nearest_level_price']:.2f}"
                )

        except Exception as e:
            print(f"❌ M5 Fibonacci xatosi: {e}")
    else:
        print("❌ M5 ma'lumotlari mavjud emas.")

    # ========================================================
    # CANDLE CONFIRMATION
    # ========================================================

    print()
    ajratgich()
    print("🕯 CANDLE CONFIRMATION")
    ajratgich()

    candle_confirmation_results = {}

    if "M5" in indicator_data:
        try:
            candle_engine = CandleConfirmationEngine()
            m5_candle_result = candle_engine.analyze_dict(
                indicator_data["M5"]
            )
            candle_confirmation_results["M5"] = m5_candle_result

            candle_direction = m5_candle_result.get("direction", "NEUTRAL")
            candle_score = xavfsiz_son(m5_candle_result.get("score", 0))
            candle_pattern = m5_candle_result.get("pattern", "NO_PATTERN")
            candle_strength = m5_candle_result.get("strength", "NONE")
            candle_confirmed = bool(m5_candle_result.get("confirmed", False))

            print(f"Yo'nalish       : {yonalish_uz(candle_direction)}")
            print(f"Pattern         : {candle_pattern}")
            print(f"Ball            : {candle_score:.1f}/100")
            print(f"Kuch            : {candle_strength}")

            tasdiq_matni = "HA" if candle_confirmed else "YO'Q"
            print(f"Tasdiq          : {tasdiq_matni}")

            reasons = m5_candle_result.get("reasons", [])
            if reasons:
                print()
                print("🧠 CANDLE SABABLARI")
                for reason in reasons:
                    print(f"  • {reason}")

        except Exception as e:
            print(f"❌ M5 Candle Confirmation xatosi: {e}")
    else:
        print("❌ M5 ma'lumotlari mavjud emas.")

    # ========================================================
    # SIGNAL ENGINE
    # ========================================================

    signal_result = None

    if "M5" in indicator_data:

        try:

            signal_engine = SignalEngine()

            signal_result = (
                signal_engine.analyze(
                    mtf_data=indicator_data,
                    mtf_trends=mtf_trends,
                    structure_results=structure_results,
                    trendline_results=trendline_results,
                    liquidity_results=liquidity_results,
                    fibonacci_results=fibonacci_results,
                    candle_results=candle_confirmation_results,
                )
            )

            m5_closed = (
                oxirgi_yopilgan_candle(
                    indicator_data["M5"]
                )
            )

            signal_price = (
                xavfsiz_son(
                    m5_closed.get(
                        "close"
                    )
                )
                if m5_closed is not None
                else current_price
            )

            # ====================================================
            # ENTRY / SL / TP OPTIMIZER
            # ====================================================

            try:
                signal_direction = yonalishni_normallashtir(
                    qiymat_olish(
                        signal_result,
                        "direction",
                        "NEUTRAL"
                    )
                )

                if signal_direction in ("BUY", "SELL"):
                    entry_exit_optimizer = EntryExitOptimizer(
                        atr_multiplier=1.20,
                        tp1_rr=1.50,
                        tp2_rr=2.50,
                        swing_lookback=50,
                    )

                    optimizer_result = entry_exit_optimizer.analyze(
                        df=indicator_data["M5"],
                        direction=signal_direction,
                        current_price=signal_price,
                        spread=spread,
                        digits=2,
                    )

                    # SignalResult obyektini Optimizer natijasi bilan yangilash.
                    if hasattr(signal_result, "entry"):
                        signal_result.entry = optimizer_result.entry
                    if hasattr(signal_result, "stop_loss"):
                        signal_result.stop_loss = optimizer_result.stop_loss
                    if hasattr(signal_result, "take_profit_1"):
                        signal_result.take_profit_1 = optimizer_result.take_profit_1
                    if hasattr(signal_result, "take_profit_2"):
                        signal_result.take_profit_2 = optimizer_result.take_profit_2
                    if hasattr(signal_result, "risk_reward_1"):
                        signal_result.risk_reward_1 = optimizer_result.risk_reward_1
                    if hasattr(signal_result, "risk_reward_2"):
                        signal_result.risk_reward_2 = optimizer_result.risk_reward_2

                    print()
                    ajratgich()
                    print("🎯 ENTRY / SL / TP OPTIMIZER")
                    ajratgich()

                    print(f"Yo'nalish       : {yonalish_uz(signal_direction)}")
                    print(f"Entry           : {optimizer_result.entry:.2f}")
                    print(f"Stop Loss       : {optimizer_result.stop_loss:.2f}")
                    print(f"TP1             : {optimizer_result.take_profit_1:.2f}")
                    print(f"TP2             : {optimizer_result.take_profit_2:.2f}")
                    print(f"Risk masofasi   : {optimizer_result.risk_distance:.2f}")
                    print(f"RR1             : {optimizer_result.risk_reward_1:.2f}")
                    print(f"RR2             : {optimizer_result.risk_reward_2:.2f}")
                    print(f"ATR             : {optimizer_result.atr:.2f}")

                    if optimizer_result.swing_high is not None:
                        print(f"Swing High      : {optimizer_result.swing_high:.2f}")
                    else:
                        print("Swing High      : N/A")

                    if optimizer_result.swing_low is not None:
                        print(f"Swing Low       : {optimizer_result.swing_low:.2f}")
                    else:
                        print("Swing Low       : N/A")

                    if optimizer_result.valid:
                        print("Holat           : ✅ PARAMETRLAR YAROQLI")
                    else:
                        print("Holat           : ⚠️ PARAMETRLAR TEKSHIRILSIN")

                    if optimizer_result.warnings:
                        print()
                        print("⚠️ OPTIMIZER OGOHLANTIRISHLARI")
                        for warning in optimizer_result.warnings:
                            print(f"  • {warning}")

                else:
                    print()
                    print(
                        "ℹ️ Signal NEUTRAL bo'lgani uchun "
                        "Entry/SL/TP hisoblanmadi."
                    )

            except Exception as e:
                print(f"❌ Entry/SL/TP Optimizer xatosi: {e}")

            # ====================================================
            # RISK MANAGER
            # ====================================================

            try:
                risk_direction = yonalishni_normallashtir(
                    qiymat_olish(signal_result, "direction", "NEUTRAL")
                )

                risk_status = str(
                    qiymat_olish(signal_result, "status", "NO_SIGNAL")
                ).upper().replace("-", "_").replace(" ", "_")

                if risk_direction in ("BUY", "SELL") and risk_status in (
                    "EARLY", "CONFIRMED", "HIGH_CONFIDENCE"
                ):
                    # Haqiqiy market entry: BUY -> Ask, SELL -> Bid.
                    live_tick = market_data.get_tick()
                    live_bid = xavfsiz_son(getattr(live_tick, "bid", 0)) if live_tick else bid
                    live_ask = xavfsiz_son(getattr(live_tick, "ask", 0)) if live_tick else ask
                    live_spread = max(live_ask - live_bid, 0.0)

                    market_entry = live_ask if risk_direction == "BUY" else live_bid
                    optimizer_sl = xavfsiz_son(
                        qiymat_olish(signal_result, "stop_loss", None),
                        0.0
                    )

                    symbol_info = mt5.symbol_info(symbol)
                    tick_size = xavfsiz_son(
                        getattr(symbol_info, "trade_tick_size", 0) if symbol_info else 0
                    )
                    tick_value = xavfsiz_son(
                        getattr(symbol_info, "trade_tick_value", 0) if symbol_info else 0
                    )

                    if tick_value <= 0 and symbol_info is not None:
                        tick_value = max(
                            xavfsiz_son(getattr(symbol_info, "trade_tick_value_profit", 0)),
                            xavfsiz_son(getattr(symbol_info, "trade_tick_value_loss", 0)),
                        )

                    risk_manager = RiskManager(
                        balance=model_balance,
                        risk_percent=risk_percent,
                        min_lot=xavfsiz_son(getattr(info, "volume_min", 0.01), 0.01),
                        max_lot=xavfsiz_son(getattr(info, "volume_max", 50.0), 50.0),
                        lot_step=xavfsiz_son(getattr(info, "volume_step", 0.01), 0.01),
                        max_spread=1.50,
                    )

                    risk_result = risk_manager.analyze(
                        direction=risk_direction,
                        entry=market_entry,
                        stop_loss=optimizer_sl,
                        tick_size=tick_size,
                        tick_value=tick_value,
                        spread=live_spread,
                    )

                    print()
                    ajratgich()
                    print("🛡 RISK MANAGER")
                    ajratgich()
                    print(f"Yo'nalish       : {yonalish_uz(risk_result.direction)}")
                    print(f"Model balans    : ${risk_result.balance:.2f}")
                    print(f"Risk            : {risk_result.risk_percent:.2f}%")
                    print(f"Maksimal risk   : ${risk_result.risk_amount:.2f}")
                    print(f"Market Entry    : {risk_result.entry:.2f}")
                    print(f"Stop Loss       : {risk_result.stop_loss:.2f}")
                    print(f"SL masofasi     : {risk_result.stop_distance:.2f}")
                    print(f"Tick Size       : {risk_result.tick_size:.5f}")
                    print(f"Tick Value      : {risk_result.tick_value:.5f}")
                    print(f"Raw Lot         : {risk_result.raw_lot:.6f}")
                    print(f"Final Lot       : {risk_result.lot:.2f}")
                    print(f"Taxminiy risk   : ${risk_result.estimated_risk:.2f}")
                    print(f"Live Bid        : {live_bid:.2f}")
                    print(f"Live Ask        : {live_ask:.2f}")
                    print(f"Spread          : {live_spread:.2f}")
                    print(f"Holat           : {'✅ RISK VALID' if risk_result.valid else '❌ RISK INVALID'}")

                    if risk_result.reasons:
                        print()
                        print("🧠 RISK SABABLARI")
                        for reason in risk_result.reasons:
                            print(f"  • {reason}")

                    if risk_result.warnings:
                        print()
                        print("⚠️ RISK OGOHLANTIRISHLARI")
                        for warning in risk_result.warnings:
                            print(f"  • {warning}")
                else:
                    print()
                    print("ℹ️ Risk Manager: hozircha hisoblashga signal yetarli emas.")
                    print("   Faqat EARLY / CONFIRMED / HIGH_CONFIDENCE signalda lot hisoblanadi.")

            except Exception as e:
                print(f"❌ Risk Manager xatosi: {e}")

            # ====================================================
            # PROFESSIONAL MARKET ENTRY
            # ====================================================
            try:
                actionable_status = str(
                    qiymat_olish(signal_result, "status", "NO_SIGNAL")
                ).upper().replace("-", "_").replace(" ", "_")

                if (
                    signal_direction in ("BUY", "SELL")
                    and actionable_status in ("EARLY", "CONFIRMED", "HIGH_CONFIDENCE")
                    and bid > 0
                    and ask > 0
                ):
                    market_entry = ask if signal_direction == "BUY" else bid
                    sl_value = xavfsiz_son(
                        qiymat_olish(signal_result, "stop_loss", None), 0.0
                    )
                    digits = int(getattr(info, "digits", 2))

                    if sl_value > 0:
                        risk_distance = abs(market_entry - sl_value)
                        if signal_direction == "BUY":
                            new_tp1 = market_entry + risk_distance * 1.50
                            new_tp2 = market_entry + risk_distance * 2.50
                        else:
                            new_tp1 = market_entry - risk_distance * 1.50
                            new_tp2 = market_entry - risk_distance * 2.50

                        if hasattr(signal_result, "entry"):
                            signal_result.entry = round(market_entry, digits)
                        if hasattr(signal_result, "take_profit_1"):
                            signal_result.take_profit_1 = round(new_tp1, digits)
                        if hasattr(signal_result, "take_profit_2"):
                            signal_result.take_profit_2 = round(new_tp2, digits)
                        if hasattr(signal_result, "risk_reward_1"):
                            signal_result.risk_reward_1 = 1.50
                        if hasattr(signal_result, "risk_reward_2"):
                            signal_result.risk_reward_2 = 2.50

                        print()
                        ajratgich()
                        print("💹 REAL MARKET ENTRY")
                        ajratgich()
                        print(f"Analiz Entry   : {signal_price:.{digits}f}")
                        print(
                            f"Market Entry   : {market_entry:.{digits}f} "
                            f"({'ASK' if signal_direction == 'BUY' else 'BID'})"
                        )
                        print(f"Stop Loss      : {sl_value:.{digits}f}")
                        print(f"TP1            : {new_tp1:.{digits}f}")
                        print(f"TP2            : {new_tp2:.{digits}f}")

            except Exception as e:
                print(f"⚠️ Market Entry xatosi: {type(e).__name__}: {e}")

            # ====================================================
            # PAPER TRADING
            # ====================================================
            paper_enabled = os.getenv("PAPER_TRADING", "0").lower() in {
                "1", "true", "yes", "on"
            }

            if paper_enabled:
                try:
                    print()
                    ajratgich()
                    print("🧪 PAPER TRADING")
                    ajratgich()

                    paper_engine = PaperTradingEngine(
                        initial_balance=model_balance,
                        risk_percent=risk_percent,
                        tp1_close_percent=50.0,
                    )

                    paper_status = str(
                        qiymat_olish(signal_result, "status", "NO_SIGNAL")
                    ).upper().replace("-", "_").replace(" ", "_")

                    paper_direction = yonalishni_normallashtir(
                        qiymat_olish(signal_result, "direction", "NEUTRAL")
                    )

                    paper_entry = xavfsiz_son(
                        qiymat_olish(signal_result, "entry", 0), 0.0
                    )
                    paper_sl = xavfsiz_son(
                        qiymat_olish(signal_result, "stop_loss", 0), 0.0
                    )
                    paper_tp1 = xavfsiz_son(
                        qiymat_olish(signal_result, "take_profit_1", 0), 0.0
                    )
                    paper_tp2 = xavfsiz_son(
                        qiymat_olish(signal_result, "take_profit_2", 0), 0.0
                    )

                    paper_lot = 0.0
                    if "risk_result" in locals() and risk_result is not None:
                        paper_lot = xavfsiz_son(
                            getattr(risk_result, "lot", 0.0), 0.0
                        )

                    paper_can_open = (
                        paper_status in ("EARLY", "CONFIRMED", "HIGH_CONFIDENCE")
                        and paper_direction in ("BUY", "SELL")
                        and paper_entry > 0
                        and paper_sl > 0
                        and paper_tp1 > 0
                        and paper_tp2 > 0
                        and paper_lot > 0
                        and "risk_result" in locals()
                        and bool(getattr(risk_result, "valid", False))
                    )

                    print(f"Rejim          : {'YOQILGAN' if paper_enabled else 'OCHIRILGAN'}")
                    print(f"Virtual balans : ${paper_engine.balance:.2f}")
                    print(f"Signal holati  : {holat_uz(paper_status)}")
                    print(f"Virtual lot    : {paper_lot:.2f}")

                    if paper_can_open:
                        opened = paper_engine.open_position(
                            direction=paper_direction,
                            entry=paper_entry,
                            stop_loss=paper_sl,
                            take_profit_1=paper_tp1,
                            take_profit_2=paper_tp2,
                            lot=paper_lot,
                        )

                        if opened:
                            print("✅ Virtual position ochildi.")
                            print(f"   Yo'nalish : {yonalish_uz(paper_direction)}")
                            print(f"   Entry     : {paper_entry:.2f}")
                            print(f"   SL        : {paper_sl:.2f}")
                            print(f"   TP1       : {paper_tp1:.2f}")
                            print(f"   TP2       : {paper_tp2:.2f}")
                            print(f"   Lot       : {paper_lot:.2f}")
                            print("ℹ️ Keyingi ishga tushirishda virtual pozitsiyani davom ettirish uchun state saqlash moduli alohida kerak.")
                        else:
                            print("❌ Virtual position ochilmadi.")
                    else:
                        print("ℹ️ Paper trade ochilmadi: signal yoki risk talablari yetarli emas.")
                        if "risk_result" in locals() and risk_result is not None:
                            print(
                                f"   Risk holati : {'VALID' if getattr(risk_result, 'valid', False) else 'INVALID'}"
                            )

                except Exception as e:
                    print(f"❌ Paper Trading xatosi: {type(e).__name__}: {e}")

            # ====================================================
            # AI ANALYST
            # ====================================================
            try:
                print()
                ajratgich()
                print("🤖 AI ANALYST")
                ajratgich()

                ai_analyst = AIAnalyst()
                ai_market_snapshot = {
                    "symbol": symbol,
                    "current_price": current_price,
                    "signal_price": signal_price,
                    "bid": bid if "bid" in locals() else None,
                    "ask": ask if "ask" in locals() else None,
                    "spread": spread if "spread" in locals() else None,
                    "mtf_trends": mtf_trends,
                }

                ai_result = ai_analyst.analyze(
                    signal_result=signal_result,
                    market_snapshot=ai_market_snapshot,
                )

                print(
                    "AI holati      : "
                    + ("YOQILGAN" if ai_result.enabled else "OFFLINE / FALLBACK")
                )
                print(f"AI yo'nalish   : {yonalish_uz(ai_result.direction)}")
                print(f"AI ishonch     : {ai_result.confidence:.1f}/100")
                print(f"AI qaror       : {ai_result.decision}")
                print(f"AI xulosa      : {ai_result.summary}")

                if ai_result.risks:
                    print()
                    print("⚠️ AI RISK LAR")
                    for risk in ai_result.risks:
                        print(f"  • {risk}")

            except Exception as e:
                print(f"❌ AI Analyst xatosi: {type(e).__name__}: {e}")

            # ====================================================
            # SIGNALNI CHIQARISH
            # ====================================================

            signalni_chiqarish(
                signal_result=signal_result,
                model_balance=model_balance,
                risk_percent=risk_percent,
                current_price=signal_price,
            )

        except TypeError as e:

            print(
                f"❌ Signal Engine argument xatosi: {e}"
            )

        except Exception as e:

            print(
                f"❌ Signal Engine xatosi: {e}"
            )

    else:

        print(
            "❌ M5 ma'lumotlari mavjud emas."
        )

    # ========================================================
    # BACKTEST ENGINE
    # ========================================================

    run_backtest = os.getenv("RUN_BACKTEST", "1").lower() in {
        "1", "true", "yes", "on"
    }

    if run_backtest and "M5" in mtf_data:
        try:
            print()
            ajratgich()
            print("📊 BACKTEST ENGINE")
            ajratgich()

            backtest_engine = BacktestEngine(
                initial_balance=model_balance,
                risk_percent=risk_percent,
                rr=1.50,
                min_score=60.0,
                warmup=250,
                max_bars=500,
            )

            backtest_result = backtest_engine.run(
                mtf_data["M5"].copy()
            )

            print(
                f"Boshlang'ich balans : ${backtest_result['initial_balance']:.2f}"
            )
            print(
                f"Yakuniy balans      : ${backtest_result['final_balance']:.2f}"
            )
            print(
                f"Sof natija          : ${backtest_result['net_profit']:.2f}"
            )
            print(
                f"Savdolar            : {backtest_result['trades']}"
            )
            print(
                f"Win                 : {backtest_result['wins']}"
            )
            print(
                f"Loss                : {backtest_result['losses']}"
            )
            print(
                f"Win Rate            : {backtest_result['win_rate']:.2f}%"
            )
            print(
                f"Jami R              : {backtest_result['total_r']:.3f}"
            )

            trade_log = backtest_result.get("trade_log", [])
            if trade_log:
                print()
                print("🧾 SO'NGGI BACKTEST SAVDOLARI")
                for trade in trade_log[-5:]:
                    print(
                        f"  {trade['direction']:<4} "
                        f"Entry={trade['entry']:.2f} "
                        f"Exit={trade['exit_price']:.2f} "
                        f"Natija={trade['result']:<5} "
                        f"R={trade['pnl_r']:+.2f}"
                    )
            else:
                print(
                    "ℹ️ Backtest shartlari bo'yicha savdo topilmadi."
                )

        except Exception as e:
            print(
                f"❌ Backtest Engine xatosi: {type(e).__name__}: {e}"
            )
    else:
        print()
        print("ℹ️ Backtest RUN_BACKTEST=0 orqali o'chirilgan.")

    # ========================================================
    # M5 CANDLE
    # ========================================================

    if "M5" in indicator_data:

        try:

            df = indicator_data[
                "M5"
            ]

            current_candle = (
                df.iloc[-1]
            )

            closed_candle = (
                df.iloc[-2]
                if len(df) >= 2
                else df.iloc[-1]
            )

            print()

            ajratgich()

            print(
                "🕯 M5 CANDLE HOLATI"
            )

            ajratgich()

            print(
                "⚠️ Current candle hali yopilmagan."
            )

            print(
                f"Current Open  : "
                f"{xavfsiz_son(current_candle['open']):.2f}"
            )

            print(
                f"Current High  : "
                f"{xavfsiz_son(current_candle['high']):.2f}"
            )

            print(
                f"Current Low   : "
                f"{xavfsiz_son(current_candle['low']):.2f}"
            )

            print(
                f"Current Close : "
                f"{xavfsiz_son(current_candle['close']):.2f}"
            )

            print()

            print(
                "✅ Signal hisoblashida "
                "oxirgi yopilgan candle ishlatiladi."
            )

            print(
                f"Closed Open   : "
                f"{xavfsiz_son(closed_candle['open']):.2f}"
            )

            print(
                f"Closed High   : "
                f"{xavfsiz_son(closed_candle['high']):.2f}"
            )

            print(
                f"Closed Low    : "
                f"{xavfsiz_son(closed_candle['low']):.2f}"
            )

            print(
                f"Closed Close  : "
                f"{xavfsiz_son(closed_candle['close']):.2f}"
            )

        except Exception as e:

            print(
                f"⚠️ Candle ma'lumotini chiqarishda xato: {e}"
            )

    # ========================================================
    # YAKUNIY HOLAT
    # ========================================================

    print()

    ajratgich()

    print(
        "📌 YAKUNIY HOLAT"
    )

    ajratgich()

    if signal_result is not None:

        direction = yonalishni_normallashtir(
            qiymat_olish(
                signal_result,
                "direction"
            )
        )

        score = xavfsiz_son(
            qiymat_olish(
                signal_result,
                "score",
                0
            )
        )

        status = qiymat_olish(
            signal_result,
            "status",
            "NO_SIGNAL"
        )

        print(
            f"Yo'nalish : "
            f"{yonalish_uz(direction)}"
        )

        print(
            f"Ball      : "
            f"{score:.1f}/100"
        )

        print(
            f"Holat     : "
            f"{holat_uz(status)}"
        )

        if str(status).upper() in (
            "EARLY",
            "CONFIRMED",
            "HIGH_CONFIDENCE",
        ):

            print()

            print(
                "ℹ️ Signal mavjud."
            )

            print(
                "ℹ️ Haqiqiy order yuborish "
                "hozircha O'CHIRILGAN."
            )

        else:

            print()

            print(
                "ℹ️ Hozircha savdo uchun "
                "yetarli tasdiq yo'q."
            )

    else:

        print(
            "⚪ Signal natijasi mavjud emas."
        )

    # ========================================================
    # KEYINGI MODULLAR
    # ========================================================

    print()

    ajratgich()

    print(
        "🧩 KEYINGI PROFESSIONAL MODULLAR"
    )

    ajratgich()

    print(
        "✅ Fibonacci"
    )

    print(
        "✅ Candle Confirmation"
    )

    print(
        "✅ Entry / Stop Loss / Take Profit optimizatsiyasi"
    )

    print(
        "✅ Risk Manager"
    )

    print(
        "✅ AI Analyst"
    )

    print(
        "✅ Backtest Engine"
    )

    print(
        "✅ Paper Trading"
    )

    print(
        "⏳ Live Execution"
    )

    print()

    ajratgich()

    print(
        "✅ ANALYSIS YAKUNLANDI"
    )

    ajratgich()

    # ========================================================
    # MT5 SHUTDOWN
    # ========================================================

    try:

        client.shutdown()

    except Exception:

        try:
            mt5.shutdown()
        except Exception:
            pass


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        signal_only = os.getenv("SIGNAL_ONLY", "1").strip().lower() in {
            "1", "true", "yes", "on"
        }

        if signal_only:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                main()
            qisqa_signalni_chiqarish(buffer.getvalue())
        else:
            main()

    except KeyboardInterrupt:

        print()
        print(
            "🛑 Bot foydalanuvchi tomonidan to'xtatildi."
        )

        try:
            mt5.shutdown()
        except Exception:
            pass

        sys.exit(0)

    except Exception as e:

        print()

        ajratgich()

        print(
            "❌ KUTILMAGAN XATO"
        )

        ajratgich()

        print(
            f"{type(e).__name__}: {e}"
        )

        print()

        try:
            mt5.shutdown()
        except Exception:
            pass

        sys.exit(1)
