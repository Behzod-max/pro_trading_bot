from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    symbol: str = os.getenv("TRADING_SYMBOL", "XAUUSD")
    timeframe: str = os.getenv("TRADING_TIMEFRAME", "M5")

    account_balance: float = float(
        os.getenv("ACCOUNT_BALANCE", "50")
    )

    risk_per_trade: float = float(
        os.getenv("RISK_PER_TRADE", "0.01")
    )

    max_daily_loss: float = float(
        os.getenv("MAX_DAILY_LOSS", "0.03")
    )

    min_signal_score: int = int(
        os.getenv("MIN_SIGNAL_SCORE", "75")
    )

    trading_mode: str = os.getenv(
        "TRADING_MODE",
        "analysis"
    )


settings = Settings()