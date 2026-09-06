"""Yahoo Finance (.KA) historical stock price collector implementation."""

import datetime
from typing import Any

import pandas as pd
from loguru import logger

from psx_predictor.collectors.base import (
    BaseCollector,
    CollectorError,
    SymbolNotFoundError,
)
from psx_predictor.config.loader import load_config


class YahooCollector(BaseCollector):
    """Collector fetching PSX historical daily prices from Yahoo Finance (.KA suffix)."""

    def __init__(self, ticker_suffix: str = ".KA") -> None:
        self.ticker_suffix = ticker_suffix

    @property
    def source_name(self) -> str:
        return "yahoo"

    def resolve_yahoo_symbol(self, symbol: str) -> str:
        """Resolve standard PSX symbol to Yahoo Finance ticker string."""
        clean_sym = symbol.strip().upper()
        if clean_sym.endswith(self.ticker_suffix):
            return clean_sym

        # Check configured stocks for explicit mapping
        try:
            cfg = load_config()
            for s in cfg.stocks:
                if s.symbol.upper() == clean_sym and s.yahoo_ticker:
                    return s.yahoo_ticker
        except Exception:
            pass

        return f"{clean_sym}{self.ticker_suffix}"

    def fetch_raw(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> pd.DataFrame:
        """Fetch historical dataframe from Yahoo Finance via yfinance."""
        import yfinance as yf

        yahoo_symbol = self.resolve_yahoo_symbol(symbol)
        logger.debug(f"[yahoo] Fetching {yahoo_symbol} from {start_date} to {end_date}")

        # yfinance end_date is exclusive, add 1 day for inclusive end date
        end_inclusive = end_date + datetime.timedelta(days=1)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_inclusive.strftime("%Y-%m-%d")

        try:
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(
                start=start_str,
                end=end_str,
                auto_adjust=False,
                actions=True,
            )
        except Exception as e:
            raise CollectorError(f"Error requesting Yahoo Finance for {symbol}: {e}") from e

        if df is None or df.empty:
            raise SymbolNotFoundError(
                f"No data returned by Yahoo Finance for ticker '{yahoo_symbol}'"
            )

        return df

    def normalize(self, raw_data: Any, symbol: str) -> pd.DataFrame:
        """Normalize yfinance historical DataFrame into standard OHLCV schema."""
        if not isinstance(raw_data, pd.DataFrame) or raw_data.empty:
            return pd.DataFrame()

        df = raw_data.copy()

        # Reset index to access Date
        if "Date" not in df.columns:
            if "date" in df.columns:
                df = df.rename(columns={"date": "Date"})
            else:
                df = df.reset_index()
                if "Date" not in df.columns:
                    df = df.rename(columns={df.columns[0]: "Date"})

        # Check required columns
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                raise CollectorError(f"Yahoo data missing expected column: '{col}'")

        # Extract date from datetime index / column
        trade_dates = []
        for d in df["Date"]:
            if isinstance(d, datetime.date) and not isinstance(d, datetime.datetime):
                trade_dates.append(d)
            elif hasattr(d, "date"):
                trade_dates.append(d.date())
            else:
                trade_dates.append(datetime.date.fromisoformat(str(d)[:10]))

        adj_close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        div_series = df["Dividends"].fillna(0.0).astype(float) if "Dividends" in df.columns else 0.0
        split_series = (
            df["Stock Splits"].replace(0, 1.0).fillna(1.0).astype(float)
            if "Stock Splits" in df.columns
            else 1.0
        )

        normalized = pd.DataFrame(
            {
                "symbol": symbol.upper(),
                "trade_date": trade_dates,
                "open": df["Open"].astype(float),
                "high": df["High"].astype(float),
                "low": df["Low"].astype(float),
                "close": df["Close"].astype(float),
                "adjusted_close": adj_close.astype(float),
                "volume": df["Volume"].fillna(0).astype(int),
                "dividend_amount": div_series,
                "split_ratio": split_series,
                "is_upper_lock": False,
                "is_lower_lock": False,
            }
        )

        # Remove zero/negative prices
        valid_mask = (
            (normalized["open"] > 0)
            & (normalized["high"] > 0)
            & (normalized["low"] > 0)
            & (normalized["close"] > 0)
            & (normalized["adjusted_close"] > 0)
        )
        normalized = normalized[valid_mask].reset_index(drop=True)
        return normalized
