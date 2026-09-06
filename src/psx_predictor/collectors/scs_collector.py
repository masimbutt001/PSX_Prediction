"""SCS Trade (scstrade.com) historical stock price collector implementation."""

import datetime
import json
import re
import time
from typing import Any, Optional

import httpx
import pandas as pd
from loguru import logger

from psx_predictor.collectors.base import (
    BaseCollector,
    CollectorError,
    EmptyResponseError,
    NetworkTimeoutError,
    RateLimitError,
    SymbolNotFoundError,
)


class SCSTradeCollector(BaseCollector):
    """Collector querying Standard Capital Securities (SCS Trade) historical prices API."""

    ENDPOINT_URL = "https://www.scstrade.com/MarketStatistics/MS_HistoricalPrices.aspx/chart"
    DEFAULT_TIMEOUT = 15.0
    MAX_RETRIES = 3

    DEFAULT_HEADERS = {
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.scstrade.com/MarketStatistics/MS_HistoricalPrices.aspx",
        "X-Requested-With": "XMLHttpRequest",
    }

    def __init__(
        self,
        client: Optional[httpx.Client] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._client = client
        self.timeout = timeout

    @property
    def source_name(self) -> str:
        return "scs"

    @staticmethod
    def parse_ms_date(ms_date_str: str) -> datetime.date:
        """Parse Microsoft AJAX JSON date format like '/Date(1705258800000)/' into a date."""
        match = re.search(r"/Date\((\d+)(?:[+-]\d+)?\)/", str(ms_date_str))
        if not match:
            # Fallback for ISO strings or direct date strings
            try:
                return datetime.date.fromisoformat(str(ms_date_str)[:10])
            except ValueError as e:
                raise ValueError(f"Unable to parse date string: {ms_date_str}") from e

        timestamp_ms = int(match.group(1))
        # Convert UTC timestamp in milliseconds to date
        dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000.0, datetime.timezone.utc)
        return dt.date()

    def fetch_raw(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> list[dict[str, Any]]:
        """Fetch raw JSON array from SCS Trade endpoint with exponential backoff retries."""
        date1_str = start_date.strftime("%d-%b-%Y")
        date2_str = end_date.strftime("%d-%b-%Y")
        payload = {
            "par": symbol.upper(),
            "date1": date1_str,
            "date2": date2_str,
        }

        should_close = False
        client = self._client
        if client is None:
            client = httpx.Client(timeout=self.timeout, headers=self.DEFAULT_HEADERS)
            should_close = True

        last_err: Optional[Exception] = None
        try:
            for attempt in range(1, self.MAX_RETRIES + 1):
                try:
                    logger.debug(
                        f"[scs] Sending POST to {self.ENDPOINT_URL} for {symbol} "
                        f"(attempt {attempt}/{self.MAX_RETRIES})"
                    )
                    resp = client.post(
                        self.ENDPOINT_URL,
                        json=payload,
                        headers=self.DEFAULT_HEADERS,
                    )

                    if resp.status_code == 429:
                        raise RateLimitError(f"SCS Trade rate limit hit for {symbol}")
                    if resp.status_code == 404:
                        raise SymbolNotFoundError(f"Symbol '{symbol}' not found on SCS Trade")
                    resp.raise_for_status()

                    data = resp.json()
                    records = data.get("d")
                    if records is None:
                        raise EmptyResponseError(
                            f"SCS Trade returned unexpected response: {resp.text[:200]}"
                        )
                    if not isinstance(records, list):
                        raise CollectorError(f"Expected list in 'd' field, got: {type(records)}")

                    logger.debug(f"[scs] Received {len(records)} raw records for {symbol}")
                    return records

                except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                    last_err = NetworkTimeoutError(
                        f"Timeout requesting SCS Trade for {symbol}: {e}"
                    )
                except (httpx.ConnectError, httpx.HTTPStatusError) as e:
                    last_err = CollectorError(f"HTTP error requesting SCS Trade for {symbol}: {e}")
                except json.JSONDecodeError as e:
                    last_err = CollectorError(f"Failed to decode JSON from SCS Trade: {e}")

                # Sleep with exponential backoff before retry if not last attempt
                if attempt < self.MAX_RETRIES:
                    backoff = 1.0 * (2 ** (attempt - 1))
                    logger.warning(
                        f"[scs] Attempt {attempt} failed ({last_err}). Retry in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)

            raise last_err or CollectorError(f"Failed to fetch data from SCS Trade for {symbol}")
        finally:
            if should_close:
                client.close()

    def normalize(self, raw_data: Any, symbol: str) -> pd.DataFrame:
        """Normalize SCS Trade records into OHLCVRecord DataFrame."""
        if not raw_data or not isinstance(raw_data, list):
            return pd.DataFrame()

        rows = []
        for item in raw_data:
            try:
                date_val = self.parse_ms_date(item["trading_Date"])
                open_val = float(item["trading_open"])
                high_val = float(item["trading_high"])
                low_val = float(item["trading_low"])
                close_val = float(item["trading_close"])
                vol_val = int(round(float(item.get("trading_vol", 0))))

                # Skip completely zero price rows (e.g. non-trading dates)
                if open_val <= 0 or high_val <= 0 or low_val <= 0 or close_val <= 0:
                    continue

                rows.append(
                    {
                        "symbol": symbol.upper(),
                        "trade_date": date_val,
                        "open": open_val,
                        "high": high_val,
                        "low": low_val,
                        "close": close_val,
                        "adjusted_close": close_val,  # Base adjusted to close initially
                        "volume": max(0, vol_val),
                        "dividend_amount": 0.0,
                        "split_ratio": 1.0,
                        "is_upper_lock": False,
                        "is_lower_lock": False,
                    }
                )
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"[scs] Skipped invalid item: {item} - reason: {e}")
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Sort chronologically
        df = df.sort_values(by="trade_date").reset_index(drop=True)
        return df
