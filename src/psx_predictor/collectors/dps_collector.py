"""PSX Data Portal (dps.psx.com.pk) official exchange collector & scraper implementation."""

import datetime
import time
from typing import Any, Optional

import httpx
import pandas as pd
from bs4 import BeautifulSoup, Tag
from loguru import logger

from psx_predictor.collectors.base import (
    BaseCollector,
    CollectorError,
    NetworkTimeoutError,
    RateLimitError,
    SymbolNotFoundError,
)


class DPSCollector(BaseCollector):
    """Collector fetching official historical data from the PSX Data Portal (DPS)."""

    TIMESERIES_URL = "https://dps.psx.com.pk/timeseries/eod/{symbol}"
    HISTORICAL_TABLE_URL = "https://dps.psx.com.pk/historical"
    DEFAULT_TIMEOUT = 15.0
    MAX_RETRIES = 3

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://dps.psx.com.pk/",
        "Accept": "application/json, text/html, */*",
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
        return "dps"

    def fetch_raw(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> Any:
        """Fetch raw data from PSX Data Portal timeseries API or HTML table fallback."""
        clean_sym = symbol.strip().upper()
        url = self.TIMESERIES_URL.format(symbol=clean_sym)

        should_close = False
        client = self._client
        if client is None:
            client = httpx.Client(timeout=self.timeout, headers=self.DEFAULT_HEADERS)
            should_close = True

        last_err: Optional[Exception] = None
        try:
            # 1. Attempt Timeseries EOD JSON endpoint
            for attempt in range(1, self.MAX_RETRIES + 1):
                try:
                    logger.debug(f"[dps] Querying timeseries endpoint {url} (attempt {attempt})")
                    resp = client.get(url, headers=self.DEFAULT_HEADERS)

                    if resp.status_code == 404:
                        raise SymbolNotFoundError(f"Symbol '{clean_sym}' not found on PSX DPS")
                    if resp.status_code == 429:
                        raise RateLimitError("PSX DPS rate limit encountered")
                    resp.raise_for_status()

                    # Check if response is JSON
                    content_type = resp.headers.get("content-type", "")
                    if "application/json" in content_type or resp.text.strip().startswith(
                        ("{", "[")
                    ):
                        data = resp.json()
                        if data:
                            return {"type": "timeseries_json", "payload": data}

                except (httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                    last_err = NetworkTimeoutError(f"DPS request timeout for {clean_sym}: {e}")
                except (httpx.ConnectError, httpx.HTTPStatusError) as e:
                    last_err = CollectorError(f"DPS HTTP error for {clean_sym}: {e}")
                except Exception as e:
                    last_err = CollectorError(f"DPS error for {clean_sym}: {e}")

                if attempt < self.MAX_RETRIES:
                    backoff = 1.0 * (2 ** (attempt - 1))
                    time.sleep(backoff)

            # 2. Fallback to HTML Historical Page if JSON fails
            logger.warning(
                f"[dps] Timeseries failed for {clean_sym}. Trying HTML table fallback..."
            )
            try:
                hist_resp = client.get(
                    self.HISTORICAL_TABLE_URL,
                    params={"symbol": clean_sym},
                    headers=self.DEFAULT_HEADERS,
                )
                if hist_resp.is_success and "<table" in hist_resp.text:
                    return {"type": "html_table", "payload": hist_resp.text}
            except Exception as e:
                logger.warning(f"[dps] HTML fallback failed: {e}")

            raise last_err or CollectorError(f"Unable to retrieve DPS data for {clean_sym}")

        finally:
            if should_close:
                client.close()

    def normalize(self, raw_data: Any, symbol: str) -> pd.DataFrame:
        """Normalize DPS timeseries JSON or HTML table into standard OHLCV schema."""
        if not isinstance(raw_data, dict):
            return pd.DataFrame()

        payload_type = raw_data.get("type")
        payload = raw_data.get("payload")

        if payload_type == "timeseries_json":
            return self._normalize_timeseries_json(payload, symbol)
        elif payload_type == "html_table":
            return self._normalize_html_table(str(payload), symbol)

        return pd.DataFrame()

    def _normalize_timeseries_json(self, payload: Any, symbol: str) -> pd.DataFrame:
        """Parse DPS timeseries JSON format.

        Format is typically:
        [[timestamp_seconds, open, high, low, close, volume], ...]
        or {"data": [[...], ...]}
        """
        items = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(items, list) or not items:
            return pd.DataFrame()

        rows = []
        for row in items:
            try:
                # Row as array: [ts, open, high, low, close, volume]
                if isinstance(row, (list, tuple)) and len(row) >= 5:
                    raw_ts = float(row[0])
                    sec_ts = raw_ts / 1000.0 if raw_ts > 1e11 else raw_ts
                    date_val = datetime.datetime.fromtimestamp(sec_ts, datetime.timezone.utc).date()

                    open_val = float(row[1])
                    high_val = float(row[2])
                    low_val = float(row[3])
                    close_val = float(row[4])
                    vol_val = int(row[5]) if len(row) > 5 else 0

                # Row as object: {"time": ts, "open": ..., ...}
                elif isinstance(row, dict):
                    raw_ts = float(row.get("time") or row.get("date") or 0)
                    sec_ts = raw_ts / 1000.0 if raw_ts > 1e11 else raw_ts
                    date_val = datetime.datetime.fromtimestamp(sec_ts, datetime.timezone.utc).date()
                    open_val = float(row["open"])
                    high_val = float(row["high"])
                    low_val = float(row["low"])
                    close_val = float(row["close"])
                    vol_val = int(row.get("volume", 0))
                else:
                    continue

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
                        "adjusted_close": close_val,
                        "volume": max(0, vol_val),
                        "dividend_amount": 0.0,
                        "split_ratio": 1.0,
                        "is_upper_lock": False,
                        "is_lower_lock": False,
                    }
                )
            except Exception as e:
                logger.debug(f"[dps] Skipped invalid timeseries entry: {e}")
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values(by="trade_date").reset_index(drop=True)
        return df

    def _normalize_html_table(self, html_text: str, symbol: str) -> pd.DataFrame:
        """Parse historical data table from PSX DPS HTML page."""
        soup = BeautifulSoup(html_text, "html.parser")
        table = soup.find("table")
        if not isinstance(table, Tag):
            return pd.DataFrame()

        rows = []
        for tr in table.find_all("tr")[1:]:  # Skip header row
            if not isinstance(tr, Tag):
                continue
            tds = tr.find_all("td")
            if len(tds) < 5:
                continue

            try:
                date_str = tds[0].get_text(strip=True)
                date_val = datetime.date.fromisoformat(date_str)
                open_val = float(tds[1].get_text(strip=True).replace(",", ""))
                high_val = float(tds[2].get_text(strip=True).replace(",", ""))
                low_val = float(tds[3].get_text(strip=True).replace(",", ""))
                close_val = float(tds[4].get_text(strip=True).replace(",", ""))
                vol_val = int(tds[5].get_text(strip=True).replace(",", "")) if len(tds) > 5 else 0

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
                        "adjusted_close": close_val,
                        "volume": max(0, vol_val),
                        "dividend_amount": 0.0,
                        "split_ratio": 1.0,
                        "is_upper_lock": False,
                        "is_lower_lock": False,
                    }
                )
            except Exception as e:
                logger.debug(f"[dps] Skipped invalid HTML row: {e}")
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values(by="trade_date").reset_index(drop=True)
        return df
