"""Collector for market-traded macroeconomic indicators: USD/PKR, Brent crude, and KSE-100."""

import datetime
import re
from typing import Optional

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, Tag
from loguru import logger

from psx_predictor.macro.schemas import MacroDataPoint, MacroIndicatorType
from psx_predictor.news.calendar import PSXMarketCalendar

_REG_MARKET_STATE = re.compile(
    r"markets__item__title[^>]*>\s*Regular\s*</div>.*?State</div>\s*<div>\s*([^<]+?)\s*<",
    re.S,
)


def _number(text: str) -> Optional[float]:
    match = re.search(r"-?[\d,]+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def parse_dps_home_kse100(
    html_text: str, calendar: Optional[PSXMarketCalendar] = None
) -> list[MacroDataPoint]:
    """Parse KSE-100 closes from the PSX Data Portal home page.

    The KSE-100 panel shows a snapshot time, the current value, the previous close and the
    day's volume. The previous close always belongs to the trading day before the snapshot
    day. The current value counts as that day's close only once the day has traded
    (volume above zero) and the regular market is no longer open.
    """
    cal = calendar or PSXMarketCalendar()
    soup = BeautifulSoup(html_text, "html.parser")
    panel = soup.find(
        lambda tag: tag.name == "div"
        and tag.get("data-name") == MarketMacroCollector.KSE100_DPS_SYMBOL
        and "marketIndices__details" in (tag.get("class") or [])
    )
    if not isinstance(panel, Tag):
        return []

    snapshot = str(panel.get("data-date") or "").strip()
    try:
        snapshot_day = datetime.date.fromisoformat(snapshot[:10])
    except ValueError:
        return []

    stats: dict[str, Optional[float]] = {}
    for item in panel.find_all("div", class_="stats_item"):
        label = item.find("div", class_="stats_label")
        value = item.find("div", class_="stats_value")
        if isinstance(label, Tag) and isinstance(value, Tag):
            stats[label.get_text(strip=True).lower()] = _number(value.get_text(" ", strip=True))

    source = "PSX Data Portal home page (KSE100)"
    points: list[MacroDataPoint] = []
    prev_close = stats.get("previous close")
    if prev_close and prev_close > 0:
        prev_day = cal.previous_trading_day(snapshot_day)
        points.append(
            MarketMacroCollector._kse100_point(prev_day.isoformat(), prev_close, source)
        )

    current = _number(str(panel.get("data-close") or ""))
    volume = stats.get("volume") or 0.0
    state_match = _REG_MARKET_STATE.search(html_text)
    state = state_match.group(1).strip().lower() if state_match else ""
    market_open = state in {"open", "pre-open", "pre open"}
    if current and current > 0 and volume > 0 and not market_open:
        if cal.is_trading_day(snapshot_day):
            points.append(
                MarketMacroCollector._kse100_point(snapshot_day.isoformat(), current, source)
            )
    return points


class MarketMacroCollector:
    """Collects daily traded macro series: USD/PKR, Brent Crude Oil, and KSE-100 Index."""

    def __init__(self, fallback_on_error: bool = False) -> None:
        """Create the collector.

        Args:
            fallback_on_error: If True, generate deterministic synthetic series when Yahoo
                Finance returns nothing. Only meant for offline demos and tests: synthetic
                values are not real market data and must never feed live features.
        """
        self.fallback_on_error = fallback_on_error

    def fetch_usd_pkr(
        self,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
    ) -> list[MacroDataPoint]:
        """Fetch historical USD/PKR interbank spot exchange rate."""
        start = start_date or datetime.date(2024, 1, 1)
        end = end_date or datetime.date.today()
        points: list[MacroDataPoint] = []

        try:
            import yfinance as yf

            end_excl = end + datetime.timedelta(days=1)
            ticker = yf.Ticker("PKR=X")
            df = ticker.history(
                start=start.isoformat(),
                end=end_excl.isoformat(),
                auto_adjust=False,
            )
            if not df.empty and "Close" in df.columns:
                close_series = df["Close"].dropna()
                for dt_idx, val in close_series.items():
                    dt_str = pd.to_datetime(dt_idx).strftime("%Y-%m-%d")
                    f_val = float(val.iloc[0]) if hasattr(val, "iloc") else float(val)
                    if not np.isnan(f_val) and f_val > 0:
                        points.append(
                            MacroDataPoint(
                                indicator=MacroIndicatorType.USD_PKR.value,
                                observation_date=dt_str,
                                public_release_date=dt_str,
                                value=round(f_val, 4),
                                unit="PKR",
                                frequency="DAILY",
                                source="Yahoo Finance (PKR=X)",
                                notes="Interbank USD/PKR spot closing exchange rate",
                            )
                        )
        except Exception as e:
            logger.warning(f"Could not fetch USD/PKR from Yahoo Finance: {e}")

        if not points and self.fallback_on_error:
            logger.info("Generating deterministic fallback seed data for USD/PKR.")
            points = self._generate_usd_pkr_seed(start, end)

        return points

    def fetch_brent_crude(
        self,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
    ) -> list[MacroDataPoint]:
        """Fetch historical Brent Crude Oil continuous futures (BZ=F)."""
        start = start_date or datetime.date(2024, 1, 1)
        end = end_date or datetime.date.today()
        points: list[MacroDataPoint] = []

        try:
            import yfinance as yf

            end_excl = end + datetime.timedelta(days=1)
            ticker = yf.Ticker("BZ=F")
            df = ticker.history(
                start=start.isoformat(),
                end=end_excl.isoformat(),
                auto_adjust=False,
            )
            if not df.empty and "Close" in df.columns:
                close_series = df["Close"].dropna()
                for dt_idx, val in close_series.items():
                    dt_str = pd.to_datetime(dt_idx).strftime("%Y-%m-%d")
                    f_val = float(val.iloc[0]) if hasattr(val, "iloc") else float(val)
                    if not np.isnan(f_val) and f_val > 0:
                        points.append(
                            MacroDataPoint(
                                indicator=MacroIndicatorType.BRENT_CRUDE.value,
                                observation_date=dt_str,
                                public_release_date=dt_str,
                                value=round(f_val, 2),
                                unit="USD_PER_BBL",
                                frequency="DAILY",
                                source="Yahoo Finance (BZ=F)",
                                notes="Brent Crude Oil ICE continuous settlement",
                            )
                        )
        except Exception as e:
            logger.warning(f"Could not fetch Brent Crude from Yahoo Finance: {e}")

        if not points and self.fallback_on_error:
            logger.info("Generating deterministic fallback seed data for Brent Crude.")
            points = self._generate_brent_seed(start, end)

        return points

    # Yahoo symbols tried in order for the KSE-100 index; Yahoo has listed it under both.
    KSE100_YAHOO_TICKERS = ("^KSE100", "^KSE")
    # Symbol of the KSE-100 index on the PSX Data Portal.
    KSE100_DPS_SYMBOL = "KSE100"
    DPS_HOME_URL = "https://dps.psx.com.pk/"
    BROWSER_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    def fetch_kse100(
        self,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
    ) -> list[MacroDataPoint]:
        """Fetch KSE-100 index closes: Yahoo Finance first, then the PSX Data Portal."""
        start = start_date or datetime.date(2024, 1, 1)
        end = end_date or datetime.date.today()

        for yahoo_symbol in self.KSE100_YAHOO_TICKERS:
            points = self._kse100_from_yahoo(yahoo_symbol, start, end)
            if points:
                return points

        points = self._kse100_from_dps(start, end)
        if points:
            return points

        points = self._kse100_from_dps_home(start, end)
        if points:
            return points

        if self.fallback_on_error:
            logger.info("Generating deterministic fallback seed data for KSE-100.")
            return self._generate_kse100_seed(start, end)
        return []

    @staticmethod
    def _kse100_point(day: str, value: float, source: str) -> MacroDataPoint:
        return MacroDataPoint(
            indicator=MacroIndicatorType.KSE_100.value,
            observation_date=day,
            public_release_date=day,
            value=round(value, 2),
            unit="POINTS",
            frequency="DAILY",
            source=source,
            notes="PSX KSE-100 Benchmark Index close",
        )

    def _kse100_from_yahoo(
        self, yahoo_symbol: str, start: datetime.date, end: datetime.date
    ) -> list[MacroDataPoint]:
        points: list[MacroDataPoint] = []
        try:
            import yfinance as yf

            end_excl = end + datetime.timedelta(days=1)
            df = yf.Ticker(yahoo_symbol).history(
                start=start.isoformat(),
                end=end_excl.isoformat(),
                auto_adjust=False,
            )
            if not df.empty and "Close" in df.columns:
                for dt_idx, val in df["Close"].dropna().items():
                    f_val = float(val.iloc[0]) if hasattr(val, "iloc") else float(val)
                    if not np.isnan(f_val) and f_val > 0:
                        day = pd.to_datetime(dt_idx).strftime("%Y-%m-%d")
                        points.append(
                            self._kse100_point(day, f_val, f"Yahoo Finance ({yahoo_symbol})")
                        )
        except Exception as e:
            logger.warning(f"Could not fetch KSE-100 from Yahoo Finance ({yahoo_symbol}): {e}")
        if not points:
            logger.info(f"Yahoo Finance returned no KSE-100 data for {yahoo_symbol}.")
        return points

    def _kse100_from_dps(
        self, start: datetime.date, end: datetime.date
    ) -> list[MacroDataPoint]:
        try:
            from psx_predictor.collectors.dps_collector import DPSCollector

            collector = DPSCollector()
            df = collector.normalize(
                collector.fetch_raw(self.KSE100_DPS_SYMBOL, start, end), self.KSE100_DPS_SYMBOL
            )
        except Exception as e:
            logger.warning(f"Could not fetch KSE-100 from PSX Data Portal: {e}")
            return []
        if df.empty or "close" not in df.columns:
            return []
        dates = pd.to_datetime(df["trade_date"]).dt.date
        df = df[(dates >= start) & (dates <= end)]
        return [
            self._kse100_point(
                pd.Timestamp(day).strftime("%Y-%m-%d"), float(close), "PSX Data Portal (KSE100)"
            )
            for day, close in zip(df["trade_date"], df["close"], strict=True)
            if pd.notna(close) and float(close) > 0
        ]

    def _kse100_from_dps_home(
        self, start: datetime.date, end: datetime.date
    ) -> list[MacroDataPoint]:
        """Read the latest KSE-100 closes shown on the PSX Data Portal home page.

        The page only shows the current day, so this builds history one run at a time.
        """
        try:
            import httpx

            resp = httpx.get(
                self.DPS_HOME_URL,
                headers={"User-Agent": self.BROWSER_USER_AGENT},
                timeout=15.0,
                follow_redirects=True,
            )
            resp.raise_for_status()
            points = parse_dps_home_kse100(resp.text)
        except Exception as e:
            logger.warning(f"Could not read KSE-100 from the PSX Data Portal home page: {e}")
            return []
        return [
            p
            for p in points
            if start <= datetime.date.fromisoformat(p.observation_date) <= end
        ]

    def fetch_all(
        self,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
    ) -> list[MacroDataPoint]:
        """Collect all market macroeconomic series."""
        return (
            self.fetch_usd_pkr(start_date, end_date)
            + self.fetch_brent_crude(start_date, end_date)
            + self.fetch_kse100(start_date, end_date)
        )

    # --- Seed Fallback Generators ---

    def _generate_usd_pkr_seed(
        self, start: datetime.date, end: datetime.date
    ) -> list[MacroDataPoint]:
        points = []
        cur = start
        val = 278.50
        while cur <= end:
            if cur.weekday() < 5:
                # Small daily change
                change = ((cur.day % 7) - 3) * 0.08
                val = round(min(285.0, max(276.0, val + change)), 4)
                dt_str = cur.isoformat()
                points.append(
                    MacroDataPoint(
                        indicator=MacroIndicatorType.USD_PKR.value,
                        observation_date=dt_str,
                        public_release_date=dt_str,
                        value=val,
                        unit="PKR",
                        frequency="DAILY",
                        source="SBP/Interbank Seed",
                        notes="Interbank spot rate",
                    )
                )
            cur += datetime.timedelta(days=1)
        return points

    def _generate_brent_seed(
        self, start: datetime.date, end: datetime.date
    ) -> list[MacroDataPoint]:
        points = []
        cur = start
        val = 82.00
        while cur <= end:
            if cur.weekday() < 5:
                change = ((cur.day % 5) - 2) * 0.45
                val = round(min(92.0, max(68.0, val + change)), 2)
                dt_str = cur.isoformat()
                points.append(
                    MacroDataPoint(
                        indicator=MacroIndicatorType.BRENT_CRUDE.value,
                        observation_date=dt_str,
                        public_release_date=dt_str,
                        value=val,
                        unit="USD_PER_BBL",
                        frequency="DAILY",
                        source="ICE Seed",
                        notes="Brent crude settlement",
                    )
                )
            cur += datetime.timedelta(days=1)
        return points

    def _generate_kse100_seed(
        self, start: datetime.date, end: datetime.date
    ) -> list[MacroDataPoint]:
        points = []
        cur = start
        val = 78000.0
        while cur <= end:
            if cur.weekday() < 5:
                change = ((cur.day % 9) - 4) * 120.0
                val = round(min(125000.0, max(60000.0, val + change)), 2)
                dt_str = cur.isoformat()
                points.append(
                    MacroDataPoint(
                        indicator=MacroIndicatorType.KSE_100.value,
                        observation_date=dt_str,
                        public_release_date=dt_str,
                        value=val,
                        unit="POINTS",
                        frequency="DAILY",
                        source="PSX Seed",
                        notes="KSE-100 benchmark index close",
                    )
                )
            cur += datetime.timedelta(days=1)
        return points
