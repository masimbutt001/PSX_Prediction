"""Collector for market-traded macroeconomic indicators: USD/PKR, Brent crude, and KSE-100."""

import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from psx_predictor.macro.schemas import MacroDataPoint, MacroIndicatorType


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
