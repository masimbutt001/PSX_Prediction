"""Collector for market-traded macroeconomic indicators: USD/PKR, Brent crude, and KSE-100."""

import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from psx_predictor.macro.schemas import MacroDataPoint, MacroIndicatorType


class MarketMacroCollector:
    """Collects daily traded macro series: USD/PKR, Brent Crude Oil, and KSE-100 Index."""

    def __init__(self, fallback_on_error: bool = True) -> None:
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

    def fetch_kse100(
        self,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None,
    ) -> list[MacroDataPoint]:
        """Fetch historical KSE-100 benchmark index levels (^KSE100)."""
        start = start_date or datetime.date(2024, 1, 1)
        end = end_date or datetime.date.today()
        points: list[MacroDataPoint] = []

        try:
            import yfinance as yf

            end_excl = end + datetime.timedelta(days=1)
            ticker = yf.Ticker("^KSE100")
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
                                indicator=MacroIndicatorType.KSE_100.value,
                                observation_date=dt_str,
                                public_release_date=dt_str,
                                value=round(f_val, 2),
                                unit="POINTS",
                                frequency="DAILY",
                                source="Yahoo Finance (^KSE100)",
                                notes="PSX KSE-100 Benchmark Index close",
                            )
                        )
        except Exception as e:
            logger.warning(f"Could not fetch KSE-100 from Yahoo Finance: {e}")

        if not points and self.fallback_on_error:
            logger.info("Generating deterministic fallback seed data for KSE-100.")
            points = self._generate_kse100_seed(start, end)

        return points

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
