"""Live (non dry-run) tests for the daily EOD pipeline.

Every network call is replaced with local fake data, so these tests run offline but still
exercise the real code path of all six steps: collectors, storage, features, prediction and
audit. They exist because the earlier test suite only covered dry runs, which hid broken
imports in the live steps.
"""

import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
import pandas as pd
import pytest
import respx

from psx_predictor.collectors.composite import CompositeCollector
from psx_predictor.macro.market_collector import MarketMacroCollector
from psx_predictor.macro.sbp_collector import SBPMacroCollector
from psx_predictor.macro.storage import MacroStorage
from psx_predictor.news.calendar import PSXMarketCalendar
from psx_predictor.news.dps_announcements import DPSAnnouncementsCollector
from psx_predictor.news.rss_collector import DEFAULT_RSS_FEEDS
from psx_predictor.predictions.predictor import LivePredictor
from psx_predictor.scheduler.jobs import DailyPipelineStep
from psx_predictor.scheduler.runner import DailyPipelineRunner
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import ensure_directories

RUN_DATE = datetime.date(2026, 11, 20)  # A Friday and a PSX trading day

RSS_XML = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Markets</title>
<item><title>OGDC profit surges on new gas discovery</title>
<link>https://example.com/ogdc-profit</link>
<description>Oil and Gas Development Company reported record profit.</description>
<pubDate>Thu, 19 Nov 2026 10:00:00 +0500</pubDate><guid>ogdc-profit-1</guid></item>
</channel></rss>"""

DPS_HTML = """<table>
<tr><th>Date</th><th>Time</th><th>Symbol</th><th>Company</th><th>Title</th></tr>
<tr><td>Nov 19, 2026</td><td>11:30 AM</td><td>OGDC</td>
<td>Oil &amp; Gas Development Company Limited</td>
<td>Financial results for the quarter ended September 30, 2026</td></tr></table>"""


def _fake_prices(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Deterministic OHLCV series over PSX trading days between start and end."""
    calendar = PSXMarketCalendar()
    days = [d for d in calendar.get_trading_days_between(start, end)][-400:]
    rng = np.random.default_rng(7)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, len(days)))
    return pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": days,
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adjusted_close": close,
            "volume": rng.integers(100_000, 900_000, len(days)),
            "dividend_amount": 0.0,
            "split_ratio": 1.0,
            "is_upper_lock": False,
            "is_lower_lock": False,
        }
    )


class _FakeTicker:
    """Stand-in for yfinance.Ticker returning a short daily Close series."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    def history(self, start: str, end: str, **_: Any) -> pd.DataFrame:
        idx = pd.bdate_range(start=start, end=end, inclusive="left")
        base = {"PKR=X": 280.0, "BZ=F": 75.0}.get(self.ticker, 120000.0)
        return pd.DataFrame({"Close": np.linspace(base, base * 1.01, len(idx))}, index=idx)


class _FailingTicker(_FakeTicker):
    def history(self, start: str, end: str, **_: Any) -> pd.DataFrame:
        raise ConnectionError("Yahoo Finance unreachable")


@pytest.fixture
def offline_network(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace price, macro and news network sources with local fake data."""

    def fake_fetch_historical(
        self: CompositeCollector,
        symbol: str,
        start_date: datetime.date,
        end_date: Optional[datetime.date] = None,
        save_raw_payload: bool = True,
    ) -> pd.DataFrame:
        return _fake_prices(symbol, start_date, end_date or RUN_DATE)

    monkeypatch.setattr(CompositeCollector, "fetch_historical", fake_fetch_historical)

    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", _FakeTicker)

    with respx.mock(assert_all_called=False) as router:
        for url in DEFAULT_RSS_FEEDS.values():
            router.get(url).mock(return_value=httpx.Response(200, text=RSS_XML))
        router.post(DPSAnnouncementsCollector.ANNOUNCEMENTS_URL).mock(
            return_value=httpx.Response(200, text=DPS_HTML)
        )
        yield router


def test_live_pipeline_runs_all_six_steps(tmp_path: Path, offline_network: Any) -> None:
    """All six steps succeed for real and leave their data on disk."""
    data_dir = tmp_path / "data"
    runner = DailyPipelineRunner(data_dir=data_dir, retry_delay=0.01)

    report = runner.run_daily_pipeline(symbols=["OGDC"])

    assert report is not None
    failures = {t.step.value: t.error for t in report.tasks if t.status != "SUCCESS"}
    assert failures == {}
    assert [t.step for t in report.tasks] == list(DailyPipelineStep)

    paths = ensure_directories(data_dir)
    assert (paths["processed_prices"] / "OGDC.parquet").exists()
    assert (paths["raw_news"]).exists()
    assert (paths["processed_macro"] / "macro_daily.parquet").exists()
    assert (paths["features_technical"] / "OGDC_tech_features.parquet").exists()
    assert (data_dir / "pipeline.log").read_text(encoding="utf-8").strip()

    predictions = read_parquet(paths["predictions"] / "predictions.parquet")
    assert list(predictions["symbol"].unique()) == ["OGDC"]


def test_skip_non_trading_days(tmp_path: Path) -> None:
    """The scheduled run does nothing on weekends and PSX holidays."""
    runner = DailyPipelineRunner(data_dir=tmp_path / "data")
    saturday = datetime.date(2026, 11, 21)
    independence_day = datetime.date(2026, 8, 14)

    assert runner.run_daily_pipeline(skip_non_trading_days=True, run_date=saturday) is None
    assert runner.run_daily_pipeline(skip_non_trading_days=True, run_date=independence_day) is None
    assert not (tmp_path / "data" / "pipeline.log").exists()


def test_macro_update_never_invents_market_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When Yahoo fails, no synthetic series is saved and the gap is reported."""
    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", _FailingTicker)
    storage = MacroStorage(storage_paths=ensure_directories(tmp_path / "data"))

    summary = storage.update(end_date=RUN_DATE)

    assert set(summary.missing_indicators) == {"USD_PKR", "BRENT_CRUDE", "KSE_100"}
    raw = read_parquet(storage.raw_file)
    assert not raw["source"].str.contains("Seed").any()
    daily = read_parquet(storage.processed_file)
    assert daily["usd_pkr"].isna().all()
    assert daily["sbp_policy_rate"].notna().all()


def test_market_collector_has_no_fallback_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", _FailingTicker)
    points = MarketMacroCollector().fetch_usd_pkr(RUN_DATE, RUN_DATE)
    assert points == []


def test_sbp_rates_read_from_config_file(tmp_path: Path) -> None:
    """Policy rates, CPI and the KIBOR spread come from the editable YAML file."""
    rates_file = tmp_path / "macro_rates.yaml"
    rates_file.write_text(
        "kibor_spread_over_policy: 0.5\n"
        "sbp_policy_rates:\n"
        '  - {observation_date: "2026-11-10", release_date: "2026-11-10", rate_pct: 8.0}\n'
        "cpi_releases:\n"
        '  - {observation_date: "2026-10-31", release_date: "2026-11-02", '
        "yoy_pct: 4.4, mom_pct: 0.2}\n",
        encoding="utf-8",
    )
    collector = SBPMacroCollector(rates_file=rates_file)

    assert [p.value for p in collector.fetch_policy_rates()] == [8.0]
    assert [p.value for p in collector.fetch_kibor_rates()] == [8.5]
    assert [p.value for p in collector.fetch_cpi_inflation()] == [4.4, 0.2]


def test_prediction_target_date_skips_psx_holidays(tmp_path: Path) -> None:
    predictor = LivePredictor(storage_paths=ensure_directories(tmp_path / "data"))
    # Thursday 2026-08-13 -> Friday 14 August is Independence Day -> next session Monday 17th
    assert predictor._next_business_day(datetime.date(2026, 8, 13)) == datetime.date(2026, 8, 17)
