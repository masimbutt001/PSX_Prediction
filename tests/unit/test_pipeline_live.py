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
from psx_predictor.collectors.dps_collector import DPSCollector
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


def _failing_dps_fetch(self: Any, symbol: str, start: Any, end: Any) -> Any:
    raise ConnectionError("PSX Data Portal unreachable")


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
    monkeypatch.setattr(DPSCollector, "fetch_raw", _failing_dps_fetch)
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


def _log_run(log_file: Path, started_at: str, success: bool, dry_run: bool = False) -> None:
    import json

    entry = {"started_at": started_at, "success": success, "dry_run": dry_run, "tasks": []}
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def test_catch_up_decision(tmp_path: Path) -> None:
    """Catch-up runs once per trading day after close, retrying failures a few times."""
    from zoneinfo import ZoneInfo

    pkt = ZoneInfo("Asia/Karachi")
    log_file = tmp_path / "pipeline.log"
    runner = DailyPipelineRunner(data_dir=tmp_path / "data", log_path=log_file)

    def at(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime.datetime:
        return datetime.datetime(y, m, d, hh, mm, tzinfo=pkt)

    # Weekend and holiday: never
    assert runner.catch_up_decision(at(2026, 11, 21, 18))[0] is False
    assert runner.catch_up_decision(at(2026, 8, 14, 18))[0] is False
    # Before close: Monday 15:59, Friday 16:30 (Friday runs from 17:00)
    assert runner.catch_up_decision(at(2026, 11, 23, 15, 59))[0] is False
    assert runner.catch_up_decision(at(2026, 11, 20, 16, 30))[0] is False
    # After close with nothing logged: run. Laptop woke at 21:00: still runs.
    assert runner.catch_up_decision(at(2026, 11, 23, 16, 0))[0] is True
    assert runner.catch_up_decision(at(2026, 11, 23, 21, 0))[0] is True

    # Dry runs don't count; a failed run is retried; a successful run stops further runs
    _log_run(log_file, "2026-11-23T11:05:00+00:00", success=True, dry_run=True)
    _log_run(log_file, "2026-11-23T11:05:00+00:00", success=False)
    assert runner.catch_up_decision(at(2026, 11, 23, 17))[0] is True
    _log_run(log_file, "2026-11-23T12:05:00+00:00", success=True)
    should_run, reason = runner.catch_up_decision(at(2026, 11, 23, 18))
    assert should_run is False and "Already completed" in reason
    # Next trading day starts fresh
    assert runner.catch_up_decision(at(2026, 11, 24, 16, 30))[0] is True

    # Three failed attempts: give up for the day
    for hour in (11, 12, 13):
        _log_run(log_file, f"2026-11-25T{hour}:05:00+00:00", success=False)
    should_run, reason = runner.catch_up_decision(at(2026, 11, 25, 20))
    assert should_run is False and "Gave up" in reason


def test_catch_up_runs_pipeline_once(tmp_path: Path, offline_network: Any) -> None:
    from zoneinfo import ZoneInfo

    now = datetime.datetime(2026, 11, 23, 16, 30, tzinfo=ZoneInfo("Asia/Karachi"))
    runner = DailyPipelineRunner(data_dir=tmp_path / "data", retry_delay=0.01)
    runner._target_symbols = ["OGDC"]

    report = runner.run_catch_up(now=now)
    assert report is not None and report.success

    log_lines = runner.log_file.read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1
    assert '"dry_run": false' in log_lines[0]

    # Not due on a Saturday: nothing runs and nothing is logged
    assert runner.run_catch_up(now=now + datetime.timedelta(days=5)) is None
    assert len(runner.log_file.read_text(encoding="utf-8").splitlines()) == 1



def test_kse100_falls_back_to_psx_data_portal(monkeypatch: pytest.MonkeyPatch) -> None:
    """If Yahoo has no KSE-100 data, the index comes from the PSX Data Portal."""
    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", _FailingTicker)
    requested: list[str] = []
    # 2026-11-19 and 2026-11-20 at 00:00 PKT, as DPS stamps sessions
    rows = [[1795028400, 160000.0, 161000.0, 159500.0, 160500.0, 1000],
            [1795114800, 160500.0, 162000.0, 160000.0, 161800.0, 1200]]

    def fake_fetch(self: Any, symbol: str, start: Any, end: Any) -> Any:
        requested.append(symbol)
        return {"type": "timeseries_json", "payload": rows}

    monkeypatch.setattr(DPSCollector, "fetch_raw", fake_fetch)
    points = MarketMacroCollector().fetch_kse100(
        datetime.date(2026, 11, 1), datetime.date(2026, 11, 20)
    )

    assert requested == ["KSE100"]
    assert [(p.observation_date, p.value) for p in points] == [
        ("2026-11-19", 160500.0),
        ("2026-11-20", 161800.0),
    ]
    assert points[0].source == "PSX Data Portal (KSE100)"


def test_news_step_survives_one_source_failing(tmp_path: Path, offline_network: Any) -> None:
    """PSX announcements dropping the connection must not lose the RSS news."""
    offline_network.post(DPSAnnouncementsCollector.ANNOUNCEMENTS_URL).mock(
        side_effect=httpx.RemoteProtocolError("Server disconnected without sending a response.")
    )
    runner = DailyPipelineRunner(data_dir=tmp_path / "data", retry_delay=0.01, max_retries=1)

    result = runner.run_step(DailyPipelineStep.COLLECT_NEWS)

    assert result.status == "SUCCESS"
    assert "RSS" in result.message and "skipped PSX announcements" in result.message


class _NoKse100Ticker(_FakeTicker):
    def history(self, start: str, end: str, **kwargs: Any) -> pd.DataFrame:
        if self.ticker.startswith("^KSE"):
            return pd.DataFrame({"Close": []})
        return super().history(start, end, **kwargs)


def test_macro_step_passes_without_kse100(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing KSE-100 day is a warning, while missing USD/PKR or Brent still fails the step."""
    import yfinance

    monkeypatch.setattr(DPSCollector, "fetch_raw", _failing_dps_fetch)
    runner = DailyPipelineRunner(data_dir=tmp_path / "data")

    monkeypatch.setattr(yfinance, "Ticker", _NoKse100Ticker)
    msg = runner._step_update_macro(dry_run=False)
    assert "no fresh data for KSE_100" in msg

    monkeypatch.setattr(yfinance, "Ticker", _FailingTicker)
    monkeypatch.setattr(DailyPipelineRunner, "_execute_with_retry", lambda self, fn, **_: fn())
    with pytest.raises(RuntimeError, match="USD_PKR"):
        runner._step_update_macro(dry_run=False)
