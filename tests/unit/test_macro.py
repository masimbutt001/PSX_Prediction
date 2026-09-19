"""Unit tests for Phase 14: Macroeconomic Data Ingestion & Temporal Alignment."""

import datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from psx_predictor.cli.main import app
from psx_predictor.macro.market_collector import MarketMacroCollector
from psx_predictor.macro.sbp_collector import SBPMacroCollector
from psx_predictor.macro.schemas import MacroDataPoint, MacroIndicatorType
from psx_predictor.macro.storage import MacroStorage
from psx_predictor.news.calendar import PSXMarketCalendar
from psx_predictor.storage.parquet_io import read_parquet


@pytest.fixture
def temp_macro_storage(tmp_path: Path) -> MacroStorage:
    """Fixture providing an isolated MacroStorage instance with temporary directories."""
    raw_macro = tmp_path / "raw" / "macro"
    raw_macro.mkdir(parents=True, exist_ok=True)
    proc_macro = tmp_path / "processed" / "macro"
    proc_macro.mkdir(parents=True, exist_ok=True)

    storage_paths = {
        "root": tmp_path,
        "raw_macro": raw_macro,
        "processed_macro": proc_macro,
    }
    return MacroStorage(calendar=PSXMarketCalendar(), storage_paths=storage_paths)


# --- SBP and Market Collector Tests ---


def test_sbp_collector_rates() -> None:
    """Verify SBP policy rates and KIBOR data points have proper dates and units."""
    collector = SBPMacroCollector()
    rates = collector.fetch_policy_rates()
    kibor = collector.fetch_kibor_rates()
    cpi = collector.fetch_cpi_inflation()

    assert len(rates) >= 15
    assert len(kibor) >= 15
    assert len(cpi) >= 30

    first_rate = rates[0]
    assert first_rate.unit == "PERCENT"
    assert first_rate.indicator == MacroIndicatorType.SBP_POLICY_RATE.value
    assert first_rate.value > 0

    first_kibor = kibor[0]
    assert first_kibor.indicator == MacroIndicatorType.KIBOR_6M.value
    assert first_kibor.value > first_rate.value  # KIBOR maintains spread over policy rate


def test_market_collector_data() -> None:
    """Verify MarketMacroCollector returns USD/PKR, Brent crude, and KSE-100."""
    collector = MarketMacroCollector(fallback_on_error=True)
    start = datetime.date(2026, 1, 1)
    end = datetime.date(2026, 1, 15)

    usd_points = collector.fetch_usd_pkr(start, end)
    brent_points = collector.fetch_brent_crude(start, end)
    kse_points = collector.fetch_kse100(start, end)

    assert len(usd_points) > 0
    assert len(brent_points) > 0
    assert len(kse_points) > 0

    assert usd_points[0].indicator == MacroIndicatorType.USD_PKR.value
    assert usd_points[0].unit == "PKR"
    assert brent_points[0].indicator == MacroIndicatorType.BRENT_CRUDE.value
    assert brent_points[0].unit == "USD_PER_BBL"
    assert kse_points[0].indicator == MacroIndicatorType.KSE_100.value
    assert kse_points[0].unit == "POINTS"


# --- Look-Ahead Bias Prevention & Forward-Fill Tests ---


def test_macro_forward_fill_only_after_release(temp_macro_storage: MacroStorage) -> None:
    """Verify that forward-fill begins strictly at public_release_date, not observation_date.

    Scenario:
    - December 2023 CPI (obs: 2023-12-31, release: 2024-01-01, value: 29.7)
    - January 2024 CPI (obs: 2024-01-31, release: 2024-02-01, value: 28.3)

    Strict Temporal Invariant:
    - On 2024-01-15, January CPI is NOT yet known. The latest available CPI MUST be 29.7!
    - On 2024-02-01, January CPI becomes public. The available CPI updates to 28.3!
    """
    raw_data = [
        MacroDataPoint(
            indicator=MacroIndicatorType.CPI_YOY.value,
            observation_date="2023-12-31",
            public_release_date="2024-01-01",
            value=29.7,
            unit="PERCENT",
        ),
        MacroDataPoint(
            indicator=MacroIndicatorType.CPI_YOY.value,
            observation_date="2024-01-31",
            public_release_date="2024-02-01",
            value=28.3,
            unit="PERCENT",
        ),
        # Seed placeholder USD/PKR so full vector builds cleanly
        MacroDataPoint(
            indicator=MacroIndicatorType.USD_PKR.value,
            observation_date="2024-01-01",
            public_release_date="2024-01-01",
            value=281.50,
            unit="PKR",
        ),
    ]
    raw_df = pd.DataFrame([p.to_dict() for p in raw_data])

    trading_dates = [
        datetime.date(2024, 1, 15),
        datetime.date(2024, 1, 31),
        datetime.date(2024, 2, 1),
        datetime.date(2024, 2, 2),
    ]

    features = temp_macro_storage.build_daily_features(raw_df=raw_df, trading_dates=trading_dates)

    val_jan15 = features[features["session_date"] == "2024-01-15"]["cpi_yoy"].iloc[0]
    val_jan31 = features[features["session_date"] == "2024-01-31"]["cpi_yoy"].iloc[0]
    val_feb01 = features[features["session_date"] == "2024-02-01"]["cpi_yoy"].iloc[0]
    val_feb02 = features[features["session_date"] == "2024-02-02"]["cpi_yoy"].iloc[0]

    # January 15 and January 31 MUST NOT see January's 28.3% inflation rate!
    assert val_jan15 == 29.7
    assert val_jan31 == 29.7

    # February 1 and February 2 can legitimately see the newly released 28.3% rate!
    assert val_feb01 == 28.3
    assert val_feb02 == 28.3


def test_oil_price_merging(temp_macro_storage: MacroStorage) -> None:
    """Verify Brent crude joins on contemporaneous trade dates with correct 1-day returns."""
    raw_data = [
        MacroDataPoint(
            indicator=MacroIndicatorType.BRENT_CRUDE.value,
            observation_date="2026-09-08",
            public_release_date="2026-09-08",
            value=80.00,
            unit="USD_PER_BBL",
        ),
        MacroDataPoint(
            indicator=MacroIndicatorType.BRENT_CRUDE.value,
            observation_date="2026-09-09",
            public_release_date="2026-09-09",
            value=82.00,
            unit="USD_PER_BBL",
        ),
        MacroDataPoint(
            indicator=MacroIndicatorType.BRENT_CRUDE.value,
            observation_date="2026-09-10",
            public_release_date="2026-09-10",
            value=77.90,
            unit="USD_PER_BBL",
        ),
    ]
    raw_df = pd.DataFrame([p.to_dict() for p in raw_data])

    trading_dates = [
        datetime.date(2026, 9, 8),
        datetime.date(2026, 9, 9),
        datetime.date(2026, 9, 10),
    ]

    features = temp_macro_storage.build_daily_features(raw_df=raw_df, trading_dates=trading_dates)

    assert len(features) == 3
    assert features.iloc[0]["brent_crude"] == 80.00
    assert features.iloc[1]["brent_crude"] == 82.00
    assert features.iloc[2]["brent_crude"] == 77.90

    # 1-day return: on Sep 9: (82 - 80) / 80 = +0.025 (+2.5%)
    # on Sep 10: (77.9 - 82) / 82 = -0.05 (-5.0%)
    ret_sep9 = features.iloc[1]["brent_return_1d"]
    ret_sep10 = features.iloc[2]["brent_return_1d"]

    assert abs(ret_sep9 - 0.025) < 1e-4
    assert abs(ret_sep10 - (-0.05)) < 1e-4


# --- CLI Integration Tests ---


def test_cli_macro_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify psx macro update and psx macro status execute cleanly via CliRunner."""
    from psx_predictor.storage.paths import ensure_directories

    monkeypatch.setenv("PSX_DATA_DIR", str(tmp_path))
    paths = ensure_directories(tmp_path)

    runner = CliRunner()

    # 1. Test update command
    res_update = runner.invoke(app, ["macro", "update"])
    assert res_update.exit_code == 0
    assert "Macroeconomic Data Ingestion Summary" in res_update.stdout
    assert "Raw Data Points Collected" in res_update.stdout
    assert "SUCCESS: Macro update completed with zero lookahead bias!" in res_update.stdout

    # Verify Parquet files exist
    raw_file = paths["raw_macro"] / "macro_raw.parquet"
    proc_file = paths["processed_macro"] / "macro_daily.parquet"
    assert raw_file.exists()
    assert proc_file.exists()

    df_raw = read_parquet(raw_file)
    assert not df_raw.empty
    df_proc = read_parquet(proc_file)
    assert not df_proc.empty
    assert "usd_pkr" in df_proc.columns
    assert "sbp_policy_rate" in df_proc.columns
    assert "brent_crude" in df_proc.columns

    # 2. Test status command
    res_status = runner.invoke(app, ["macro", "status"])
    assert res_status.exit_code == 0
    assert "Macroeconomic Storage" in res_status.stdout
    assert "Raw Macro" in res_status.stdout
    assert "Daily Macro" in res_status.stdout
