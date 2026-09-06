"""Unit tests for historical data bootstrap pipeline and quality auditing."""

import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from typer.testing import CliRunner

from psx_predictor.cli.main import app
from psx_predictor.collectors.base import BaseCollector, CollectorError
from psx_predictor.processing.pipeline import (
    DataBootstrapPipeline,
    compute_continuous_adjusted_close,
)
from psx_predictor.processing.quality_report import DataQualityAuditor
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import get_storage_paths


def _generate_clean_ohlcv(
    symbol: str, dates: list[datetime.date], base_price: float = 100.0
) -> pd.DataFrame:
    """Helper to generate valid OHLCV test data."""
    rows = []
    for idx, d in enumerate(dates):
        p = base_price + idx
        rows.append(
            {
                "symbol": symbol,
                "trade_date": d,
                "open": p,
                "high": p + 2.0,
                "low": p - 1.0,
                "close": p + 1.0,
                "adjusted_close": p + 1.0,
                "volume": 100000 + idx * 1000,
                "dividend_amount": 0.0,
                "split_ratio": 1.0,
                "is_upper_lock": False,
                "is_lower_lock": False,
            }
        )
    return pd.DataFrame(rows)


def test_adjusted_close_calculation_dividends() -> None:
    """Verify backward adjustment scales down historical closes when dividends occur."""
    dates = [
        datetime.date(2024, 1, 1),
        datetime.date(2024, 1, 2),
        datetime.date(
            2024, 1, 3
        ),  # Ex-dividend date: 10 PKR dividend (prev close = 100) -> factor = 0.9
        datetime.date(2024, 1, 4),
    ]
    df = pd.DataFrame(
        {
            "trade_date": dates,
            "close": [100.0, 100.0, 95.0, 96.0],
            "dividend_amount": [0.0, 0.0, 10.0, 0.0],
            "split_ratio": [1.0, 1.0, 1.0, 1.0],
        }
    )

    adj = compute_continuous_adjusted_close(df)

    # Days 2 and 3 should have cum_factor 1.0
    assert adj.iloc[3] == 96.0
    assert adj.iloc[2] == 95.0
    # Days 0 and 1 should have cum_factor 0.9 (1.0 - 10/100)
    assert adj.iloc[1] == 90.0
    assert adj.iloc[0] == 90.0


def test_adjusted_close_calculation_splits() -> None:
    """Verify backward adjustment divides historical closes by split ratio."""
    dates = [
        datetime.date(2024, 1, 1),
        datetime.date(2024, 1, 2),  # 2-for-1 split (split_ratio = 2.0)
        datetime.date(2024, 1, 3),
    ]
    df = pd.DataFrame(
        {
            "trade_date": dates,
            "close": [200.0, 105.0, 110.0],
            "dividend_amount": [0.0, 0.0, 0.0],
            "split_ratio": [1.0, 2.0, 1.0],
        }
    )

    adj = compute_continuous_adjusted_close(df)

    # Day 0 should be scaled by 0.5 (1 / 2.0)
    assert adj.iloc[0] == 100.0
    # Day 1 and 2 remain unscaled
    assert adj.iloc[1] == 105.0
    assert adj.iloc[2] == 110.0


def test_bootstrap_orchestration_success(tmp_path: Path) -> None:
    """Test full bootstrap pipeline writing Parquet files and manifest."""
    storage_paths = get_storage_paths(tmp_path)
    mock_collector = MagicMock(spec=BaseCollector)
    mock_collector.source_name = "mock_source"

    dates = [datetime.date(2024, 1, i) for i in range(1, 10)]
    ogdc_df = _generate_clean_ohlcv("OGDC", dates, 120.0)
    ppl_df = _generate_clean_ohlcv("PPL", dates, 80.0)

    def fetch_side_effect(symbol: str, **kwargs):
        return ogdc_df if symbol == "OGDC" else ppl_df

    mock_collector.fetch_historical.side_effect = fetch_side_effect

    pipeline = DataBootstrapPipeline(storage_paths=storage_paths, collector=mock_collector)
    res = pipeline.bootstrap_universe(symbols=["OGDC", "PPL"], years=1)

    assert res.total_symbols == 2
    assert res.successful_symbols == 2
    assert res.failed_symbols == 0
    assert res.total_records == 18

    # Verify Parquet datasets
    ogdc_file = storage_paths["processed_prices"] / "OGDC.parquet"
    ppl_file = storage_paths["processed_prices"] / "PPL.parquet"
    assert ogdc_file.exists()
    assert ppl_file.exists()

    loaded_ogdc = read_parquet(ogdc_file)
    assert len(loaded_ogdc) == 9
    assert loaded_ogdc["symbol"].iloc[0] == "OGDC"

    # Verify manifest
    manifest_file = storage_paths["processed_prices"] / "bootstrap_manifest.json"
    assert manifest_file.exists()


def test_failure_isolation(tmp_path: Path) -> None:
    """Test that a failure in one ticker does not halt or corrupt other tickers."""
    storage_paths = get_storage_paths(tmp_path)
    mock_collector = MagicMock(spec=BaseCollector)
    mock_collector.source_name = "mock_source"

    dates = [datetime.date(2024, 1, i) for i in range(1, 6)]
    ogdc_df = _generate_clean_ohlcv("OGDC", dates, 120.0)
    luck_df = _generate_clean_ohlcv("LUCK", dates, 500.0)

    def fetch_side_effect(symbol: str, **kwargs):
        if symbol == "PPL":
            raise CollectorError("PPL connection dropped by exchange")
        return ogdc_df if symbol == "OGDC" else luck_df

    mock_collector.fetch_historical.side_effect = fetch_side_effect

    pipeline = DataBootstrapPipeline(storage_paths=storage_paths, collector=mock_collector)
    res = pipeline.bootstrap_universe(symbols=["OGDC", "PPL", "LUCK"], years=1)

    assert res.total_symbols == 3
    assert res.successful_symbols == 2
    assert res.failed_symbols == 1

    assert res.symbol_results["OGDC"].status == "SUCCESS"
    assert res.symbol_results["PPL"].status == "FAILED"
    assert "PPL connection dropped" in str(res.symbol_results["PPL"].error_message)
    assert res.symbol_results["LUCK"].status == "SUCCESS"

    # Files check
    assert (storage_paths["processed_prices"] / "OGDC.parquet").exists()
    assert not (storage_paths["processed_prices"] / "PPL.parquet").exists()
    assert (storage_paths["processed_prices"] / "LUCK.parquet").exists()


def test_bootstrap_dry_run(tmp_path: Path) -> None:
    """Test that dry-run validates datasets without writing to disk."""
    storage_paths = get_storage_paths(tmp_path)
    mock_collector = MagicMock(spec=BaseCollector)
    mock_collector.source_name = "mock_source"

    dates = [datetime.date(2024, 1, i) for i in range(1, 5)]
    ogdc_df = _generate_clean_ohlcv("OGDC", dates, 120.0)
    mock_collector.fetch_historical.return_value = ogdc_df

    pipeline = DataBootstrapPipeline(storage_paths=storage_paths, collector=mock_collector)
    res = pipeline.bootstrap_universe(symbols=["OGDC"], years=1, dry_run=True)

    assert res.successful_symbols == 1
    assert not (storage_paths["processed_prices"] / "OGDC.parquet").exists()
    assert not (storage_paths["processed_prices"] / "bootstrap_manifest.json").exists()


def test_bootstrap_skip_existing(tmp_path: Path) -> None:
    """Test overwrite=False skips existing processed Parquet files."""
    storage_paths = get_storage_paths(tmp_path)
    mock_collector = MagicMock(spec=BaseCollector)
    mock_collector.source_name = "mock_source"

    # Pre-create OGDC.parquet
    dates = [datetime.date(2024, 1, i) for i in range(1, 5)]
    ogdc_df = _generate_clean_ohlcv("OGDC", dates, 120.0)
    ogdc_file = storage_paths["processed_prices"] / "OGDC.parquet"
    ogdc_file.parent.mkdir(parents=True, exist_ok=True)
    ogdc_df.to_parquet(ogdc_file)

    pipeline = DataBootstrapPipeline(storage_paths=storage_paths, collector=mock_collector)
    res = pipeline.bootstrap_symbol(
        symbol="OGDC",
        start_date=datetime.date(2024, 1, 1),
        overwrite=False,
    )

    assert res.status == "SKIPPED"
    # Collector should not have been called
    mock_collector.fetch_historical.assert_not_called()


def test_quality_auditor_healthy_dataset() -> None:
    """Test auditor flags consecutive daily trading sessions as HEALTHY."""
    # 20 consecutive weekdays (Mon-Fri)
    dates: list[datetime.date] = []
    curr = datetime.date(2024, 1, 1)
    while len(dates) < 20:
        if curr.weekday() < 5:
            dates.append(curr)
        curr += datetime.timedelta(days=1)

    df = _generate_clean_ohlcv("OGDC", dates, 100.0)
    auditor = DataQualityAuditor()
    report = auditor.audit_dataframe("OGDC", df)

    assert report.symbol == "OGDC"
    assert report.total_sessions == 20
    assert report.health_status == "HEALTHY"
    assert len(report.gaps) == 0
    assert report.coverage_ratio >= 0.95


def test_quality_auditor_detects_gaps() -> None:
    """Test auditor detects an anomalous missing gap in session dates."""
    dates = [
        datetime.date(2024, 1, 1),
        datetime.date(2024, 1, 2),
        # Gap of 14 calendar days (10 weekdays)
        datetime.date(2024, 1, 17),
        datetime.date(2024, 1, 18),
        datetime.date(2024, 1, 19),
    ]
    df = _generate_clean_ohlcv("OGDC", dates, 100.0)
    auditor = DataQualityAuditor()
    report = auditor.audit_dataframe("OGDC", df, max_gap_days=4)

    assert len(report.gaps) == 1
    gap = report.gaps[0]
    assert gap.start_date == "2024-01-02"
    assert gap.end_date == "2024-01-17"
    assert gap.calendar_days == 15
    assert gap.missing_weekdays >= 10
    assert report.health_status in ("WARNING", "CRITICAL")


def test_quality_auditor_circuit_locks_and_dividends() -> None:
    """Test auditor tracks locks and corporate actions accurately."""
    dates = [datetime.date(2024, 1, i) for i in range(1, 6)]
    df = _generate_clean_ohlcv("OGDC", dates, 100.0)
    df.loc[1, "is_upper_lock"] = True
    df.loc[2, "is_lower_lock"] = True
    df.loc[3, "dividend_amount"] = 5.5
    df.loc[4, "split_ratio"] = 2.0

    auditor = DataQualityAuditor()
    report = auditor.audit_dataframe("OGDC", df)

    assert report.upper_locks == 1
    assert report.lower_locks == 1
    assert report.dividend_events == 1
    assert report.total_dividends == 5.5
    assert report.split_events == 1


def test_cli_bootstrap_and_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Typer CLI commands for bootstrap and audit."""
    runner = CliRunner()

    # Pre-populate processed data in tmp_path
    storage_paths = get_storage_paths(tmp_path)
    dates = [
        d
        for d in (datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(21))
        if d.weekday() < 5
    ]
    df = _generate_clean_ohlcv("OGDC", dates, 100.0)
    ogdc_file = storage_paths["processed_prices"] / "OGDC.parquet"
    ogdc_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ogdc_file)

    # Monkeypatch auditor storage paths
    from psx_predictor.processing import quality_report

    monkeypatch.setattr(
        quality_report,
        "load_config",
        lambda *args, **kwargs: MagicMock(settings=MagicMock(data_dir=tmp_path)),
    )

    # Test audit single symbol
    res_audit_single = runner.invoke(app, ["data", "audit", "--symbols", "OGDC"])
    assert res_audit_single.exit_code == 0
    assert "Data Quality Report: OGDC" in res_audit_single.stdout
    assert "HEALTHY" in res_audit_single.stdout

    # Test audit universe
    res_audit_uni = runner.invoke(app, ["data", "audit"])
    assert res_audit_uni.exit_code == 0
    assert "PSX Universe Data Quality Audit" in res_audit_uni.stdout
