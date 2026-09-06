"""Unit tests for incremental daily market data updater and idempotency."""

import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from typer.testing import CliRunner

from psx_predictor.cli.main import app
from psx_predictor.collectors.base import BaseCollector, CollectorError
from psx_predictor.processing.updater import IncrementalUpdater
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import get_storage_paths


def _generate_ohlcv_dataset(
    symbol: str, dates: list[datetime.date], base_price: float = 100.0
) -> pd.DataFrame:
    """Helper to generate a clean OHLCV test dataset."""
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
                "volume": 100000 + idx * 500,
                "dividend_amount": 0.0,
                "split_ratio": 1.0,
                "is_upper_lock": False,
                "is_lower_lock": False,
            }
        )
    return pd.DataFrame(rows)


def test_incremental_fetch_determines_correct_start_date(tmp_path: Path) -> None:
    """Verify incremental query start date is strictly T_max + 1 day."""
    storage_paths = get_storage_paths(tmp_path)
    dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)]
    init_df = _generate_ohlcv_dataset("OGDC", dates)
    dest_file = storage_paths["processed_prices"] / "OGDC.parquet"
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    init_df.to_parquet(dest_file)

    mock_collector = MagicMock(spec=BaseCollector)
    mock_collector.source_name = "mock_source"

    # Delta for Jan 4
    delta_dates = [datetime.date(2024, 1, 4)]
    delta_df = _generate_ohlcv_dataset("OGDC", delta_dates, base_price=105.0)
    mock_collector.fetch_historical.return_value = delta_df

    updater = IncrementalUpdater(storage_paths=storage_paths, collector=mock_collector)
    res = updater.update_symbol("OGDC", end_date=datetime.date(2024, 1, 4))

    assert res.status == "UPDATED"
    assert res.records_added == 1
    assert res.previous_latest_date == "2024-01-03"
    assert res.new_latest_date == "2024-01-04"

    # Verify mock_collector call arguments: start_date MUST be 2024-01-04 (2024-01-03 + 1 day)
    mock_collector.fetch_historical.assert_called_once()
    call_kwargs = mock_collector.fetch_historical.call_args.kwargs
    assert call_kwargs["start_date"] == datetime.date(2024, 1, 4)
    assert call_kwargs["end_date"] == datetime.date(2024, 1, 4)


def test_idempotency_double_run(tmp_path: Path) -> None:
    """Running update twice consecutively must add 0 rows and return ALREADY_UP_TO_DATE."""
    storage_paths = get_storage_paths(tmp_path)
    dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]
    init_df = _generate_ohlcv_dataset("OGDC", dates)
    dest_file = storage_paths["processed_prices"] / "OGDC.parquet"
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    init_df.to_parquet(dest_file)

    mock_collector = MagicMock(spec=BaseCollector)
    mock_collector.source_name = "mock_source"

    # First update introduces Jan 3
    delta_df = _generate_ohlcv_dataset("OGDC", [datetime.date(2024, 1, 3)])
    mock_collector.fetch_historical.return_value = delta_df

    updater = IncrementalUpdater(storage_paths=storage_paths, collector=mock_collector)

    # 1. First run: should update
    res1 = updater.update_symbol("OGDC", end_date=datetime.date(2024, 1, 3))
    assert res1.status == "UPDATED"
    assert res1.records_added == 1
    assert res1.total_records == 3

    # Verify Parquet on disk has 3 rows
    saved_df1 = read_parquet(dest_file)
    assert len(saved_df1) == 3

    # Reset mock call count
    mock_collector.fetch_historical.reset_mock()

    # 2. Second run for same end_date: must be ALREADY_UP_TO_DATE with 0 records added
    res2 = updater.update_symbol("OGDC", end_date=datetime.date(2024, 1, 3))
    assert res2.status == "ALREADY_UP_TO_DATE"
    assert res2.records_added == 0
    assert res2.total_records == 3

    # Parquet on disk still has exactly 3 rows
    saved_df2 = read_parquet(dest_file)
    assert len(saved_df2) == 3

    # Collector shouldn't even be called because start_date > end_date
    mock_collector.fetch_historical.assert_not_called()


def test_dividend_in_delta_updates_adjustments(tmp_path: Path) -> None:
    """A dividend event in a new delta must retroactively adjust historical closes."""
    storage_paths = get_storage_paths(tmp_path)
    # Existing data: 2 days at close 100
    dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]
    init_df = pd.DataFrame(
        {
            "symbol": ["OGDC", "OGDC"],
            "trade_date": dates,
            "open": [99.0, 99.0],
            "high": [101.0, 101.0],
            "low": [98.0, 98.0],
            "close": [100.0, 100.0],
            "adjusted_close": [100.0, 100.0],
            "volume": [50000, 50000],
            "dividend_amount": [0.0, 0.0],
            "split_ratio": [1.0, 1.0],
            "is_upper_lock": [False, False],
            "is_lower_lock": [False, False],
        }
    )
    dest_file = storage_paths["processed_prices"] / "OGDC.parquet"
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    init_df.to_parquet(dest_file)

    # New delta: Jan 3 pays 10 PKR dividend (10% of prior close 100) -> historical factor = 0.9
    delta_df = pd.DataFrame(
        {
            "symbol": ["OGDC"],
            "trade_date": [datetime.date(2024, 1, 3)],
            "open": [90.0],
            "high": [92.0],
            "low": [89.0],
            "close": [91.0],
            "adjusted_close": [91.0],
            "volume": [60000],
            "dividend_amount": [10.0],
            "split_ratio": [1.0],
            "is_upper_lock": [False],
            "is_lower_lock": [False],
        }
    )

    mock_collector = MagicMock(spec=BaseCollector)
    mock_collector.source_name = "mock_source"
    mock_collector.fetch_historical.return_value = delta_df

    updater = IncrementalUpdater(storage_paths=storage_paths, collector=mock_collector)
    res = updater.update_symbol("OGDC", end_date=datetime.date(2024, 1, 3))

    assert res.status == "UPDATED"
    updated_df = read_parquet(dest_file)
    assert len(updated_df) == 3

    # Historical closes for Jan 1 and Jan 2 should now be adjusted down by 10% (from 100.0 to 90.0)
    assert updated_df["adjusted_close"].iloc[0] == 90.0
    assert updated_df["adjusted_close"].iloc[1] == 90.0
    assert updated_df["adjusted_close"].iloc[2] == 91.0


def test_failure_isolation_in_updater(tmp_path: Path) -> None:
    """An error on one ticker must not interrupt other tickers in the update universe."""
    storage_paths = get_storage_paths(tmp_path)
    dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]

    # Setup OGDC, PPL, and LUCK datasets
    for sym in ["OGDC", "PPL", "LUCK"]:
        df = _generate_ohlcv_dataset(sym, dates)
        file_p = storage_paths["processed_prices"] / f"{sym}.parquet"
        file_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(file_p)

    mock_collector = MagicMock(spec=BaseCollector)
    mock_collector.source_name = "mock_source"

    def fetch_side_effect(symbol: str, **kwargs):
        if symbol == "PPL":
            raise CollectorError("PPL connection reset")
        return _generate_ohlcv_dataset(symbol, [datetime.date(2024, 1, 3)])

    mock_collector.fetch_historical.side_effect = fetch_side_effect

    updater = IncrementalUpdater(storage_paths=storage_paths, collector=mock_collector)
    universe_res = updater.update_universe(
        symbols=["OGDC", "PPL", "LUCK"],
        end_date=datetime.date(2024, 1, 3),
    )

    assert universe_res.total_symbols == 3
    assert universe_res.updated_symbols == 2
    assert universe_res.failed_symbols == 1

    assert universe_res.symbol_results["OGDC"].status == "UPDATED"
    assert universe_res.symbol_results["PPL"].status == "FAILED"
    assert "PPL connection reset" in str(universe_res.symbol_results["PPL"].error_message)
    assert universe_res.symbol_results["LUCK"].status == "UPDATED"


def test_unbootstrapped_symbol_handling(tmp_path: Path) -> None:
    """Verify unbootstrapped symbols trigger auto-bootstrap or fail cleanly."""
    storage_paths = get_storage_paths(tmp_path)
    mock_collector = MagicMock(spec=BaseCollector)
    mock_collector.source_name = "mock_source"
    mock_collector.fetch_historical.return_value = _generate_ohlcv_dataset(
        "SYS", [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]
    )

    updater = IncrementalUpdater(storage_paths=storage_paths, collector=mock_collector)

    # 1. With auto_bootstrap=True
    res_boot = updater.update_symbol("SYS", auto_bootstrap=True)
    assert res_boot.status == "BOOTSTRAPPED"
    assert (storage_paths["processed_prices"] / "SYS.parquet").exists()

    # 2. For an unknown symbol with auto_bootstrap=False
    res_fail = updater.update_symbol("UNKNOWN", auto_bootstrap=False)
    assert res_fail.status == "FAILED"
    assert "auto_bootstrap=False" in str(res_fail.error_message)


def test_cli_update_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CLI psx data update command with dry-run."""
    runner = CliRunner()
    storage_paths = get_storage_paths(tmp_path)
    dates = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]
    init_df = _generate_ohlcv_dataset("OGDC", dates)
    dest_file = storage_paths["processed_prices"] / "OGDC.parquet"
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    init_df.to_parquet(dest_file)

    from psx_predictor.processing import updater

    monkeypatch.setattr(
        updater,
        "load_config",
        lambda *args, **kwargs: MagicMock(
            settings=MagicMock(data_dir=tmp_path),
            get_enabled_stocks=lambda: [MagicMock(symbol="OGDC")],
        ),
    )

    res = runner.invoke(app, ["data", "update", "--symbols", "OGDC", "--dry-run"])
    assert res.exit_code == 0
    assert "Starting incremental update" in res.stdout
    assert "Incremental Daily Update Execution Results" in res.stdout
