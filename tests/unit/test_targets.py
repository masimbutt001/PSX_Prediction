"""Unit tests for supervised learning prediction targets, forward alignment, and zero leakage."""

import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from psx_predictor.cli.main import app
from psx_predictor.features.builder import TechnicalFeatureBuilder
from psx_predictor.features.targets import (
    compute_all_prediction_targets,
    compute_binary_direction_target,
    compute_future_return_target,
    compute_threshold_3class_target,
)
from psx_predictor.features.technical import compute_all_technical_features
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import get_storage_paths


def test_target_alignment_hand_calculated() -> None:
    """Verify target calculation against manually calculated labels."""
    prices = pd.Series([100.0, 102.0, 101.0, 103.0, 103.0, 105.0])

    # 1. Binary Direction: 1 if next > current else 0
    t_dir = compute_binary_direction_target(prices)
    assert t_dir.iloc[0] == 1.0  # 102 > 100
    assert t_dir.iloc[1] == 0.0  # 101 <= 102
    assert t_dir.iloc[2] == 1.0  # 103 > 101
    assert t_dir.iloc[3] == 0.0  # 103 <= 103 (flat is not >)
    assert t_dir.iloc[4] == 1.0  # 105 > 103
    assert np.isnan(t_dir.iloc[5])  # Last row must be NaN

    # 2. 3-Class with threshold 0.0075 (0.75%)
    t_3c = compute_threshold_3class_target(prices, threshold=0.0075)
    assert t_3c.iloc[0] == 1.0  # +2.0% > 0.75% -> UP
    assert t_3c.iloc[1] == -1.0  # -0.98% < -0.75% -> DOWN
    assert t_3c.iloc[2] == 1.0  # +1.98% > 0.75% -> UP
    assert t_3c.iloc[3] == 0.0  # 0.0% -> NEUTRAL
    assert t_3c.iloc[4] == 1.0  # +1.94% > 0.75% -> UP
    assert np.isnan(t_3c.iloc[5])  # Last row must be NaN

    # 3. 5-Day Future Return
    t_ret5 = compute_future_return_target(prices, horizon=5)
    # Day 0 to Day 5: (105 / 100) - 1 = +0.05
    assert np.isclose(t_ret5.iloc[0], 0.05)
    # Days 1..5 have fewer than 5 forward periods and must be NaN
    for i in range(1, 6):
        assert np.isnan(t_ret5.iloc[i])


def test_last_row_target_is_nan() -> None:
    """Verify that forward targets strictly assign NaN to incomplete future periods."""
    n = 50
    dates = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(n)]
    df = pd.DataFrame(
        {
            "symbol": ["TEST"] * n,
            "trade_date": dates,
            "close": [100.0 + i for i in range(n)],
            "adjusted_close": [100.0 + i for i in range(n)],
        }
    )

    targets = compute_all_prediction_targets(df, threshold=0.0075, horizon=5)

    # Direction target: only row T (last row) is NaN
    assert np.isnan(targets["target_next_day_dir"].iloc[-1])
    assert not targets["target_next_day_dir"].iloc[:-1].isna().any()

    # 3-Class target: only row T (last row) is NaN
    assert np.isnan(targets["target_next_day_3class"].iloc[-1])
    assert not targets["target_next_day_3class"].iloc[:-1].isna().any()

    # 5-day Return: exactly the last 5 rows are NaN
    assert targets["target_return_5d"].iloc[-5:].isna().all()
    assert not targets["target_return_5d"].iloc[:-5].isna().any()


def test_threshold_3class_boundaries() -> None:
    """Verify precise behavior around threshold boundaries (+0.75% and -0.75%)."""
    # 100.0 -> 100.75 (+0.75% exact boundary: should be NEUTRAL since not strictly >)
    # 100.0 -> 100.76 (+0.76% > 0.75%: UP)
    # 100.0 -> 99.25 (-0.75% exact boundary: should be NEUTRAL since not strictly <)
    # 100.0 -> 99.24 (-0.76% < -0.75%: DOWN)
    prices = pd.Series([100.0, 100.75, 100.0, 100.76, 100.0, 99.25, 100.0, 99.24])
    t_3c = compute_threshold_3class_target(prices, threshold=0.0075)

    assert t_3c.iloc[0] == 0.0  # exactly 0.75% -> NEUTRAL
    assert t_3c.iloc[2] == 1.0  # 0.76% -> UP
    assert t_3c.iloc[4] == 0.0  # exactly -0.75% -> NEUTRAL
    assert t_3c.iloc[6] == -1.0  # -0.76% -> DOWN


def test_target_no_leakage_into_features() -> None:
    """Ensure that generating forward targets does not alter contemporaneous input features."""
    dates = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(30)]
    df = pd.DataFrame(
        {
            "symbol": ["OGDC"] * 30,
            "trade_date": dates,
            "open": [100.0 + i for i in range(30)],
            "high": [102.0 + i for i in range(30)],
            "low": [99.0 + i for i in range(30)],
            "close": [101.0 + i for i in range(30)],
            "adjusted_close": [101.0 + i for i in range(30)],
            "volume": [100000] * 30,
        }
    )

    # Compute features alone
    features_only = compute_all_technical_features(df)

    # Compute targets
    targets = compute_all_prediction_targets(df)
    features_with_targets = pd.concat([features_only, targets], axis=1)

    # Features must match 100% across all rows
    for col in features_only.columns:
        s1 = features_only[col]
        s2 = features_with_targets[col]
        if pd.api.types.is_numeric_dtype(s1.dtype):
            both_nan = s1.isna() & s2.isna()
            assert np.all(np.isclose(s1[~both_nan], s2[~both_nan], atol=1e-12))
        else:
            assert s1.equals(s2)


def test_builder_with_targets(tmp_path: Path) -> None:
    """Test TechnicalFeatureBuilder properly attaches prediction targets."""
    storage_paths = get_storage_paths(tmp_path)
    dates = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(40)]
    df = pd.DataFrame(
        {
            "symbol": ["OGDC"] * 40,
            "trade_date": dates,
            "open": [100.0 + i for i in range(40)],
            "high": [102.0 + i for i in range(40)],
            "low": [99.0 + i for i in range(40)],
            "close": [101.0 + i for i in range(40)],
            "adjusted_close": [101.0 + i for i in range(40)],
            "volume": [100000] * 40,
            "dividend_amount": [0.0] * 40,
            "split_ratio": [1.0] * 40,
            "is_upper_lock": [False] * 40,
            "is_lower_lock": [False] * 40,
        }
    )

    proc_file = storage_paths["processed_prices"] / "OGDC.parquet"
    proc_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(proc_file)

    builder = TechnicalFeatureBuilder(storage_paths=storage_paths)
    features_df = builder.build_for_symbol("OGDC", save=True, with_targets=True)
    assert len(features_df) == 40

    dest_file = storage_paths["features_technical"] / "OGDC_tech_features.parquet"
    assert dest_file.exists()

    loaded = read_parquet(dest_file)
    assert len(loaded) == 40
    # Targets present
    assert "target_next_day_dir" in loaded.columns
    assert "target_next_day_3class" in loaded.columns
    assert "target_return_5d" in loaded.columns
    # Last row target NaN verification
    assert np.isnan(loaded["target_next_day_dir"].iloc[-1])
    assert np.isnan(loaded["target_return_5d"].iloc[-1])
    assert not np.isnan(loaded["target_next_day_dir"].iloc[0])


def test_cli_features_build_with_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CLI psx features build --with-targets."""
    runner = CliRunner()
    storage_paths = get_storage_paths(tmp_path)
    dates = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(30)]
    df = pd.DataFrame(
        {
            "symbol": ["OGDC"] * 30,
            "trade_date": dates,
            "open": [100.0 + i for i in range(30)],
            "high": [102.0 + i for i in range(30)],
            "low": [99.0 + i for i in range(30)],
            "close": [101.0 + i for i in range(30)],
            "adjusted_close": [101.0 + i for i in range(30)],
            "volume": [100000] * 30,
            "dividend_amount": [0.0] * 30,
            "split_ratio": [1.0] * 30,
            "is_upper_lock": [False] * 30,
            "is_lower_lock": [False] * 30,
        }
    )
    proc_file = storage_paths["processed_prices"] / "OGDC.parquet"
    proc_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(proc_file)

    from psx_predictor.features import builder

    monkeypatch.setattr(
        builder,
        "load_config",
        lambda *args, **kwargs: MagicMock(
            settings=MagicMock(data_dir=tmp_path),
        ),
    )

    args = ["features", "build", "--with-targets", "--symbols", "OGDC", "--dry-run"]
    res = runner.invoke(app, args)
    assert res.exit_code == 0
    assert "Targets: INCLUDED" in res.stdout
    assert "SUCCESS" in res.stdout
