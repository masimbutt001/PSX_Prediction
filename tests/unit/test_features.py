"""Unit tests for vectorized technical features, indicator precision, and zero look-ahead bias."""

import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from psx_predictor.cli.main import app
from psx_predictor.features.builder import TechnicalFeatureBuilder
from psx_predictor.features.technical import (
    compute_all_technical_features,
    compute_log_returns,
    compute_macd,
    compute_moving_averages,
    compute_rsi,
)
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import get_storage_paths


def _generate_synthetic_ohlcv(n_days: int = 250, base_price: float = 100.0) -> pd.DataFrame:
    """Generate realistic OHLCV price series for testing."""
    np.random.seed(42)
    dates = [datetime.date(2024, 1, 1) + datetime.timedelta(days=i) for i in range(n_days)]

    # Generate random walk prices
    returns = np.random.normal(0.0005, 0.015, n_days)
    price_path = base_price * np.exp(np.cumsum(returns))

    rows = []
    for i, d in enumerate(dates):
        c = float(price_path[i])
        o = float(c * (1.0 + np.random.normal(0, 0.003)))
        h = max(o, c) * (1.0 + abs(np.random.normal(0, 0.008)))
        low_val = min(o, c) * (1.0 - abs(np.random.normal(0, 0.008)))
        rows.append(
            {
                "symbol": "TEST",
                "trade_date": d,
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(low_val, 2),
                "close": round(c, 2),
                "adjusted_close": round(c, 2),
                "volume": int(np.random.randint(50000, 500000)),
                "dividend_amount": 0.0,
                "split_ratio": 1.0,
                "is_upper_lock": False,
                "is_lower_lock": False,
            }
        )
    return pd.DataFrame(rows)


def test_indicator_exact_values() -> None:
    """Verify exact mathematical values of basic indicators."""
    # 1. Test Moving Averages on arithmetic sequence
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    ma_dict = compute_moving_averages(series, sma_windows=(3, 5), ema_windows=())
    # SMA(3) at index 2: (10 + 20 + 30) / 3 = 20.0
    assert ma_dict["sma_3"].iloc[2] == 20.0
    # SMA(5) at index 4: 30.0
    assert ma_dict["sma_5"].iloc[4] == 30.0
    # Normalized distance at index 4: (50 - 30) / 30 = 20 / 30 = 0.6667
    assert np.isclose(ma_dict["dist_sma_5"].iloc[4], 20.0 / 30.0)

    # 2. Test Log Returns
    p = pd.Series([100.0, 110.0])
    rets = compute_log_returns(p, windows=(1,))
    expected = np.log(110.0 / 100.0)
    assert np.isclose(rets["return_1d"].iloc[1], expected)

    # 3. Test RSI on strictly ascending prices
    asc_series = pd.Series([float(i) for i in range(1, 30)])
    rsi_asc = compute_rsi(asc_series, window=14)
    # Strictly positive gains and zero loss must produce RSI 100.0
    assert rsi_asc.iloc[-1] == 100.0

    # 4. Test MACD on completely flat prices
    flat_series = pd.Series([50.0] * 50)
    macd = compute_macd(flat_series)
    assert np.isclose(macd["macd_line"].iloc[-1], 0.0)
    assert np.isclose(macd["macd_signal"].iloc[-1], 0.0)
    assert np.isclose(macd["macd_hist"].iloc[-1], 0.0)


def test_zero_lookahead_perturbation() -> None:
    """Mathematical proof of zero look-ahead bias: altering row T must not change rows 0..T-1."""
    df1 = _generate_synthetic_ohlcv(n_days=100)
    df2 = df1.copy()

    # Drastically perturb the final session in df2
    df2.loc[99, "close"] = df1.loc[99, "close"] * 3.5
    df2.loc[99, "adjusted_close"] = df1.loc[99, "adjusted_close"] * 3.5
    df2.loc[99, "high"] = df1.loc[99, "high"] * 4.0
    df2.loc[99, "volume"] = df1.loc[99, "volume"] * 10

    feats1 = compute_all_technical_features(df1)
    feats2 = compute_all_technical_features(df2)

    # Exclude non-numeric or raw input columns to check all derived features
    feature_cols = [c for c in feats1.columns if c not in df1.columns]
    assert len(feature_cols) >= 20

    # Every single row up to index 98 must be 100% bit-for-bit or floating-point identical
    for col in feature_cols:
        s1 = feats1[col].iloc[:99]
        s2 = feats2[col].iloc[:99]
        # Check equality including NaNs at the same positions
        both_nan = s1.isna() & s2.isna()
        numeric_match = np.isclose(s1[~both_nan], s2[~both_nan], atol=1e-10)
        assert np.all(numeric_match), f"Look-ahead leakage detected in feature '{col}'!"

    # The final row at index 99 SHOULD differ between df1 and df2
    assert feats1["return_1d"].iloc[99] != feats2["return_1d"].iloc[99]


def test_no_inf_or_unexpected_nan() -> None:
    """Verify output contains zero infinities and NaNs only during expected warmup."""
    df = _generate_synthetic_ohlcv(n_days=250)
    feats = compute_all_technical_features(df)

    feature_cols = [c for c in feats.columns if c not in df.columns]

    for col in feature_cols:
        # Zero infinite values allowed anywhere
        assert not np.isinf(feats[col]).any(), f"Column {col} contains infinite values"

    # From session index 200 onward (full warmup), all standard features should be non-null
    post_warmup = feats.iloc[200:]
    for col in feature_cols:
        assert not post_warmup[col].isna().any(), f"Unexpected NaN in post-warmup session for {col}"


def test_builder_atomic_write(tmp_path: Path) -> None:
    """Test TechnicalFeatureBuilder reads processed Parquet and atomically writes feature store."""
    storage_paths = get_storage_paths(tmp_path)
    df = _generate_synthetic_ohlcv(n_days=50)

    # Save to processed_prices/OGDC.parquet
    proc_file = storage_paths["processed_prices"] / "OGDC.parquet"
    proc_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(proc_file)

    builder = TechnicalFeatureBuilder(storage_paths=storage_paths)
    _ = builder.build_for_symbol("OGDC", save=True)

    dest_file = storage_paths["features_technical"] / "OGDC_tech_features.parquet"
    assert dest_file.exists()

    loaded = read_parquet(dest_file)
    assert len(loaded) == len(df)
    assert "rsi_14" in loaded.columns
    assert "macd_line" in loaded.columns
    assert "volatility_20d" in loaded.columns
    assert "bb_upper" in loaded.columns
    assert "obv" in loaded.columns


def test_builder_universe(tmp_path: Path) -> None:
    """Test universe feature generation creates feature manifest and handles multiple symbols."""
    storage_paths = get_storage_paths(tmp_path)

    for sym in ["OGDC", "PPL"]:
        df = _generate_synthetic_ohlcv(n_days=40)
        file_p = storage_paths["processed_prices"] / f"{sym}.parquet"
        file_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(file_p)

    builder = TechnicalFeatureBuilder(storage_paths=storage_paths)
    uni_res = builder.build_universe(symbols=["OGDC", "PPL"], save=True)

    assert uni_res.total_symbols == 2
    assert uni_res.successful_symbols == 2
    assert uni_res.failed_symbols == 0
    assert (storage_paths["features_technical"] / "OGDC_tech_features.parquet").exists()
    assert (storage_paths["features_technical"] / "PPL_tech_features.parquet").exists()
    assert (storage_paths["features_technical"] / "feature_manifest.json").exists()


def test_cli_features_build_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CLI psx features build command."""
    runner = CliRunner()
    storage_paths = get_storage_paths(tmp_path)
    df = _generate_synthetic_ohlcv(n_days=30)
    file_p = storage_paths["processed_prices"] / "OGDC.parquet"
    file_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_p)

    from psx_predictor.features import builder

    monkeypatch.setattr(
        builder,
        "load_config",
        lambda *args, **kwargs: MagicMock(
            settings=MagicMock(data_dir=tmp_path),
        ),
    )

    res = runner.invoke(app, ["features", "build", "--symbols", "OGDC", "--dry-run"])
    assert res.exit_code == 0
    assert "Starting technical feature generation" in res.stdout
    assert "Technical Feature Generation Results" in res.stdout
    assert "SUCCESS" in res.stdout
