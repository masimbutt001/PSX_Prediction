"""Tests for inferring splits/bonus issues missing from raw PSX price data."""

import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from psx_predictor.collectors.base import BaseCollector
from psx_predictor.features.builder import TechnicalFeatureBuilder
from psx_predictor.processing.corporate_actions import (
    adjusted_ohlcv,
    apply_corporate_actions,
    infer_split_ratios,
)
from psx_predictor.processing.updater import IncrementalUpdater
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import get_storage_paths


def _prices(close: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    n = len(close)
    c = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "symbol": "LUCK",
            "trade_date": pd.bdate_range(start, periods=n).date,
            "open": c,
            "high": c * 1.01,
            "low": c * 0.99,
            "close": c,
            "adjusted_close": c,
            "volume": np.full(n, 1000.0),
            "dividend_amount": 0.0,
            "split_ratio": 1.0,
            "is_upper_lock": False,
            "is_lower_lock": False,
        }
    )


def test_split_is_inferred_and_snapped_to_a_clean_ratio() -> None:
    df = _prices([1000.0] * 5 + [202.0] * 5)  # 1:5 split, opens 1% above 200
    out, events = infer_split_ratios(df)
    assert len(events) == 1 and events[0][1] == pytest.approx(5.0)
    assert out["split_ratio"].iloc[5] == pytest.approx(5.0)
    assert (out["split_ratio"].drop(index=5) == 1.0).all()


def test_bonus_issue_ratio_and_normal_moves_ignored() -> None:
    df = _prices([100.0, 109.0, 99.0, 50.0, 50.5])  # +9%, -9% are real; -49% is a 1:2
    _, events = infer_split_ratios(df)
    assert [r for _, r in events] == [pytest.approx(2.0)]


def test_known_split_from_source_is_not_doubled() -> None:
    df = _prices([1000.0] * 3 + [200.0] * 3)
    df.loc[3, "split_ratio"] = 5.0
    _, events = infer_split_ratios(df)
    assert events == []


def test_apply_back_adjusts_history_and_is_idempotent() -> None:
    df = _prices([1000.0] * 5 + [200.0] * 5)
    out, events = apply_corporate_actions(df)
    assert len(events) == 1
    assert np.allclose(out["adjusted_close"], 200.0)
    assert np.allclose(out["close"].iloc[:5], 1000.0)  # raw prices untouched
    again, events2 = apply_corporate_actions(out)
    assert events2 == [] and np.allclose(again["adjusted_close"], 200.0)


def test_adjusted_ohlcv_scales_bars_and_volume() -> None:
    out, _ = apply_corporate_actions(_prices([1000.0] * 5 + [200.0] * 5))
    bars = adjusted_ohlcv(out)
    assert np.allclose(bars["close"], 200.0)
    assert np.allclose(bars["high"], 202.0)
    assert bars["volume"].iloc[0] == pytest.approx(5000.0)
    assert bars["volume"].iloc[-1] == pytest.approx(1000.0)


def test_feature_builder_sees_no_crash_from_a_raw_split(tmp_path: Path) -> None:
    paths = get_storage_paths(tmp_path)
    paths["processed_prices"].mkdir(parents=True, exist_ok=True)
    paths["features_technical"].mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    base = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    raw = base.copy()
    raw[:200] *= 5.0
    _prices(list(raw)).to_parquet(paths["processed_prices"] / "LUCK.parquet")

    feats = TechnicalFeatureBuilder(storage_paths=paths).build_for_symbol(
        "LUCK", with_targets=True, save=False
    )
    assert feats["log_ret_1d"].abs().max() < 0.1
    assert feats["atr_pct_14"].iloc[-1] < 5.0
    assert feats["target_next_day_dir"].iloc[198] in (0, 1)


def test_updater_records_inferred_split_once(tmp_path: Path) -> None:
    paths = get_storage_paths(tmp_path)
    dest = paths["processed_prices"] / "LUCK.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    _prices([1000.0] * 5 + [200.0] * 5).to_parquet(dest)

    delta = _prices([201.0], start="2024-01-15")
    collector = MagicMock(spec=BaseCollector)
    collector.source_name = "mock"
    collector.fetch_historical.return_value = delta
    res = IncrementalUpdater(storage_paths=paths, collector=collector).update_symbol(
        "LUCK", end_date=datetime.date(2024, 1, 15)
    )
    assert res.status == "UPDATED"
    stored = read_parquet(dest)
    assert stored["split_ratio"].iloc[5] == pytest.approx(5.0)
    assert np.allclose(stored["adjusted_close"].iloc[:5], 200.0)
