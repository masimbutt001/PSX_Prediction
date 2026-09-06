"""Unit tests for Phase 09: Random Forest, XGBoost, and Model Comparator."""

import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from psx_predictor.cli.main import app
from psx_predictor.models.comparator import ModelComparator
from psx_predictor.models.trees import RandomForestBaseline, XGBoostBaseline
from psx_predictor.storage.paths import get_storage_paths


def test_random_forest_fit_predict() -> None:
    """Verify RandomForestBaseline fits and predicts probabilities cleanly."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feat_1": np.random.normal(0, 1, 100),
            "feat_2": np.random.normal(0, 2, 100),
        }
    )
    y = pd.Series(np.random.choice([0.0, 1.0], 100))

    rf = RandomForestBaseline(n_estimators=30, max_depth=3, min_samples_leaf=5)
    rf.fit(X, y)

    assert rf.is_fitted
    preds = rf.predict(X)
    assert len(preds) == 100
    assert set(np.unique(preds)).issubset({0.0, 1.0})

    proba = rf.predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

    importances = rf.get_feature_importances()
    assert "feat_1" in importances and "feat_2" in importances
    assert np.isclose(sum(importances.values()), 1.0)


def test_xgboost_fit_predict() -> None:
    """Verify XGBoostBaseline binary classification and reproducibility."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feat_1": np.random.normal(0, 1, 100),
            "feat_2": np.random.normal(0, 2, 100),
        }
    )
    y = pd.Series(np.random.choice([0.0, 1.0], 100))

    xgb_model = XGBoostBaseline(n_estimators=30, max_depth=3, learning_rate=0.05)
    xgb_model.fit(X, y)

    assert xgb_model.is_fitted
    preds = xgb_model.predict(X)
    assert len(preds) == 100

    proba = xgb_model.predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

    importances = xgb_model.get_feature_importances()
    assert len(importances) == 2


def test_xgboost_multiclass() -> None:
    """Verify XGBoostBaseline supports 3-class target (-1, 0, 1) mapping."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feat_1": np.random.normal(0, 1, 100),
            "feat_2": np.random.normal(0, 2, 100),
        }
    )
    y = pd.Series(np.random.choice([-1.0, 0.0, 1.0], 100))

    xgb_model = XGBoostBaseline(n_estimators=20, max_depth=3)
    xgb_model.fit(X, y)

    assert xgb_model.is_fitted
    preds = xgb_model.predict(X)
    assert set(np.unique(preds)).issubset({-1.0, 0.0, 1.0})

    proba = xgb_model.predict_proba(X)
    assert proba.shape == (100, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_xgboost_early_stopping_chronological() -> None:
    """Verify internal chronological validation split triggers early stopping."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "feat_1": np.random.normal(0, 1, 200),
            "feat_2": np.random.normal(0, 1, 200),
        }
    )
    y = pd.Series(np.random.choice([0.0, 1.0], 200))

    xgb_model = XGBoostBaseline(
        n_estimators=100,
        early_stopping_rounds=10,
        val_ratio=0.15,
    )
    xgb_model.fit(X, y)

    assert xgb_model.is_fitted
    assert xgb_model.clf_ is not None
    # Best iteration must have stopped before or at 100
    assert hasattr(xgb_model.clf_, "best_iteration")


def test_model_comparator_unified_table(tmp_path: Path) -> None:
    """Verify ModelComparator benchmarks all models on test slice."""
    storage_paths = get_storage_paths(tmp_path)
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(120)]

    df = pd.DataFrame(
        {
            "symbol": ["OGDC"] * 120,
            "trade_date": dates,
            "close": np.linspace(100, 150, 120),
            "sma_20": np.linspace(102, 148, 120),
            "sma_50": np.linspace(98, 140, 120),
            "log_ret_1d": [0.005] * 120,
            "target_next_day_dir": np.random.choice([0.0, 1.0], size=120),
            "is_upper_lock": [False] * 120,
            "is_lower_lock": [False] * 120,
        }
    )
    df.loc[119, "target_next_day_dir"] = np.nan

    feat_dir = storage_paths["features_technical"]
    feat_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(feat_dir / "OGDC_tech_features.parquet")

    comparator = ModelComparator(storage_paths=storage_paths)
    summaries = comparator.compare_universe(symbol="OGDC", train_ratio=0.8)

    # 6 models: Majority, Naive, SMA, Logistic, Random Forest, XGBoost
    assert len(summaries) == 6
    names = [s.model_name for s in summaries]
    assert "Random Forest" in names
    assert "XGBoost" in names
    assert "Logistic Regression (L2)" in names


def test_cli_model_train_trees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI psx model train --model trees."""
    runner = CliRunner()
    storage_paths = get_storage_paths(tmp_path)
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(120)]

    df = pd.DataFrame(
        {
            "symbol": ["OGDC"] * 120,
            "trade_date": dates,
            "close": np.linspace(100, 150, 120),
            "sma_20": np.linspace(102, 148, 120),
            "sma_50": np.linspace(98, 140, 120),
            "log_ret_1d": [0.005] * 120,
            "target_next_day_dir": np.random.choice([0.0, 1.0], size=120),
        }
    )
    df.loc[119, "target_next_day_dir"] = np.nan

    feat_dir = storage_paths["features_technical"]
    feat_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(feat_dir / "OGDC_tech_features.parquet")

    from psx_predictor.models import trainer

    monkeypatch.setattr(
        trainer,
        "load_config",
        lambda *args, **kwargs: MagicMock(
            settings=MagicMock(data_dir=tmp_path),
        ),
    )

    args = ["model", "train", "--symbol", "OGDC", "--model", "trees"]
    res = runner.invoke(app, args, env={"COLUMNS": "160"})

    assert res.exit_code == 0
    assert "Random Forest" in res.stdout
    assert "XGBoost" in res.stdout


def test_cli_model_compare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI psx model compare command outputs comparison table."""
    runner = CliRunner()
    storage_paths = get_storage_paths(tmp_path)
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(120)]

    df = pd.DataFrame(
        {
            "symbol": ["OGDC"] * 120,
            "trade_date": dates,
            "close": np.linspace(100, 150, 120),
            "sma_20": np.linspace(102, 148, 120),
            "sma_50": np.linspace(98, 140, 120),
            "log_ret_1d": [0.005] * 120,
            "target_next_day_dir": np.random.choice([0.0, 1.0], size=120),
            "is_upper_lock": [False] * 120,
            "is_lower_lock": [False] * 120,
        }
    )
    df.loc[119, "target_next_day_dir"] = np.nan

    feat_dir = storage_paths["features_technical"]
    feat_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(feat_dir / "OGDC_tech_features.parquet")

    from psx_predictor.models import comparator

    monkeypatch.setattr(
        comparator,
        "load_config",
        lambda *args, **kwargs: MagicMock(
            settings=MagicMock(data_dir=tmp_path),
        ),
    )

    args = ["model", "compare", "--symbol", "OGDC"]
    res = runner.invoke(app, args, env={"COLUMNS": "160"})

    assert res.exit_code == 0
    assert "Initiating multi-model benchmark for OGDC" in res.stdout
    assert "Comprehensive Model Benchmark Comparison: OGDC" in res.stdout
    assert "Random Forest" in res.stdout
    assert "XGBoost" in res.stdout
