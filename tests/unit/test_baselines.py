"""Unit tests for Phase 07: baseline models, chronological splits, and linear classifiers."""

import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from psx_predictor.cli.main import app
from psx_predictor.models.base import BaseModel
from psx_predictor.models.baselines import (
    MajorityClassifier,
    NaivePersistenceClassifier,
    SMACrossoverClassifier,
)
from psx_predictor.models.evaluation import evaluate_classifier
from psx_predictor.models.linear import LogisticRegressionBaseline
from psx_predictor.models.split import chronological_train_test_split
from psx_predictor.storage.paths import get_storage_paths


def test_majority_classifier_predicts_mode() -> None:
    """Verify MajorityClassifier always predicts empirical training mode."""
    # 70% ones, 30% zeros
    y_train = pd.Series([1.0] * 70 + [0.0] * 30)
    X_train = pd.DataFrame({"feat_1": range(100)})
    X_test = pd.DataFrame({"feat_1": range(20)})

    clf = MajorityClassifier()
    clf.fit(X_train, y_train)

    assert clf.is_fitted
    assert clf.majority_class_ == 1.0

    preds = clf.predict(X_test)
    assert len(preds) == 20
    assert np.all(preds == 1.0)

    proba = clf.predict_proba(X_test)
    assert proba.shape == (20, 2)
    # Check empirical priors: [0.3, 0.7]
    assert np.isclose(proba[0, 0], 0.3)
    assert np.isclose(proba[0, 1], 0.7)


def test_naive_persistence_logic() -> None:
    """Verify NaivePersistenceClassifier uses contemporaneous returns."""
    X = pd.DataFrame(
        {
            "log_ret_1d": [0.02, -0.01, 0.005, -0.03, 0.0],
            "other_feat": [1, 2, 3, 4, 5],
        }
    )
    y = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0])

    clf = NaivePersistenceClassifier(return_col="log_ret_1d")
    clf.fit(X, y)

    preds = clf.predict(X)
    # Expected: ret > 0 -> 1.0, ret <= 0 -> 0.0
    expected = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
    assert np.array_equal(preds, expected)

    proba = clf.predict_proba(X)
    assert proba.shape == (5, 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_sma_crossover_logic() -> None:
    """Verify SMACrossoverClassifier triggers Up only when SMA20 > SMA50."""
    X = pd.DataFrame(
        {
            "sma_20": [110.0, 95.0, 105.0, 100.0],
            "sma_50": [100.0, 100.0, 102.0, 100.0],
        }
    )
    y = pd.Series([1.0, 0.0, 1.0, 0.0])

    clf = SMACrossoverClassifier(fast_col="sma_20", slow_col="sma_50")
    clf.fit(X, y)

    preds = clf.predict(X)
    # 110 > 100 -> 1.0, 95 < 100 -> 0.0, 105 > 102 -> 1.0, 100 == 100 -> 0.0
    assert np.array_equal(preds, np.array([1.0, 0.0, 1.0, 0.0]))


def test_scaler_no_data_leakage() -> None:
    """Ensure LogisticRegressionBaseline standardizer is fit exclusively on training split."""
    # Create distinct train and test distributions
    np.random.seed(42)
    X_train = pd.DataFrame(
        {
            "feat_1": np.random.normal(loc=10.0, scale=2.0, size=100),
            "feat_2": np.random.normal(loc=50.0, scale=5.0, size=100),
        }
    )
    y_train = pd.Series(np.random.choice([0.0, 1.0], size=100))

    X_test = pd.DataFrame(
        {
            "feat_1": np.random.normal(loc=100.0, scale=20.0, size=50),
            "feat_2": np.random.normal(loc=500.0, scale=50.0, size=50),
        }
    )

    model = LogisticRegressionBaseline(C=1.0)
    model.fit(X_train, y_train)

    assert model.scaler_ is not None
    # Scaler mean must match X_train mean within small precision, NOT test mean
    assert np.allclose(model.scaler_.mean_, X_train.mean().to_numpy(), atol=1e-5)
    assert not np.allclose(model.scaler_.mean_, X_test.mean().to_numpy(), atol=1.0)

    # Predictions and probabilities work without re-fitting
    preds = model.predict(X_test)
    assert len(preds) == 50
    proba = model.predict_proba(X_test)
    assert proba.shape == (50, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_chronological_split_invariants() -> None:
    """Ensure chronological split maintains strict temporal order with no leaks."""
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(100)]
    df = pd.DataFrame(
        {
            "trade_date": dates,
            "log_ret_1d": np.random.normal(0, 0.01, size=100),
            "sma_20": np.linspace(100, 150, 100),
            "sma_50": np.linspace(90, 140, 100),
            "target_next_day_dir": np.random.choice([0.0, 1.0], size=100),
        }
    )
    # Introduce NaN on the final row to simulate incomplete forward target
    df.loc[99, "target_next_day_dir"] = np.nan

    split = chronological_train_test_split(df, train_ratio=0.8)

    # Total usable rows: 99 (row 99 dropped due to NaN target)
    assert len(split.X_train) + len(split.X_test) == 99
    assert len(split.X_train) == int(np.floor(99 * 0.8))

    # Strict date ordering: test dates must be strictly greater than max train date
    max_train_date = split.train_dates.max()
    min_test_date = split.test_dates.min()
    assert min_test_date > max_train_date


def test_evaluation_metrics_exact() -> None:
    """Verify evaluation metric calculations against known values."""
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])
    y_prob = np.array(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.6, 0.4],
            [0.2, 0.8],
            [0.7, 0.3],
        ]
    )

    metrics = evaluate_classifier(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        model_name="TestModel",
        target_name="target_dir",
        n_train=20,
    )

    # 4 correct out of 5 -> 80% accuracy
    assert np.isclose(metrics.accuracy, 0.8)
    assert metrics.n_train == 20
    assert metrics.n_test == 5
    assert metrics.roc_auc is not None and metrics.roc_auc > 0.5
    assert metrics.brier_score is not None
    assert metrics.to_dict()["accuracy"] == 0.8


def test_model_save_and_load(tmp_path: Path) -> None:
    """Verify fitted models serialize and deserialize cleanly."""
    X = pd.DataFrame({"feat_1": [1.0, 2.0, 3.0, 4.0], "feat_2": [5.0, 6.0, 7.0, 8.0]})
    y = pd.Series([0.0, 0.0, 1.0, 1.0])

    model = LogisticRegressionBaseline()
    model.fit(X, y)

    save_path = tmp_path / "model.joblib"
    model.save(save_path)
    assert save_path.exists()

    loaded = BaseModel.load(save_path)
    assert isinstance(loaded, LogisticRegressionBaseline)
    assert np.array_equal(loaded.predict(X), model.predict(X))
    assert np.allclose(loaded.predict_proba(X), model.predict_proba(X))


def test_cli_model_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI command psx model train executes and reports table."""
    runner = CliRunner()
    storage_paths = get_storage_paths(tmp_path)

    # Create dummy technical features parquet with targets
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(100)]
    df = pd.DataFrame(
        {
            "symbol": ["OGDC"] * 100,
            "trade_date": dates,
            "log_ret_1d": np.random.normal(0, 0.01, size=100),
            "sma_20": np.linspace(100, 150, 100),
            "sma_50": np.linspace(90, 140, 100),
            "target_next_day_dir": np.random.choice([0.0, 1.0], size=100),
        }
    )
    # Final row NaN target
    df.loc[99, "target_next_day_dir"] = np.nan

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

    args = ["model", "train", "--symbol", "OGDC", "--model", "all"]
    res = runner.invoke(app, args, env={"COLUMNS": "160"})

    assert res.exit_code == 0
    assert "Initiating model benchmark for OGDC" in res.stdout
    assert "Model Evaluation Benchmark: OGDC" in res.stdout
    assert "Majority Class Baseline" in res.stdout
    assert "Logistic Regression (L2)" in res.stdout
