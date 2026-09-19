"""Unit tests for Multi-Modal Stacking Ensemble and Out-of-Fold weight optimization."""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from psx_predictor.cli.main import app
from psx_predictor.features.merger import (
    INTERACTION_FEATURE_COLUMNS,
    MACRO_FEATURE_COLUMNS,
    NEWS_FEATURE_COLUMNS,
)
from psx_predictor.models.ensemble import MultiModalStackingEnsemble
from psx_predictor.models.split import DEFAULT_TECHNICAL_FEATURES
from psx_predictor.models.trainer import ModelTrainer
from psx_predictor.models.weighting import (
    OutOfFoldWeightOptimizer,
    generate_expanding_folds,
)
from psx_predictor.storage.parquet_io import write_parquet_atomic
from psx_predictor.storage.paths import ensure_directories


def _create_synthetic_combined_dataset(
    n_days: int = 150, symbol: str = "OGDC"
) -> pd.DataFrame:
    """Create realistic multi-modal feature dataset with technical, news, and macro series."""
    np.random.seed(42)
    dates = [datetime.date(2025, 6, 1) + datetime.timedelta(days=i) for i in range(n_days)]
    trading_dates = [d for d in dates if d.weekday() < 5]
    n = len(trading_dates)

    rows = []
    base_price = 120.0
    returns = np.random.normal(0.0005, 0.018, n)
    prices = base_price * np.exp(np.cumsum(returns))

    for i, d in enumerate(trading_dates):
        c = float(prices[i])
        o = float(c * (1.0 + np.random.normal(0, 0.003)))
        h = max(o, c) * 1.008
        l_val = min(o, c) * 0.992
        v = int(np.random.randint(50_000, 500_000))

        row: dict[str, object] = {
            "symbol": symbol,
            "trade_date": d,
            "session_date": d.strftime("%Y-%m-%d"),
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l_val, 2),
            "close": round(c, 2),
            "volume": v,
        }

        # Add technical features
        for f in DEFAULT_TECHNICAL_FEATURES:
            row[f] = float(np.random.normal(0.0, 1.0))

        # Add news features
        for nf in NEWS_FEATURE_COLUMNS:
            val = np.random.uniform(-0.5, 0.8) if "sentiment" in nf else np.random.randint(0, 4)
            row[nf] = float(val)

        # Add macro features
        for mf in MACRO_FEATURE_COLUMNS + INTERACTION_FEATURE_COLUMNS:
            row[mf] = float(np.random.normal(20.0, 2.0))

        # Target direction
        row["target_next_day_dir"] = int(returns[min(i + 1, n - 1)] > 0)
        rows.append(row)

    return pd.DataFrame(rows)


def test_generate_expanding_folds() -> None:
    """Verify expanding chronological folds have strictly zero forward leakage."""
    n_samples = 100
    folds = generate_expanding_folds(n_samples=n_samples, n_splits=4, min_train_ratio=0.4)

    assert len(folds) == 4
    prev_val_max = -1

    for f in folds:
        train_max = int(np.max(f.train_indices))
        val_min = int(np.min(f.val_indices))
        val_max = int(np.max(f.val_indices))

        # Train set must strictly precede validation set
        assert train_max < val_min, f"Train max {train_max} >= Val min {val_min}!"
        # Validation windows must be strictly sequential
        assert val_min > prev_val_max, "Validation folds overlap in time!"
        prev_val_max = val_max
        assert val_max <= n_samples


def test_oof_predictions_and_weight_optimization() -> None:
    """Verify OOF prediction generation and constrained optimization weights."""
    df = _create_synthetic_combined_dataset(n_days=150)
    optimizer = OutOfFoldWeightOptimizer(n_splits=3)

    from psx_predictor.models.linear import LogisticRegressionBaseline

    base_models = {
        "price": (LogisticRegressionBaseline(), DEFAULT_TECHNICAL_FEATURES[:10]),
        "news": (LogisticRegressionBaseline(), NEWS_FEATURE_COLUMNS[:4]),
        "macro": (LogisticRegressionBaseline(), MACRO_FEATURE_COLUMNS[:4]),
    }

    y = df["target_next_day_dir"].to_numpy()
    Z_oof, y_oof, names = optimizer.compute_oof_predictions(base_models=base_models, X=df, y=y)

    assert Z_oof.shape[1] == 3
    assert len(y_oof) == Z_oof.shape[0]
    assert names == ["price", "news", "macro"]
    assert np.all((Z_oof >= 0.0) & (Z_oof <= 1.0))

    # Optimize weights
    weights = optimizer.optimize_weights(Z_oof, y_oof)
    assert len(weights) == 3
    assert np.all(weights >= 0.0), f"Negative weights observed: {weights}"
    assert pytest.approx(np.sum(weights), abs=1e-5) == 1.0


def test_multimodal_stacking_ensemble_fit_and_predict() -> None:
    """Verify full end-to-end fit and inference on MultiModalStackingEnsemble."""
    df = _create_synthetic_combined_dataset(n_days=180)
    split_idx = int(len(df) * 0.75)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    y_train = train_df["target_next_day_dir"].to_numpy()

    ensemble = MultiModalStackingEnsemble(price_model_type="random_forest", n_splits=3)
    ensemble.fit(train_df, y_train)

    assert ensemble.is_fitted
    assert len(ensemble.modality_weights_) == 3
    total_w = sum(ensemble.modality_weights_.values())
    assert pytest.approx(total_w, abs=1e-3) == 1.0

    # Test predict_proba
    probs = ensemble.predict_proba(test_df)
    assert probs.shape == (len(test_df), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    row_sums = np.sum(probs, axis=1)
    assert np.all(np.isclose(row_sums, 1.0, atol=1e-5))

    # Test predict
    preds = ensemble.predict(test_df)
    assert len(preds) == len(test_df)
    assert set(np.unique(preds)).issubset({0, 1})

    # Test modality contributions dictionary
    meta = ensemble.get_modality_contributions()
    assert "weights" in meta
    assert "feature_counts" in meta


def test_ensemble_weighted_average_mode() -> None:
    """Verify ensemble prediction under weighted average meta-learner mode."""
    df = _create_synthetic_combined_dataset(n_days=140)
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    y_train = train_df["target_next_day_dir"].to_numpy()

    ensemble = MultiModalStackingEnsemble(
        price_model_type="random_forest",
        n_splits=3,
        meta_learner_type="weighted_average",
    )
    ensemble.fit(train_df, y_train)

    probs = ensemble.predict_proba(test_df)
    assert probs.shape == (len(test_df), 2)
    assert np.all(np.isclose(np.sum(probs, axis=1), 1.0, atol=1e-5))


def test_model_trainer_with_ensemble(tmp_path: Path) -> None:
    """Verify ModelTrainer trains and evaluates MultiModalStackingEnsemble."""
    paths = ensure_directories(tmp_path)
    df = _create_synthetic_combined_dataset(n_days=160, symbol="TEST_SYM")

    # Save combined features to test path
    comb_file = paths["features_combined"] / "TEST_SYM_combined.parquet"
    write_parquet_atomic(df, comb_file)

    trainer = ModelTrainer(storage_paths=paths)
    results = trainer.train_and_evaluate(
        symbol="TEST_SYM",
        model_type="ensemble",
        train_ratio=0.75,
        save_models=False,
    )

    assert len(results) == 1
    res = results[0]
    assert res.model_name == "MultiModalStackingEnsemble"
    assert 0.0 <= res.accuracy <= 1.0
    assert 0.0 <= res.f1_macro <= 1.0
    assert res.roc_auc is not None and 0.0 <= res.roc_auc <= 1.0


def test_cli_model_train_ensemble(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI 'psx model train --model ensemble' executes cleanly via CliRunner."""
    monkeypatch.setenv("PSX_DATA_DIR", str(tmp_path))
    paths = ensure_directories(tmp_path)

    df = _create_synthetic_combined_dataset(n_days=160, symbol="OGDC")
    comb_file = paths["features_combined"] / "OGDC_combined.parquet"
    write_parquet_atomic(df, comb_file)

    runner = CliRunner()
    res = runner.invoke(
        app,
        ["model", "train", "--symbol", "OGDC", "--model", "ensemble"],
        env={"COLUMNS": "160"},
    )

    assert res.exit_code == 0
    assert "Initiating model benchmark for OGDC" in res.stdout
    assert "MultiModalStackingEnsemble" in res.stdout
