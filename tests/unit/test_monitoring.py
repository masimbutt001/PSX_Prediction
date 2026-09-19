"""Unit tests for Phase 20: Model Monitoring, Calibration, Drift, and Health Audit."""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from psx_predictor.cli.main import app as cli_app
from psx_predictor.monitoring.calibration import (
    CalibrationStatus,
    CalibrationTracker,
    compute_brier_score,
    compute_calibration_curve,
    compute_expected_calibration_error,
)
from psx_predictor.monitoring.drift import (
    DriftStatus,
    FeatureDriftDetector,
    compute_ks_drift,
    compute_psi,
)
from psx_predictor.monitoring.health import (
    HealthStatus,
    SystemHealthMonitor,
)
from psx_predictor.storage.parquet_io import write_parquet_atomic


def test_brier_score_computation_exact() -> None:
    """Validate mathematical exactness of Brier verification scoring."""
    # 1. Perfect forecast: 0.0 error
    probs_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    outcomes_perfect = np.array([1, 0, 1, 0])
    assert compute_brier_score(probs_perfect, outcomes_perfect) == 0.0

    # 2. Inverted forecast: 1.0 error
    probs_inverted = np.array([0.0, 1.0, 0.0])
    outcomes_inverted = np.array([1, 0, 1])
    assert compute_brier_score(probs_inverted, outcomes_inverted) == 1.0

    # 3. Uninformative 0.5 coin flip: 0.25 error
    probs_coin = np.array([0.5, 0.5, 0.5, 0.5])
    outcomes_coin = np.array([1, 0, 1, 0])
    assert compute_brier_score(probs_coin, outcomes_coin) == 0.25

    # 4. Empty input
    assert compute_brier_score(np.array([]), np.array([])) == 0.0

    # 5. Length mismatch raises ValueError
    with pytest.raises(ValueError, match="Array length mismatch"):
        compute_brier_score(np.array([0.5]), np.array([1, 0]))


def test_calibration_curve_and_ece() -> None:
    """Verify reliability diagram binning and Expected Calibration Error (ECE)."""
    np.random.seed(42)
    # Generate calibrated probabilities: P ~ Uniform(0, 1), outcome ~ Bern(P)
    probs = np.random.uniform(0.05, 0.95, size=2000)
    outcomes = (np.random.uniform(0, 1, size=2000) < probs).astype(float)

    pred_probs, true_props, counts = compute_calibration_curve(probs, outcomes, n_bins=10)
    assert len(pred_probs) == 10
    assert len(true_props) == 10
    assert len(counts) == 10
    assert np.sum(counts) == 2000

    ece = compute_expected_calibration_error(probs, outcomes, n_bins=10)
    # A genuinely calibrated distribution should have low ECE (< 0.06)
    assert ece < 0.06


def test_calibration_tracker_alerts() -> None:
    """Verify CalibrationTracker flags severe miscalibration and healthy models appropriately."""
    tracker = CalibrationTracker()

    # 1. Severely overconfident wrong predictions
    dates = pd.date_range("2026-01-01", periods=30)
    bad_df = pd.DataFrame(
        {
            "target_date": [d.strftime("%Y-%m-%d") for d in dates],
            "up_probability": [0.95] * 30,  # High confidence UP
            "realized_outcome": [-0.02] * 30,  # Actually fell 2% every day
            "is_correct": [False] * 30,
        }
    )

    bad_report = tracker.evaluate_predictions(df=bad_df)
    assert bad_report.status == CalibrationStatus.SEVERE_MISCALIBRATION
    assert bad_report.brier_score > 0.8
    assert "Severe miscalibration detected" in bad_report.alert_message

    # 2. Well-calibrated predictions
    np.random.seed(42)
    good_probs = np.random.uniform(0.1, 0.9, size=200)
    good_binary = (np.random.uniform(0, 1, size=200) < good_probs).astype(float)
    good_outcomes = np.where(good_binary == 1.0, 0.02, -0.02)
    dates_good = pd.date_range("2026-01-01", periods=200)

    good_df = pd.DataFrame(
        {
            "target_date": [d.strftime("%Y-%m-%d") for d in dates_good],
            "up_probability": good_probs,
            "realized_outcome": good_outcomes,
            "is_correct": [True] * 200,
        }
    )

    good_report = tracker.evaluate_predictions(df=good_df)
    assert good_report.status == CalibrationStatus.HEALTHY
    assert good_report.brier_score < 0.22


def test_psi_identical_distributions() -> None:
    """Verify PSI of identical Gaussian distributions is practically zero (< 0.05)."""
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1000)
    act = np.random.normal(0, 1, 1000)

    psi = compute_psi(ref, act, num_bins=10)
    assert psi < 0.05
    assert psi >= 0.0


def test_psi_detects_distributional_shift() -> None:
    """Verify PSI accurately discriminates between moderate and severe distribution shifts."""
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1500)

    # Moderate shift (mean shift +0.35) -> 0.10 <= PSI < 0.25 (WARNING)
    moderate_act = np.random.normal(0.35, 1, 1500)
    psi_mod = compute_psi(ref, moderate_act, num_bins=10)
    assert 0.08 <= psi_mod < 0.30

    # Severe shift (mean shift +2.0) -> PSI >= 0.25 (ALERT)
    severe_act = np.random.normal(2.0, 1, 1500)
    psi_sev = compute_psi(ref, severe_act, num_bins=10)
    assert psi_sev >= 0.25


def test_ks_drift_detection() -> None:
    """Verify two-sample Kolmogorov-Smirnov test catches distributional divergence."""
    np.random.seed(42)
    ref = np.random.normal(0, 1, 500)
    identical_act = np.random.normal(0, 1, 500)
    shifted_act = np.random.normal(1.5, 1, 500)

    ks_stat_ident, p_ident = compute_ks_drift(ref, identical_act)
    assert p_ident > 0.05  # Fail to reject null hypothesis of same distribution

    ks_stat_shift, p_shift = compute_ks_drift(ref, shifted_act)
    assert ks_stat_shift > 0.3
    assert p_shift < 1e-5  # Strongly reject null hypothesis


def test_feature_drift_detector_dataframe() -> None:
    """Verify FeatureDriftDetector correctly flags drifted features in a multi-feature dataset."""
    np.random.seed(42)
    n_samples = 300

    ref_df = pd.DataFrame(
        {
            "trade_date": ["2025-01-01"] * n_samples,
            "rsi_14": np.random.normal(50, 10, n_samples),
            "volatility_20": np.random.normal(0.02, 0.005, n_samples),
        }
    )

    act_df = pd.DataFrame(
        {
            "trade_date": ["2026-01-01"] * n_samples,
            "rsi_14": np.random.normal(50, 10, n_samples),  # Stable
            "volatility_20": np.random.normal(0.06, 0.01, n_samples),  # Severe drift
        }
    )

    detector = FeatureDriftDetector()
    summary = detector.evaluate_features(reference_df=ref_df, current_df=act_df)

    assert summary.total_features == 2
    assert summary.stable_features == 1
    assert summary.alert_features == 1
    assert summary.overall_status == DriftStatus.ALERT
    assert summary.retraining_recommended is True
    assert summary.feature_results["volatility_20"].status == DriftStatus.ALERT
    assert summary.feature_results["rsi_14"].status == DriftStatus.STABLE


def test_data_feed_health_fresh_and_stale(tmp_path: Path) -> None:
    """Verify SystemHealthMonitor discriminates between fresh and stale price feeds."""
    paths = {
        "root": tmp_path,
        "processed_prices": tmp_path / "processed" / "prices",
        "features_combined": tmp_path / "features" / "combined",
        "features_technical": tmp_path / "features" / "technical",
        "predictions": tmp_path / "predictions",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    today = datetime.datetime.now(datetime.timezone.utc).date()
    fresh_date = today.strftime("%Y-%m-%d")
    stale_date = (today - datetime.timedelta(days=15)).strftime("%Y-%m-%d")

    # Create fresh parquet for OGDC
    dates_fresh = [
        (today - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(5, -1, -1)
    ]
    fresh_df = pd.DataFrame(
        {
            "trade_date": dates_fresh,
            "close": [120.0 + i for i in range(6)],
            "volume": [1_000_000] * 6,
        }
    )
    write_parquet_atomic(fresh_df, paths["processed_prices"] / "OGDC.parquet")

    # Create stale parquet for HBL
    stale_df = pd.DataFrame(
        {
            "trade_date": [stale_date],
            "close": [100.0],
            "volume": [500_000],
        }
    )
    write_parquet_atomic(stale_df, paths["processed_prices"] / "HBL.parquet")

    monitor = SystemHealthMonitor(storage_paths=paths)
    data_summary = monitor.check_data_feed(symbols=["OGDC", "HBL"])

    assert data_summary.total_symbols == 2
    assert data_summary.healthy_symbols == 1
    assert "HBL" in data_summary.stale_symbols
    assert data_summary.status == HealthStatus.DEGRADED
    assert data_summary.latest_date == fresh_date


def test_cli_health_check() -> None:
    """Verify 'psx health check' executes and prints health status cleanly."""
    runner = CliRunner()
    result = runner.invoke(cli_app, ["health", "check"])
    assert result.exit_code == 0
    assert "PSX Predictor - System Health Audit" in result.output
    assert "Subsystem Health Summary" in result.output
    assert "Data Feeds" in result.output
    assert "Model Reliability" in result.output
    assert "Distribution Drift" in result.output


def test_cli_health_drift_no_features() -> None:
    """Verify 'psx health drift' handles symbols with no features gracefully."""
    runner = CliRunner()
    result = runner.invoke(cli_app, ["health", "drift", "--symbol", "NONEXISTENT"])
    assert result.exit_code == 0
    assert "No multi-modal or technical features found" in result.output
