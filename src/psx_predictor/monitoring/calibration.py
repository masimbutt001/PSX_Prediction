"""Probability calibration tracking, reliability curves, and Brier score monitoring."""

import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from psx_predictor.config.loader import load_config
from psx_predictor.predictions.registry import PredictionRegistry
from psx_predictor.storage.paths import ensure_directories


class CalibrationStatus(str, Enum):
    """Reliability assessment status for model forecast probabilities."""

    HEALTHY = "HEALTHY"
    MODERATE_MISCALIBRATION = "MODERATE_MISCALIBRATION"
    SEVERE_MISCALIBRATION = "SEVERE_MISCALIBRATION"


def compute_brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute standard Brier verification score for binary probabilistic forecasts.

    Formula:
        Brier = (1 / N) * sum((P_i - y_i)^2)
        Range: [0.0 (perfect foresight), 1.0 (completely opposite)].
        Uninformative reference (0.5 coin-flip) = 0.25.

    Args:
        probabilities: Array of predicted probabilities in [0.0, 1.0].
        outcomes: Array of binary realized outcomes in {0, 1}.

    Returns:
        Mean squared probability verification error.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)

    if len(p) == 0:
        return 0.0
    if len(p) != len(y):
        raise ValueError(f"Array length mismatch: probabilities={len(p)}, outcomes={len(y)}")

    # Clamp probabilities to valid [0, 1] range to avoid floating point overshoot
    p = np.clip(p, 0.0, 1.0)
    return float(np.mean((p - y) ** 2))


def compute_calibration_curve(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reliability diagram bins comparing predicted confidence vs realized frequency.

    Args:
        probabilities: Array of predicted probabilities in [0.0, 1.0].
        outcomes: Array of binary realized outcomes in {0, 1}.
        n_bins: Number of confidence bins (default 10).

    Returns:
        Tuple of (bin_pred_probs, bin_true_proportions, bin_counts).
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)

    if len(p) == 0:
        return np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins, dtype=int)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_pred_probs = []
    bin_true_proportions = []
    bin_counts = []

    for i in range(n_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)

        count = int(np.sum(mask))
        bin_counts.append(count)

        if count > 0:
            bin_pred_probs.append(float(np.mean(p[mask])))
            bin_true_proportions.append(float(np.mean(y[mask])))
        else:
            bin_pred_probs.append(float((low + high) / 2.0))
            bin_true_proportions.append(0.0)

    return (
        np.array(bin_pred_probs),
        np.array(bin_true_proportions),
        np.array(bin_counts, dtype=int),
    )


def compute_expected_calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) across confidence bins.

    Formula:
        ECE = sum_{b=1}^B (N_b / N) * |acc(B_b) - conf(B_b)|

    Args:
        probabilities: Array of predicted probabilities in [0.0, 1.0].
        outcomes: Array of binary realized outcomes in {0, 1}.
        n_bins: Number of bins.

    Returns:
        Weighted expected calibration error.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)

    n_total = len(p)
    if n_total == 0:
        return 0.0

    pred_probs, true_props, counts = compute_calibration_curve(p, y, n_bins=n_bins)
    ece = 0.0
    for i in range(len(counts)):
        if counts[i] > 0:
            weight = counts[i] / n_total
            ece += weight * abs(true_props[i] - pred_probs[i])

    return float(ece)


@dataclass
class CalibrationReport:
    """Report encapsulating reliability, Brier performance, and miscalibration alerts."""

    total_samples: int
    brier_score: float
    brier_30d: Optional[float]
    brier_90d: Optional[float]
    ece: float
    status: CalibrationStatus
    alert_message: str
    bin_pred_probs: list[float] = field(default_factory=list)
    bin_true_proportions: list[float] = field(default_factory=list)
    bin_counts: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert calibration report to JSON-serializable dictionary."""
        return {
            "total_samples": self.total_samples,
            "brier_score": round(self.brier_score, 4),
            "brier_30d": round(self.brier_30d, 4) if self.brier_30d is not None else None,
            "brier_90d": round(self.brier_90d, 4) if self.brier_90d is not None else None,
            "ece": round(self.ece, 4),
            "status": self.status.value,
            "alert_message": self.alert_message,
            "bin_pred_probs": [round(x, 3) for x in self.bin_pred_probs],
            "bin_true_proportions": [round(x, 3) for x in self.bin_true_proportions],
            "bin_counts": self.bin_counts,
        }


class CalibrationTracker:
    """Monitors rolling probability calibration of live models against verified outcomes."""

    def __init__(self, storage_paths: Optional[dict[str, Path]] = None) -> None:
        if storage_paths is None:
            config = load_config()
            self.storage_paths = ensure_directories(config.settings.data_dir)
        else:
            self.storage_paths = storage_paths

        self.registry = PredictionRegistry(storage_paths=self.storage_paths)

    def evaluate_predictions(
        self,
        df: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
    ) -> CalibrationReport:
        """Evaluate calibration health on resolved prediction records.

        Args:
            df: Optional pre-loaded DataFrame of prediction records.
            symbol: Optional symbol filter.

        Returns:
            CalibrationReport with Brier metrics, ECE, and reliability curve bins.
        """
        if df is None:
            df = self.registry.get_predictions(symbol=symbol)

        if df.empty:
            return CalibrationReport(
                total_samples=0,
                brier_score=0.0,
                brier_30d=None,
                brier_90d=None,
                ece=0.0,
                status=CalibrationStatus.HEALTHY,
                alert_message="No predictions logged in registry yet.",
            )

        # Filter to resolved predictions
        resolved_mask = df["realized_outcome"].notna()
        resolved_df = df[resolved_mask].copy()

        if resolved_df.empty:
            return CalibrationReport(
                total_samples=0,
                brier_score=0.0,
                brier_30d=None,
                brier_90d=None,
                ece=0.0,
                status=CalibrationStatus.HEALTHY,
                alert_message="No resolved predictions available for calibration audit.",
            )

        # Sort by target_date
        resolved_df["date_val"] = pd.to_datetime(resolved_df["target_date"])
        resolved_df = resolved_df.sort_values("date_val").reset_index(drop=True)

        probs = resolved_df["up_probability"].to_numpy(dtype=float)
        # Outcome = 1 if realized return > 0 else 0
        outcomes = np.where(resolved_df["realized_outcome"].to_numpy(dtype=float) > 0, 1.0, 0.0)

        total_samples = len(probs)
        overall_brier = compute_brier_score(probs, outcomes)
        ece = compute_expected_calibration_error(probs, outcomes, n_bins=10)
        pred_p, true_p, counts = compute_calibration_curve(probs, outcomes, n_bins=10)

        # Rolling 30-day and 90-day Brier scores
        max_date = resolved_df["date_val"].max()
        cutoff_30d = max_date - datetime.timedelta(days=30)
        cutoff_90d = max_date - datetime.timedelta(days=90)

        mask_30d = resolved_df["date_val"] >= cutoff_30d
        mask_90d = resolved_df["date_val"] >= cutoff_90d

        brier_30d: Optional[float] = None
        if np.sum(mask_30d) >= 5:
            brier_30d = compute_brier_score(probs[mask_30d], outcomes[mask_30d])

        brier_90d: Optional[float] = None
        if np.sum(mask_90d) >= 10:
            brier_90d = compute_brier_score(probs[mask_90d], outcomes[mask_90d])

        # Status determination based on empirical calibration thresholds
        effective_brier = brier_30d if brier_30d is not None else overall_brier
        if effective_brier >= 0.28 or ece >= 0.20:
            status = CalibrationStatus.SEVERE_MISCALIBRATION
            alert_message = (
                f"Severe miscalibration detected: Brier={effective_brier:.3f} "
                f"(threshold 0.28), ECE={ece:.3f}. Recommendation: Recalibrate probabilities "
                f"via Platt scaling or Isotonic regression."
            )
        elif effective_brier >= 0.22 or ece >= 0.10:
            status = CalibrationStatus.MODERATE_MISCALIBRATION
            alert_message = (
                f"Moderate miscalibration warning: Brier={effective_brier:.3f}, ECE={ece:.3f}. "
                f"Predicted probabilities deviate moderately from empirical win rates."
            )
        else:
            status = CalibrationStatus.HEALTHY
            alert_message = (
                f"Probability calibration healthy: Brier={effective_brier:.3f}, ECE={ece:.3f}."
            )

        return CalibrationReport(
            total_samples=total_samples,
            brier_score=overall_brier,
            brier_30d=brier_30d,
            brier_90d=brier_90d,
            ece=ece,
            status=status,
            alert_message=alert_message,
            bin_pred_probs=pred_p.tolist(),
            bin_true_proportions=true_p.tolist(),
            bin_counts=counts.tolist(),
        )
