"""Feature distribution drift detection using Population Stability Index (PSI) and KS-test."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from psx_predictor.config.loader import load_config
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import ensure_directories


class DriftStatus(str, Enum):
    """Distribution drift classification based on regulatory statistical thresholds."""

    STABLE = "STABLE"  # PSI < 0.10
    WARNING = "WARNING"  # 0.10 <= PSI < 0.25
    ALERT = "ALERT"  # PSI >= 0.25 (triggers retraining recommendation)


def compute_psi(
    reference: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Compute Population Stability Index (PSI) measuring distributional shift.

    Formula:
        PSI = sum_{k=1}^K (Actual_k - Expected_k) * ln(Actual_k / Expected_k)

    Thresholds:
        - PSI < 0.10: Insignificant change (STABLE)
        - 0.10 <= PSI < 0.25: Moderate change (WARNING)
        - PSI >= 0.25: Significant change (ALERT)

    Args:
        reference: Baseline reference distribution (e.g., training sample).
        actual: Current target distribution (e.g., live inference window).
        num_bins: Number of quantile bins (default 10).
        epsilon: Small positive constant to avoid division by zero or log(0).

    Returns:
        Non-negative PSI floating point scalar.
    """
    ref = np.asarray(reference, dtype=float)
    act = np.asarray(actual, dtype=float)

    # Filter out NaNs and infs
    clean_ref = ref[np.isfinite(ref)]
    clean_act = act[np.isfinite(act)]

    if len(clean_ref) == 0 or len(clean_act) == 0:
        return 0.0

    # If reference has almost zero variance, return 0.0
    if np.isclose(np.var(clean_ref), 0.0):
        return 0.0

    # Determine quantile bin edges from reference distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(clean_ref, percentiles)

    # Expand bounds to cover entire support
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Handle duplicates in bin edges for low-cardinality discrete data
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    # Count observed samples per bin
    ref_counts, _ = np.histogram(clean_ref, bins=bin_edges)
    act_counts, _ = np.histogram(clean_act, bins=bin_edges)

    # Compute empirical proportions
    ref_pct = ref_counts / len(clean_ref)
    act_pct = act_counts / len(clean_act)

    # Add epsilon smoothing and re-normalize
    ref_pct = np.clip(ref_pct, epsilon, 1.0)
    act_pct = np.clip(act_pct, epsilon, 1.0)

    ref_pct = ref_pct / np.sum(ref_pct)
    act_pct = act_pct / np.sum(act_pct)

    # Compute PSI sum
    psi = np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct))
    return float(max(0.0, psi))


def compute_ks_drift(
    reference: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """Perform two-sample Kolmogorov-Smirnov test between reference and target arrays.

    Args:
        reference: Reference baseline array.
        actual: Target sample array.

    Returns:
        Tuple of (ks_statistic, p_value).
    """
    ref = np.asarray(reference, dtype=float)
    act = np.asarray(actual, dtype=float)

    clean_ref = ref[np.isfinite(ref)]
    clean_act = act[np.isfinite(act)]

    if len(clean_ref) == 0 or len(clean_act) == 0:
        return 0.0, 1.0

    res = ks_2samp(clean_ref, clean_act)
    return float(res.statistic), float(res.pvalue)


@dataclass
class FeatureDriftResult:
    """Statistical drift evaluation for an individual predictive feature."""

    feature_name: str
    psi: float
    ks_stat: float
    ks_p_value: float
    status: DriftStatus
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "feature_name": self.feature_name,
            "psi": round(self.psi, 4),
            "ks_stat": round(self.ks_stat, 4),
            "ks_p_value": round(self.ks_p_value, 4),
            "status": self.status.value,
            "recommendation": self.recommendation,
        }


@dataclass
class DriftSummary:
    """Consolidated summary of feature drift across an entire dataset or symbol."""

    total_features: int
    stable_features: int
    warning_features: int
    alert_features: int
    overall_status: DriftStatus
    retraining_recommended: bool
    top_drifting: list[FeatureDriftResult] = field(default_factory=list)
    feature_results: dict[str, FeatureDriftResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert drift summary to serializable dictionary."""
        return {
            "total_features": self.total_features,
            "stable_features": self.stable_features,
            "warning_features": self.warning_features,
            "alert_features": self.alert_features,
            "overall_status": self.overall_status.value,
            "retraining_recommended": self.retraining_recommended,
            "top_drifting": [r.to_dict() for r in self.top_drifting],
        }


class FeatureDriftDetector:
    """Analyzes distributional divergence between training baseline and recent market sessions."""

    def __init__(self, storage_paths: Optional[dict[str, Path]] = None) -> None:
        if storage_paths is None:
            config = load_config()
            self.storage_paths = ensure_directories(config.settings.data_dir)
        else:
            self.storage_paths = storage_paths

    def evaluate_features(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_columns: Optional[list[str]] = None,
        num_bins: int = 10,
    ) -> DriftSummary:
        """Evaluate PSI and KS drift across multiple numeric feature columns.

        Args:
            reference_df: Historical training baseline dataframe.
            current_df: Recent production or test window dataframe.
            feature_columns: Specific columns to check. If None, all numeric columns used.
            num_bins: Quantile bins for PSI.

        Returns:
            DriftSummary with per-feature statistics and overall retraining recommendation.
        """
        if feature_columns is None:
            # Exclude non-feature or date columns
            exclude_cols = {
                "trade_date",
                "date",
                "symbol",
                "target",
                "target_date",
                "target_direction",
                "target_return",
            }
            num_cols = [
                c
                for c in reference_df.select_dtypes(include=[np.number]).columns
                if c not in exclude_cols
            ]
            # Must also exist in current_df
            feature_columns = [c for c in num_cols if c in current_df.columns]

        if not feature_columns:
            return DriftSummary(
                total_features=0,
                stable_features=0,
                warning_features=0,
                alert_features=0,
                overall_status=DriftStatus.STABLE,
                retraining_recommended=False,
            )

        results: dict[str, FeatureDriftResult] = {}
        stable_count = 0
        warning_count = 0
        alert_count = 0

        for col in feature_columns:
            ref_vals = reference_df[col].dropna().to_numpy()
            cur_vals = current_df[col].dropna().to_numpy()

            if len(ref_vals) < 10 or len(cur_vals) < 5:
                # Insufficient data
                continue

            psi = compute_psi(ref_vals, cur_vals, num_bins=num_bins)
            ks_stat, ks_p = compute_ks_drift(ref_vals, cur_vals)

            if psi >= 0.25:
                status = DriftStatus.ALERT
                rec = "Significant drift. Retrain model or drop unstable feature."
                alert_count += 1
            elif psi >= 0.10:
                status = DriftStatus.WARNING
                rec = "Moderate drift. Monitor feature distribution closely."
                warning_count += 1
            else:
                status = DriftStatus.STABLE
                rec = "Distribution stable."
                stable_count += 1

            results[col] = FeatureDriftResult(
                feature_name=col,
                psi=psi,
                ks_stat=ks_stat,
                ks_p_value=ks_p,
                status=status,
                recommendation=rec,
            )

        total_tested = stable_count + warning_count + alert_count
        # Overall status
        if alert_count > 0 or (total_tested > 0 and (warning_count / total_tested) > 0.4):
            overall_status = DriftStatus.ALERT
            retrain_rec = True
        elif warning_count > 0:
            overall_status = DriftStatus.WARNING
            retrain_rec = False
        else:
            overall_status = DriftStatus.STABLE
            retrain_rec = False

        # Rank top drifting features by PSI descending
        top_drifting = sorted(results.values(), key=lambda r: r.psi, reverse=True)[:10]

        return DriftSummary(
            total_features=total_tested,
            stable_features=stable_count,
            warning_features=warning_count,
            alert_features=alert_count,
            overall_status=overall_status,
            retraining_recommended=retrain_rec,
            top_drifting=top_drifting,
            feature_results=results,
        )

    def evaluate_symbol_drift(
        self,
        symbol: str,
        recent_sessions: int = 60,
        reference_sessions: int = 250,
    ) -> DriftSummary:
        """Evaluate recent feature drift against historical baseline for a specific symbol.

        Args:
            symbol: Ticker symbol (e.g. 'OGDC').
            recent_sessions: Number of trailing sessions to treat as current target.
            reference_sessions: Number of historical sessions to treat as baseline.

        Returns:
            DriftSummary for the symbol.
        """
        combined_file = self.storage_paths.get(
            "features_combined", self.storage_paths["root"] / "features" / "combined"
        ) / f"{symbol}.parquet"

        if combined_file.exists():
            target_file = combined_file
        else:
            tech_file = self.storage_paths.get(
                "features_technical", self.storage_paths["root"] / "features" / "technical"
            ) / f"{symbol}.parquet"
            target_file = tech_file

        if not target_file.exists():
            return DriftSummary(
                total_features=0,
                stable_features=0,
                warning_features=0,
                alert_features=0,
                overall_status=DriftStatus.STABLE,
                retraining_recommended=False,
            )

        df = read_parquet(target_file)
        if len(df) < (recent_sessions + 20):
            return DriftSummary(
                total_features=0,
                stable_features=0,
                warning_features=0,
                alert_features=0,
                overall_status=DriftStatus.STABLE,
                retraining_recommended=False,
            )

        # Sort chronologically
        if "trade_date" in df.columns:
            df = df.sort_values("trade_date").reset_index(drop=True)

        current_df = df.tail(recent_sessions)
        start_idx = max(0, len(df) - recent_sessions - reference_sessions)
        end_idx = len(df) - recent_sessions
        reference_df = df.iloc[start_idx:end_idx]

        return self.evaluate_features(reference_df, current_df)
