"""System health monitoring, data feed freshness, and predictive model integrity checks."""

import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from psx_predictor.config.loader import load_config
from psx_predictor.monitoring.calibration import CalibrationStatus, CalibrationTracker
from psx_predictor.monitoring.drift import DriftStatus, FeatureDriftDetector
from psx_predictor.predictions.auditor import PredictionAuditor
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import ensure_directories


class HealthStatus(str, Enum):
    """Overall operational health grade for data feeds, models, and drift."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


@dataclass
class DataFeedSummary:
    """Summary of raw/processed market data freshness, gaps, and coverage."""

    total_symbols: int
    healthy_symbols: int
    stale_symbols: list[str]
    missing_session_gaps: dict[str, int]
    latest_date: Optional[str]
    status: HealthStatus
    details: str

    def to_dict(self) -> dict[str, Any]:
        """Convert data feed health summary to serializable dictionary."""
        return {
            "total_symbols": self.total_symbols,
            "healthy_symbols": self.healthy_symbols,
            "stale_symbols": self.stale_symbols,
            "missing_session_gaps": self.missing_session_gaps,
            "latest_date": self.latest_date,
            "status": self.status.value,
            "details": self.details,
        }


@dataclass
class ModelAuditSummary:
    """Summary of model prediction audit outcomes, hit rates, and calibration."""

    total_predictions: int
    resolved_predictions: int
    pending_predictions: int
    hit_rate: float
    brier_score: Optional[float]
    calibration_status: str
    status: HealthStatus

    def to_dict(self) -> dict[str, Any]:
        """Convert model audit summary to serializable dictionary."""
        return {
            "total_predictions": self.total_predictions,
            "resolved_predictions": self.resolved_predictions,
            "pending_predictions": self.pending_predictions,
            "hit_rate": round(self.hit_rate, 4),
            "brier_score": round(self.brier_score, 4) if self.brier_score is not None else None,
            "calibration_status": self.calibration_status,
            "status": self.status.value,
        }


@dataclass
class DriftHealthSummary:
    """Summary of active feature drift across evaluated market universe."""

    symbols_evaluated: int
    symbols_alerting: list[str]
    symbols_warning: list[str]
    top_drifting_features: list[dict[str, Any]]
    retraining_recommended: bool
    status: HealthStatus

    def to_dict(self) -> dict[str, Any]:
        """Convert drift summary to serializable dictionary."""
        return {
            "symbols_evaluated": self.symbols_evaluated,
            "symbols_alerting": self.symbols_alerting,
            "symbols_warning": self.symbols_warning,
            "top_drifting_features": self.top_drifting_features,
            "retraining_recommended": self.retraining_recommended,
            "status": self.status.value,
        }


@dataclass
class SystemHealthReport:
    """Consolidated system diagnostic report across data, models, and distributions."""

    timestamp: str
    overall_status: HealthStatus
    data_feed: DataFeedSummary
    model_audit: ModelAuditSummary
    drift: DriftHealthSummary
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert full health report to serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status.value,
            "data_feed": self.data_feed.to_dict(),
            "model_audit": self.model_audit.to_dict(),
            "drift": self.drift.to_dict(),
            "recommendations": self.recommendations,
        }


class SystemHealthMonitor:
    """Orchestrates comprehensive real-time diagnostic checks on the PSX Predictor platform."""

    def __init__(self, storage_paths: Optional[dict[str, Path]] = None) -> None:
        self.config = load_config()
        if storage_paths is None:
            self.storage_paths = ensure_directories(self.config.settings.data_dir)
        else:
            self.storage_paths = storage_paths

        self.auditor = PredictionAuditor(storage_paths=self.storage_paths)
        self.calibration_tracker = CalibrationTracker(storage_paths=self.storage_paths)
        self.drift_detector = FeatureDriftDetector(storage_paths=self.storage_paths)

    def check_data_feed(self, symbols: Optional[list[str]] = None) -> DataFeedSummary:
        """Verify price series availability, freshness, and absence of business-day gaps."""
        target_symbols = symbols or [s.symbol for s in self.config.get_enabled_stocks()]
        if not target_symbols:
            return DataFeedSummary(
                total_symbols=0,
                healthy_symbols=0,
                stale_symbols=[],
                missing_session_gaps={},
                latest_date=None,
                status=HealthStatus.CRITICAL,
                details="No enabled stock symbols configured in settings.",
            )

        healthy_count = 0
        stale_symbols = []
        missing_gaps: dict[str, int] = {}
        all_latest_dates: list[str] = []

        now = datetime.datetime.now(datetime.timezone.utc).date()

        for sym in target_symbols:
            p_file = self.storage_paths["processed_prices"] / f"{sym}.parquet"
            if not p_file.exists():
                stale_symbols.append(sym)
                continue

            try:
                df = read_parquet(p_file)
                if df.empty or "trade_date" not in df.columns:
                    stale_symbols.append(sym)
                    continue

                dates = pd.to_datetime(df["trade_date"]).sort_values().reset_index(drop=True)
                latest_dt = dates.iloc[-1].date()
                all_latest_dates.append(str(latest_dt))

                # Check if data is older than 5 calendar days
                days_old = (now - latest_dt).days
                if days_old > 5:
                    stale_symbols.append(sym)
                else:
                    healthy_count += 1

                # Check for gaps > 4 business days between consecutive dates
                if len(dates) >= 2:
                    diffs = (dates.diff().dt.days).dropna()
                    big_gaps = int(np.sum(diffs > 4))
                    if big_gaps > 0:
                        missing_gaps[sym] = big_gaps

            except Exception as exc:
                logger.warning(f"Could not read price file for {sym}: {exc}")
                stale_symbols.append(sym)

        overall_latest = max(all_latest_dates) if all_latest_dates else None

        if len(stale_symbols) == len(target_symbols):
            status = HealthStatus.CRITICAL
            details = "All price feeds are unavailable or out-of-date."
        elif len(stale_symbols) > 0 or len(missing_gaps) > 0:
            status = HealthStatus.DEGRADED
            details = f"{len(stale_symbols)} symbols stale, {len(missing_gaps)} have gaps."
        else:
            status = HealthStatus.HEALTHY
            details = (
                f"All {healthy_count} active symbol feeds up-to-date through {overall_latest}."
            )

        return DataFeedSummary(
            total_symbols=len(target_symbols),
            healthy_symbols=healthy_count,
            stale_symbols=stale_symbols,
            missing_session_gaps=missing_gaps,
            latest_date=overall_latest,
            status=status,
            details=details,
        )

    def check_model_audit(self, symbol: Optional[str] = None) -> ModelAuditSummary:
        """Inspect prediction accuracy, Brier score, and calibration reliability."""
        _, audit_summary = self.auditor.reconcile(symbol=symbol)
        cal_report = self.calibration_tracker.evaluate_predictions(symbol=symbol)

        if audit_summary.total_records == 0:
            return ModelAuditSummary(
                total_predictions=0,
                resolved_predictions=0,
                pending_predictions=0,
                hit_rate=0.0,
                brier_score=None,
                calibration_status="NO_DATA",
                status=HealthStatus.HEALTHY,
            )

        # Evaluate performance status
        status = HealthStatus.HEALTHY
        if cal_report.status == CalibrationStatus.SEVERE_MISCALIBRATION:
            status = HealthStatus.CRITICAL
        elif (
            cal_report.status == CalibrationStatus.MODERATE_MISCALIBRATION
            or (audit_summary.resolved_records >= 10 and audit_summary.hit_rate < 0.45)
        ):
            status = HealthStatus.DEGRADED

        return ModelAuditSummary(
            total_predictions=audit_summary.total_records,
            resolved_predictions=audit_summary.resolved_records,
            pending_predictions=audit_summary.pending_records,
            hit_rate=audit_summary.hit_rate,
            brier_score=audit_summary.brier_score,
            calibration_status=cal_report.status.value,
            status=status,
        )

    def check_drift(self, symbols: Optional[list[str]] = None) -> DriftHealthSummary:
        """Evaluate Population Stability Index across multi-modal feature matrices."""
        target_symbols = symbols or [s.symbol for s in self.config.get_enabled_stocks()]

        alerting_symbols: list[str] = []
        warning_symbols: list[str] = []
        all_top_features: list[dict[str, Any]] = []
        retraining_flag = False

        evaluated_count = 0
        for sym in target_symbols:
            drift_summary = self.drift_detector.evaluate_symbol_drift(sym)
            if drift_summary.total_features > 0:
                evaluated_count += 1
                if drift_summary.overall_status == DriftStatus.ALERT:
                    alerting_symbols.append(sym)
                    retraining_flag = True
                elif drift_summary.overall_status == DriftStatus.WARNING:
                    warning_symbols.append(sym)

                for f in drift_summary.top_drifting[:3]:
                    all_top_features.append(
                        {
                            "symbol": sym,
                            "feature": f.feature_name,
                            "psi": f.psi,
                            "status": f.status.value,
                        }
                    )

        # Sort top drifting across universe
        all_top_features = sorted(all_top_features, key=lambda x: x["psi"], reverse=True)[:5]

        if alerting_symbols:
            status = HealthStatus.CRITICAL
        elif warning_symbols:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        return DriftHealthSummary(
            symbols_evaluated=evaluated_count,
            symbols_alerting=alerting_symbols,
            symbols_warning=warning_symbols,
            top_drifting_features=all_top_features,
            retraining_recommended=retraining_flag,
            status=status,
        )

    def run_health_check(self, symbols: Optional[list[str]] = None) -> SystemHealthReport:
        """Execute end-to-end platform health inspection and formulate recommendations."""
        data_feed = self.check_data_feed(symbols=symbols)
        model_audit = self.check_model_audit()
        drift = self.check_drift(symbols=symbols)

        # Determine overall status
        statuses = [data_feed.status, model_audit.status, drift.status]
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        recommendations: list[str] = []
        if data_feed.status != HealthStatus.HEALTHY:
            recommendations.append(
                "Execute 'psx data update' or 'psx schedule run-daily' to refresh stale feeds."
            )
        if model_audit.status == HealthStatus.CRITICAL:
            recommendations.append(
                "Severe prediction miscalibration. Recalibrate ensemble model probabilities."
            )
        if drift.retraining_recommended:
            alert_str = ", ".join(drift.symbols_alerting[:3])
            recommendations.append(
                f"Significant feature drift detected on {len(drift.symbols_alerting)} symbols "
                f"({alert_str}). Run 'psx train' to update model weights."
            )
        if not recommendations:
            recommendations.append("All systems operational. No remediation required.")

        return SystemHealthReport(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            overall_status=overall_status,
            data_feed=data_feed,
            model_audit=model_audit,
            drift=drift,
            recommendations=recommendations,
        )
