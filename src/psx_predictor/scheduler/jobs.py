"""Pipeline step definitions, task records, and cron schedule trigger mappings."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from apscheduler.triggers.cron import CronTrigger


class DailyPipelineStep(str, Enum):
    """Enumeration of standard daily EOD pipeline tasks in topological order."""

    UPDATE_PRICES = "update_prices"
    COLLECT_NEWS = "collect_news"
    UPDATE_MACRO = "update_macro"
    BUILD_FEATURES = "build_features"
    GENERATE_PREDICTIONS = "generate_predictions"
    AUDIT_PREDICTIONS = "audit_predictions"


@dataclass
class PipelineTaskResult:
    """Execution status and performance summary for an individual pipeline step."""

    step: DailyPipelineStep
    status: str  # "SUCCESS", "FAILED", "SKIPPED"
    duration_seconds: float
    message: str
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert task result to serializable dictionary."""
        return {
            "step": self.step.value,
            "status": self.status,
            "duration_seconds": round(self.duration_seconds, 2),
            "message": self.message,
            "error": self.error,
        }


@dataclass
class PipelineRunReport:
    """Consolidated end-to-end execution audit report for a daily pipeline run."""

    started_at: str
    ended_at: str
    duration_seconds: float
    success: bool
    tasks: list[PipelineTaskResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert full run report to serializable dictionary."""
        return {
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": round(self.duration_seconds, 2),
            "success": self.success,
            "tasks": [t.to_dict() for t in self.tasks],
        }


def create_cron_triggers(timezone: str = "Asia/Karachi") -> list[CronTrigger]:
    """Construct APScheduler CronTrigger instances mapped to PSX operational sessions.

    PSX operating schedule:
      - Monday to Thursday: Market close at 15:30 PKT -> EOD batch run at 16:00 PKT.
      - Friday: Market close at 16:30 PKT -> EOD batch run at 17:00 PKT.

    Args:
        timezone: Local timezone string (default 'Asia/Karachi').

    Returns:
        List containing Mon-Thu and Friday CronTrigger instances.
    """
    mon_thu_trigger = CronTrigger(
        day_of_week="mon-thu",
        hour=16,
        minute=0,
        timezone=timezone,
    )
    fri_trigger = CronTrigger(
        day_of_week="fri",
        hour=17,
        minute=0,
        timezone=timezone,
    )
    return [mon_thu_trigger, fri_trigger]
