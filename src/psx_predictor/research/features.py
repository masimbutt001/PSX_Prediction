"""Point-in-time, recency-decayed research features.

A research file dated D is usable for every session on or after D (predictions are made
after the close, so same-day research is known in time). Its influence halves every
``half_life_days`` and drops to zero after ``max_age_days``. Each score is scaled by
that weight and by the research's own confidence, so old or low-confidence research
moves the features toward neutral (0).
"""

import datetime
from collections.abc import Iterable
from typing import Any, Optional

import pandas as pd

from psx_predictor.research.schemas import ResearchReport

DEFAULT_HALF_LIFE_DAYS = 30.0
DEFAULT_MAX_AGE_DAYS = 120
EVENT_HORIZON_DAYS = 30
EVENT_SOON_DAYS = 5
DEFAULT_CONFIDENCE = 0.5

RESEARCH_FEATURE_COLUMNS: list[str] = [
    "research_available",
    "research_age_days",
    "research_weight",
    "research_sentiment",
    "research_news_sentiment",
    "research_outlook_short",
    "research_outlook_medium",
    "research_stance",
    "research_fundamental",
    "research_valuation",
    "research_risk",
    "research_upside",
    "research_days_to_event",
    "research_event_soon",
]


def _to_date(value: Any) -> datetime.date:
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date()
    return pd.Timestamp(value).date()


def _centered(score: Optional[float]) -> float:
    """Map a 0..100 score to -1..1 around 50 (None -> 0)."""
    return 0.0 if score is None else (float(score) - 50.0) / 50.0


def _signed(value: Optional[float]) -> float:
    return 0.0 if value is None else float(value)


def recency_weight(
    age_days: int,
    confidence: Optional[float],
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> float:
    """Exponential recency decay times research confidence."""
    if age_days < 0 or age_days > max_age_days:
        return 0.0
    conf = DEFAULT_CONFIDENCE if confidence is None else float(confidence)
    return float(0.5 ** (age_days / half_life_days)) * conf


def _neutral_row(max_age_days: int) -> dict[str, float]:
    row = {col: 0.0 for col in RESEARCH_FEATURE_COLUMNS}
    row["research_age_days"] = float(max_age_days)
    row["research_days_to_event"] = float(EVENT_HORIZON_DAYS)
    return row


def _days_to_next_event(reports: list[ResearchReport], session: datetime.date) -> Optional[int]:
    best: Optional[int] = None
    for report in reports:
        for event in report.upcoming_events:
            if event.date is None or event.date < session:
                continue
            days = (event.date - session).days
            if best is None or days < best:
                best = days
    return best


def _row_for_session(
    reports: list[ResearchReport],
    session: datetime.date,
    half_life_days: float,
    max_age_days: int,
) -> dict[str, float]:
    row = _neutral_row(max_age_days)
    known = [r for r in reports if r.research_date <= session]
    if not known:
        return row

    days_to_event = _days_to_next_event(
        [r for r in known if (session - r.research_date).days <= max_age_days], session
    )
    if days_to_event is not None and days_to_event <= EVENT_HORIZON_DAYS:
        row["research_days_to_event"] = float(days_to_event)
        row["research_event_soon"] = 1.0 if days_to_event <= EVENT_SOON_DAYS else 0.0

    latest = known[-1]
    age = (session - latest.research_date).days
    weight = recency_weight(age, latest.scores.confidence, half_life_days, max_age_days)
    if weight <= 0.0:
        return row

    s = latest.scores
    upside = latest.recommendation.upside_pct
    upside_unit = 0.0 if upside is None else max(-1.0, min(1.0, float(upside) / 100.0))

    row.update(
        {
            "research_available": 1.0,
            "research_age_days": float(age),
            "research_weight": weight,
            "research_sentiment": _signed(s.overall_sentiment) * weight,
            "research_news_sentiment": _signed(s.news_sentiment) * weight,
            "research_outlook_short": _signed(s.outlook_short_term) * weight,
            "research_outlook_medium": _signed(s.outlook_medium_term) * weight,
            "research_stance": _signed(latest.stance_value) * weight,
            "research_fundamental": _centered(s.fundamental_score) * weight,
            "research_valuation": _centered(s.valuation_score) * weight,
            "research_risk": _centered(s.risk_score) * weight,
            "research_upside": upside_unit * weight,
        }
    )
    return row


def build_research_features(
    session_dates: Iterable[Any],
    reports: list[ResearchReport],
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> pd.DataFrame:
    """Research feature frame aligned row-by-row with ``session_dates``.

    Rows with no usable research get neutral values (0, age = max_age_days,
    days_to_event = 30), so the columns are always present.
    """
    ordered = sorted(reports, key=lambda r: r.research_date)
    rows = [
        _row_for_session(ordered, _to_date(d), half_life_days, max_age_days)
        for d in session_dates
    ]
    return pd.DataFrame(rows, columns=RESEARCH_FEATURE_COLUMNS)


def research_context_for_date(
    reports: list[ResearchReport],
    session: Any,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> dict[str, Any]:
    """Human-facing summary of the research that applies to a session (empty if none)."""
    day = _to_date(session)
    known = sorted(
        (r for r in reports if r.research_date <= day), key=lambda r: r.research_date
    )
    if not known:
        return {}
    latest = known[-1]
    age = (day - latest.research_date).days
    weight = recency_weight(age, latest.scores.confidence, half_life_days, max_age_days)

    upcoming = sorted(
        (e for e in latest.upcoming_events if e.date is not None and e.date >= day),
        key=lambda e: e.date or day,
    )
    next_event = (
        {
            "date": str(upcoming[0].date),
            "type": upcoming[0].event_type,
            "description": upcoming[0].description,
        }
        if upcoming
        else None
    )
    level = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    top_risks = sorted(latest.risks, key=lambda r: -level[r.severity])[:3]
    top_catalysts = sorted(latest.catalysts, key=lambda c: -level[c.impact])[:3]

    return {
        "research_date": str(latest.research_date),
        "age_days": age,
        "weight": round(weight, 4),
        "stale": weight == 0.0,
        "stance": latest.recommendation.stance,
        "overall_sentiment": latest.scores.overall_sentiment,
        "outlook_short_term": latest.scores.outlook_short_term,
        "outlook_medium_term": latest.scores.outlook_medium_term,
        "risk_score": latest.scores.risk_score,
        "confidence": latest.scores.confidence,
        "fair_value_pkr": latest.recommendation.fair_value_pkr,
        "upside_pct": latest.recommendation.upside_pct,
        "next_event": next_event,
        "top_risks": [r.title for r in top_risks],
        "top_catalysts": [f"{c.title} ({c.direction.lower()})" for c in top_catalysts],
    }
