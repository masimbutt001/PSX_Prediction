"""Table and chart helpers for the Research dashboard page."""

import datetime
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from psx_predictor.research.features import recency_weight
from psx_predictor.research.store import ResearchFile

SCORE_HISTORY_COLUMNS = [
    "overall_sentiment",
    "news_sentiment",
    "outlook_short_term",
    "outlook_medium_term",
]


def files_table(files: list[ResearchFile]) -> pd.DataFrame:
    """One row per research file with headline scores and validation status."""
    rows = []
    for rf in files:
        rep = rf.report
        rows.append(
            {
                "Symbol": rf.symbol,
                "Research date": str(rf.research_date or "?"),
                "Stance": rep.recommendation.stance if rep else None,
                "Sentiment": rep.scores.overall_sentiment if rep else None,
                "Outlook 1-4w": rep.scores.outlook_short_term if rep else None,
                "Outlook 3-12m": rep.scores.outlook_medium_term if rep else None,
                "Risk": rep.scores.risk_score if rep else None,
                "Valuation": rep.scores.valuation_score if rep else None,
                "Confidence": rep.scores.confidence if rep else None,
                "Status": "OK" if rf.is_valid else "INVALID",
                "File": rf.path.name,
            }
        )
    return pd.DataFrame(rows)


def overview_table(
    files: list[ResearchFile], today: Optional[datetime.date] = None
) -> pd.DataFrame:
    """Latest valid research per symbol, ranked by recency-weighted outlook."""
    today = today or datetime.date.today()
    latest: dict[str, ResearchFile] = {}
    for rf in files:
        if not rf.is_valid or rf.research_date is None:
            continue
        cur = latest.get(rf.symbol)
        if cur is None or (cur.research_date or datetime.date.min) < rf.research_date:
            latest[rf.symbol] = rf

    rows = []
    for sym, rf in latest.items():
        rep = rf.report
        assert rep is not None
        age = (today - rep.research_date).days
        weight = recency_weight(age, rep.scores.confidence)
        outlook = rep.scores.outlook_short_term or 0.0
        rows.append(
            {
                "Symbol": sym,
                "Research date": str(rep.research_date),
                "Age (days)": age,
                "Stance": rep.recommendation.stance,
                "Upside %": rep.recommendation.upside_pct,
                "Sentiment": rep.scores.overall_sentiment,
                "Outlook 1-4w": rep.scores.outlook_short_term,
                "Risk": rep.scores.risk_score,
                "Confidence": rep.scores.confidence,
                "Weighted outlook": round(outlook * weight, 3),
                "Stale": weight == 0.0 or age > 30,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Weighted outlook", ascending=False).reset_index(drop=True)
    return df


def score_history(files: list[ResearchFile]) -> pd.DataFrame:
    """Score time series for one symbol's valid research files, oldest first."""
    rows = []
    for rf in files:
        if not rf.is_valid or rf.report is None:
            continue
        s = rf.report.scores
        rows.append(
            {
                "research_date": pd.Timestamp(rf.report.research_date),
                **{c: getattr(s, c) for c in SCORE_HISTORY_COLUMNS},
                "risk_score": s.risk_score,
                "valuation_score": s.valuation_score,
                "fundamental_score": s.fundamental_score,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("research_date").reset_index(drop=True)
    return df


def create_score_history_chart(history: pd.DataFrame, symbol: str) -> go.Figure:
    """Line chart of sentiment and outlook scores (-1..1) across research dates."""
    fig = go.Figure()
    labels = {
        "overall_sentiment": "Overall sentiment",
        "news_sentiment": "News sentiment",
        "outlook_short_term": "Outlook 1-4 weeks",
        "outlook_medium_term": "Outlook 3-12 months",
    }
    for col, label in labels.items():
        if col in history.columns:
            fig.add_trace(
                go.Scatter(
                    x=history["research_date"],
                    y=history[col],
                    mode="lines+markers",
                    name=label,
                )
            )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=f"{symbol} research scores over time",
        yaxis={"range": [-1.05, 1.05], "title": "Score (-1 to 1)"},
        xaxis={"title": "Research date"},
        height=380,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        legend={"orientation": "h", "y": -0.25},
    )
    return fig
