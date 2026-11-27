"""Pydantic schema for stock research JSON files (schema version 1.0).

The layout is defined in docs/research/antigravity_research_prompt.md. Files live in
data/research/<SYMBOL>_<YYYY-MM-DD>.json next to a human-readable .md report.
"""

import datetime
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

RESEARCH_SCHEMA_VERSION = "1.0"

Sentiment = Optional[float]
Stance = Literal["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
Horizon = Literal["SHORT", "MEDIUM", "LONG"]
Level = Literal["LOW", "MEDIUM", "HIGH"]
Impact = Literal["POSITIVE", "NEGATIVE", "NEUTRAL", "UNCERTAIN"]

STANCE_VALUES: dict[str, float] = {
    "STRONG_BUY": 1.0,
    "BUY": 0.5,
    "HOLD": 0.0,
    "SELL": -0.5,
    "STRONG_SELL": -1.0,
}


class _Section(BaseModel):
    """Base for nested sections: unknown keys are ignored so small drifts still load."""

    model_config = ConfigDict(extra="ignore")


class PriceContext(_Section):
    last_close_pkr: Optional[float] = None
    last_close_date: Optional[datetime.date] = None
    high_52w_pkr: Optional[float] = None
    low_52w_pkr: Optional[float] = None
    market_cap_pkr_bn: Optional[float] = None
    shares_outstanding_mn: Optional[float] = None
    free_float_pct: Optional[float] = None
    avg_daily_volume_shares: Optional[float] = None


class Scores(_Section):
    overall_sentiment: Sentiment = Field(default=None, ge=-1.0, le=1.0)
    news_sentiment: Sentiment = Field(default=None, ge=-1.0, le=1.0)
    outlook_short_term: Sentiment = Field(default=None, ge=-1.0, le=1.0)
    outlook_medium_term: Sentiment = Field(default=None, ge=-1.0, le=1.0)
    fundamental_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    valuation_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    growth_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    dividend_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    financial_health_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    management_governance_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    risk_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class Recommendation(_Section):
    stance: Optional[Stance] = None
    horizon: Optional[Horizon] = None
    fair_value_pkr: Optional[float] = None
    fair_value_method: Optional[str] = None
    upside_pct: Optional[float] = None
    rationale: Optional[str] = None


class Fundamentals(_Section):
    fiscal_year_end_month: Optional[int] = Field(default=None, ge=1, le=12)
    latest_period: Optional[str] = None
    latest_period_end: Optional[datetime.date] = None
    revenue_pkr_bn: Optional[float] = None
    revenue_growth_yoy_pct: Optional[float] = None
    net_income_pkr_bn: Optional[float] = None
    net_income_growth_yoy_pct: Optional[float] = None
    eps_pkr: Optional[float] = None
    eps_ttm_pkr: Optional[float] = None
    eps_growth_yoy_pct: Optional[float] = None
    gross_margin_pct: Optional[float] = None
    net_margin_pct: Optional[float] = None
    roe_pct: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    book_value_per_share_pkr: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    source_ids: list[str] = Field(default_factory=list)


class EarningsEntry(_Section):
    period: str
    period_end: Optional[datetime.date] = None
    announcement_date: Optional[datetime.date] = None
    revenue_pkr_bn: Optional[float] = None
    net_income_pkr_bn: Optional[float] = None
    eps_pkr: Optional[float] = None
    eps_yoy_pct: Optional[float] = None
    vs_expectation: Literal["BEAT", "INLINE", "MISS", "UNKNOWN"] = "UNKNOWN"
    source_ids: list[str] = Field(default_factory=list)


class DividendEntry(_Section):
    period: str
    type: Literal["INTERIM", "FINAL", "SPECIAL", "BONUS", "RIGHT"]
    amount_pkr: Optional[float] = None
    bonus_or_right_pct: Optional[float] = None
    announcement_date: Optional[datetime.date] = None
    book_closure_start: Optional[datetime.date] = None
    source_ids: list[str] = Field(default_factory=list)


class Dividends(_Section):
    trailing_12m_dps_pkr: Optional[float] = None
    dividend_yield_pct: Optional[float] = None
    payout_ratio_pct: Optional[float] = None
    history: list[DividendEntry] = Field(default_factory=list)


class UpcomingEvent(_Section):
    date: Optional[datetime.date] = None
    date_is_estimate: bool = False
    event_type: Literal[
        "EARNINGS",
        "BOARD_MEETING",
        "BOOK_CLOSURE",
        "EX_DIVIDEND",
        "AGM",
        "EGM",
        "POLICY_RATE",
        "BUDGET",
        "INDEX_REBALANCE",
        "OTHER",
    ]
    description: str = ""
    expected_impact: Impact = "UNCERTAIN"
    source_ids: list[str] = Field(default_factory=list)


class Catalyst(_Section):
    title: str
    description: str = ""
    direction: Literal["POSITIVE", "NEGATIVE"]
    impact: Level = "MEDIUM"
    probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    timeframe: Horizon = "MEDIUM"
    source_ids: list[str] = Field(default_factory=list)


class Risk(_Section):
    title: str
    category: Literal[
        "REGULATORY",
        "MACRO",
        "FX",
        "COMMODITY",
        "CIRCULAR_DEBT",
        "GOVERNANCE",
        "LIQUIDITY",
        "COMPETITION",
        "POLITICAL",
        "LEGAL",
        "OPERATIONAL",
        "OTHER",
    ] = "OTHER"
    severity: Level = "MEDIUM"
    likelihood: Level = "MEDIUM"
    description: str = ""
    source_ids: list[str] = Field(default_factory=list)


class NewsItem(_Section):
    date: datetime.date
    headline: str
    summary: str = ""
    category: Literal[
        "EARNINGS",
        "DIVIDEND",
        "CORPORATE_ACTION",
        "REGULATORY",
        "MACRO",
        "SECTOR",
        "MANAGEMENT",
        "LEGAL",
        "PROJECT",
        "OTHER",
    ] = "OTHER"
    sentiment: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    importance: Level = "MEDIUM"
    source_ids: list[str] = Field(default_factory=list)


class Peer(_Section):
    symbol: str
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield_pct: Optional[float] = None
    note: Optional[str] = None


class SectorView(_Section):
    name: Optional[str] = None
    outlook: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    key_drivers: list[str] = Field(default_factory=list)
    peers: list[Peer] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class Shareholder(_Section):
    name: str
    stake_pct: Optional[float] = None


class Management(_Section):
    ceo: Optional[str] = None
    chairman: Optional[str] = None
    major_shareholders: list[Shareholder] = Field(default_factory=list)
    recent_changes: list[str] = Field(default_factory=list)
    governance_notes: Optional[str] = None
    source_ids: list[str] = Field(default_factory=list)


class AnalystView(_Section):
    broker: str
    date: Optional[datetime.date] = None
    rating: Literal[
        "BUY", "OVERWEIGHT", "HOLD", "NEUTRAL", "UNDERWEIGHT", "SELL", "UNKNOWN"
    ] = "UNKNOWN"
    target_price_pkr: Optional[float] = None
    source_ids: list[str] = Field(default_factory=list)


class MarketSentiment(_Section):
    forum_sentiment: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    institutional_flows_note: Optional[str] = None
    notes: Optional[str] = None


class ChartHighlight(_Section):
    chart: Literal[
        "SCORES",
        "PRICE_TREND",
        "PRICE_RANGE",
        "VALUATION_VS_PEERS",
        "EARNINGS",
        "DIVIDENDS",
        "RISK_GRID",
        "EVENTS",
        "OTHER",
    ] = "OTHER"
    highlight: str = ""


class Pronunciation(_Section):
    term: str
    say: str


class VideoBrief(_Section):
    """Spoken-word material for the research video (optional; added after schema 1.0)."""

    hook: Optional[str] = None
    story: list[str] = Field(default_factory=list)
    tone: Optional[Literal["UPBEAT", "BALANCED", "CAUTIOUS", "NEGATIVE"]] = None
    chart_highlights: list[ChartHighlight] = Field(default_factory=list)
    pronunciations: list[Pronunciation] = Field(default_factory=list)
    closing_line: Optional[str] = None

    @field_validator("story", mode="before")
    @classmethod
    def _story_as_sentences(cls, value: Any) -> Any:
        """Accept a single paragraph (a common model slip) and split it into sentences."""
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
            return [p.strip() for p in parts if p.strip()]
        return value


class Source(_Section):
    id: str
    title: str = ""
    publisher: Optional[str] = None
    url: Optional[str] = None
    published_date: Optional[datetime.date] = None
    accessed_date: Optional[datetime.date] = None
    type: Literal[
        "PSX_FILING",
        "COMPANY_REPORT",
        "BROKER_REPORT",
        "NEWS",
        "REGULATOR",
        "DATA_PROVIDER",
        "OTHER",
    ] = "OTHER"


class ResearchReport(BaseModel):
    """One research snapshot of one stock on one date."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str
    symbol: str
    company_name: Optional[str] = None
    sector: Optional[str] = None
    research_date: datetime.date
    data_as_of: Optional[datetime.date] = None
    generated_by: Optional[str] = None
    model_name: Optional[str] = None

    price_context: PriceContext = Field(default_factory=PriceContext)
    scores: Scores
    recommendation: Recommendation = Field(default_factory=Recommendation)
    fundamentals: Fundamentals = Field(default_factory=Fundamentals)
    earnings: list[EarningsEntry] = Field(default_factory=list)
    dividends: Dividends = Field(default_factory=Dividends)
    upcoming_events: list[UpcomingEvent] = Field(default_factory=list)
    catalysts: list[Catalyst] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    news: list[NewsItem] = Field(default_factory=list)
    sector_view: SectorView = Field(default_factory=SectorView)
    management: Management = Field(default_factory=Management)
    analyst_views: list[AnalystView] = Field(default_factory=list)
    market_sentiment: MarketSentiment = Field(default_factory=MarketSentiment)
    key_takeaways: list[str] = Field(default_factory=list)
    video_brief: Optional[VideoBrief] = None
    data_gaps: list[str] = Field(default_factory=list)
    notes: Optional[str] = None
    sources: list[Source] = Field(default_factory=list)

    @property
    def stance_value(self) -> Optional[float]:
        """Recommendation stance mapped to -1..1 (None when missing)."""
        stance = self.recommendation.stance
        return STANCE_VALUES.get(stance) if stance else None


EXPECTED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(ResearchReport.model_fields.keys())
# Keys added after files were already being written; their absence is not worth a warning.
OPTIONAL_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"video_brief"})
