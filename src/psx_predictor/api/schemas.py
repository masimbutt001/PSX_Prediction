"""Pydantic request and response schemas for the PSX Predictor FastAPI backend."""

from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Service health and diagnostic status."""

    status: str = Field(default="ok", description="Overall platform status ('ok', 'degraded')")
    version: str = Field(..., description="Platform semantic version")
    timestamp: str = Field(..., description="Current ISO-8601 server timestamp")
    database_connected: bool = Field(default=True, description="Local analytical store access")
    total_configured_stocks: int = Field(default=0, description="Count of universe stocks")


class StockSummary(BaseModel):
    """Metadata and status for an individual PSX stock."""

    symbol: str = Field(..., description="PSX ticker symbol (e.g. 'OGDC')")
    name: str = Field(..., description="Company name")
    sector: str = Field(..., description="Industry sector classification")
    is_enabled: bool = Field(default=True, description="Whether active for modeling")
    has_data: bool = Field(default=False, description="Whether historical data exists locally")
    total_sessions: Optional[int] = Field(default=None, description="Count of historical sessions")
    latest_date: Optional[str] = Field(default=None, description="Latest available session date")


class StockHistoryPoint(BaseModel):
    """Single-session OHLCV price observation."""

    date: str = Field(..., description="Trading date in YYYY-MM-DD")
    open: float = Field(..., description="Opening price in PKR")
    high: float = Field(..., description="Session high price in PKR")
    low: float = Field(..., description="Session low price in PKR")
    close: float = Field(..., description="Closing price in PKR")
    volume: int = Field(..., description="Total share volume traded")
    adjusted_close: Optional[float] = Field(default=None, description="Corporate actions adjusted")


class StockHistoryResponse(BaseModel):
    """Time-series OHLCV history for a stock."""

    symbol: str = Field(..., description="PSX ticker symbol")
    total_sessions: int = Field(..., description="Total sessions returned")
    history: list[StockHistoryPoint] = Field(default_factory=list)


class FeatureDriver(BaseModel):
    """Feature importance driver explaining model prediction."""

    name: str = Field(..., description="Indicator or feature column name")
    importance: float = Field(..., description="Relative feature contribution")
    value: Optional[float] = Field(default=None, description="Observed feature value")


class LatestPredictionResponse(BaseModel):
    """Latest market directional prediction and confidence."""

    symbol: str = Field(..., description="PSX ticker symbol")
    session_date: str = Field(..., description="Session date for forecast")
    prediction_date: str = Field(..., description="Timestamp forecast was computed")
    model_name: str = Field(..., description="Predictive model identifier")
    target_name: str = Field(..., description="Target variable being predicted")
    predicted_direction: int = Field(..., description="Predicted direction: 1 (UP) or 0 (DOWN)")
    up_probability: float = Field(..., description="Calibrated probability of UP return")
    signal: str = Field(..., description="Trading signal recommendation: 'BUY', 'SELL', 'HOLD'")
    confidence: float = Field(..., description="Confidence metric derived from probability margin")
    modality_contributions: Optional[dict[str, float]] = Field(
        default=None, description="Learned modality weights (Price, News, Macro)"
    )
    top_drivers: list[FeatureDriver] = Field(default_factory=list)


class PredictionRecordItem(BaseModel):
    """Historical audited prediction record."""

    prediction_id: str = Field(..., description="Unique prediction identifier")
    symbol: str = Field(..., description="PSX ticker symbol")
    target_session_date: str = Field(..., description="Target trading session date")
    created_at: str = Field(..., description="Generation ISO-8601 timestamp")
    model_name: str = Field(..., description="Model identifier")
    predicted_label: int = Field(..., description="Forecasted class label")
    probability: float = Field(..., description="Forecasted positive class probability")
    realized_label: Optional[int] = Field(default=None, description="Actual session outcome")
    is_correct: Optional[bool] = Field(default=None, description="Whether forecast matched outcome")


class EquityPoint(BaseModel):
    """Single point along a backtest equity curve."""

    date: str = Field(..., description="Session date")
    portfolio_value: float = Field(..., description="Total portfolio value in PKR")
    cash: float = Field(..., description="Uninvested cash balance in PKR")
    drawdown: float = Field(..., description="Drawdown percentage from peak")


class BacktestSummaryResponse(BaseModel):
    """Performance summary and simulation metrics for a strategy/model backtest."""

    symbol: str = Field(..., description="PSX ticker symbol")
    strategy_name: str = Field(..., description="Evaluated strategy or model name")
    start_date: str = Field(..., description="Evaluation window start date")
    end_date: str = Field(..., description="Evaluation window end date")
    total_return: float = Field(..., description="Cumulative strategy return")
    cagr: float = Field(..., description="Compound annual growth rate")
    sharpe_ratio: float = Field(..., description="Annualized Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    win_rate: float = Field(..., description="Percentage of winning sessions/trades")
    total_trades: int = Field(..., description="Total trade executions simulated")
    equity_curve: list[EquityPoint] = Field(default_factory=list)
