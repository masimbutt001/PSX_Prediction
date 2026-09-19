"""Plotly interactive visualization components for the PSX Predictor Dashboard."""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_candlestick_chart(
    df: pd.DataFrame,
    symbol: str,
    show_volume: bool = True,
    show_ma: bool = True,
    show_bb: bool = True,
) -> go.Figure:
    """Generate interactive multi-panel candlestick chart with indicators and volume."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No price data available for {symbol}",
            template="plotly_dark",
        )
        return fig

    d_col = "trade_date" if "trade_date" in df.columns else "date"
    dates = df[d_col]

    if show_volume and "volume" in df.columns:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.75, 0.25],
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    # 1. Main Candlestick Trace
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=f"{symbol} Price",
            increasing_line_color="#10b981",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )

    # 2. Moving Average Overlays
    if show_ma and len(df) >= 20:
        sma_20 = df["close"].rolling(20).mean()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=sma_20,
                mode="lines",
                name="SMA 20",
                line={"color": "#f59e0b", "width": 1.5},
            ),
            row=1,
            col=1,
        )

    if show_ma and len(df) >= 50:
        sma_50 = df["close"].rolling(50).mean()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=sma_50,
                mode="lines",
                name="SMA 50",
                line={"color": "#38bdf8", "width": 1.5},
            ),
            row=1,
            col=1,
        )

    # 3. Bollinger Bands Overlay
    if show_bb and len(df) >= 20:
        roll_mean = df["close"].rolling(20).mean()
        roll_std = df["close"].rolling(20).std()
        bb_upper = roll_mean + (roll_std * 2.0)
        bb_lower = roll_mean - (roll_std * 2.0)

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=bb_upper,
                mode="lines",
                name="BB Upper",
                line={"color": "rgba(148, 163, 184, 0.4)", "width": 1, "dash": "dot"},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=bb_lower,
                mode="lines",
                name="BB Lower",
                fill="tonexty",
                fillcolor="rgba(148, 163, 184, 0.05)",
                line={"color": "rgba(148, 163, 184, 0.4)", "width": 1, "dash": "dot"},
            ),
            row=1,
            col=1,
        )

    # 4. Volume Bars
    if show_volume and "volume" in df.columns:
        colors = np.where(df["close"] >= df["open"], "#10b981", "#ef4444")
        fig.add_trace(
            go.Bar(
                x=dates,
                y=df["volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.8,
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    # Formatting and theme
    fig.update_layout(
        title=f"{symbol} — Interactive Technical Analysis",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#21262d")
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor="#21262d", title_text="Price (PKR)", row=1, col=1
    )

    return fig


def create_equity_curve_chart(
    equity_df: pd.DataFrame,
    symbol: str,
    benchmark_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Generate strategy equity curve comparison chart in PKR."""
    fig = go.Figure()

    if equity_df.empty:
        fig.update_layout(title=f"No equity data for {symbol}", template="plotly_dark")
        return fig

    d_col = "date" if "date" in equity_df.columns else "trade_date"
    val_col = "portfolio_value" if "portfolio_value" in equity_df.columns else "equity"

    fig.add_trace(
        go.Scatter(
            x=equity_df[d_col],
            y=equity_df[val_col],
            mode="lines",
            name=f"{symbol} Strategy",
            line={"color": "#10b981", "width": 2.5},
        )
    )

    if benchmark_df is not None and not benchmark_df.empty:
        b_d_col = "date" if "date" in benchmark_df.columns else "trade_date"
        b_val_col = "portfolio_value" if "portfolio_value" in benchmark_df.columns else "equity"
        fig.add_trace(
            go.Scatter(
                x=benchmark_df[b_d_col],
                y=benchmark_df[b_val_col],
                mode="lines",
                name="Buy & Hold Benchmark",
                line={"color": "#94a3b8", "width": 1.8, "dash": "dash"},
            )
        )

    fig.update_layout(
        title=f"{symbol} — Cumulative Equity Progression (PKR)",
        template="plotly_dark",
        hovermode="x unified",
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#21262d")
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor="#21262d", title_text="Portfolio Value (PKR)"
    )

    return fig


def create_drawdown_chart(
    drawdown_series: pd.Series,
    dates: pd.Series,
) -> go.Figure:
    """Generate underwater drawdown curve chart."""
    fig = go.Figure()

    dd_pct = drawdown_series * 100.0 if drawdown_series.abs().max() <= 1.0 else drawdown_series
    # Ensure drawdowns are displayed as negative or non-positive
    dd_vals = -np.abs(dd_pct.values)

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=dd_vals,
            mode="lines",
            name="Drawdown",
            line={"color": "#ef4444", "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.2)",
        )
    )

    fig.update_layout(
        title="Underwater Drawdown Profile",
        template="plotly_dark",
        hovermode="x unified",
        margin={"l": 40, "r": 40, "t": 50, "b": 40},
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#21262d")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#21262d", title_text="Drawdown (%)")

    return fig
