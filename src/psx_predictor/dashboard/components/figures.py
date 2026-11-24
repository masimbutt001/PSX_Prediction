"""Themed Plotly figures for the redesigned dashboard pages."""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from psx_predictor.dashboard.theme import MONO_FAMILY, Mode, palette, style_figure
from psx_predictor.monitoring.calibration import compute_calibration_curve

EQUITY_COLUMNS = ("total_equity", "portfolio_value", "equity")


def _equity_col(df: pd.DataFrame) -> str:
    for col in EQUITY_COLUMNS:
        if col in df.columns:
            return col
    raise KeyError(f"Equity curve needs one of {EQUITY_COLUMNS}; found {list(df.columns)}")


def _date_col(df: pd.DataFrame) -> str:
    return "trade_date" if "trade_date" in df.columns else "date"


def price_chart(
    df: pd.DataFrame,
    symbol: str,
    sma20: bool = True,
    sma50: bool = True,
    bollinger: bool = False,
    volume: bool = True,
    forecast: Optional[tuple[str, float]] = None,
    mode: Optional[Mode] = None,
) -> go.Figure:
    """Candlesticks with optional SMA 20/50, Bollinger bands, volume and next-session marker.

    `forecast` is (target_date, up_probability); it adds a shaded marker after the last bar.
    """
    p = palette(mode)
    rows = 2 if volume else 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22] if volume else [1.0],
    )
    if df.empty:
        return style_figure(fig, height=460, mode=mode)

    x = pd.to_datetime(df[_date_col(df)])
    close = df["close"].astype(float)
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=close,
            name=symbol,
            increasing={"line": {"color": p.up, "width": 1}, "fillcolor": p.up},
            decreasing={"line": {"color": p.down, "width": 1}, "fillcolor": p.down},
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    if sma20:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=close.rolling(20).mean(),
                name="SMA 20",
                line={"color": p.sma20, "width": 1.6},
            ),
            row=1,
            col=1,
        )
    if sma50:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=close.rolling(50).mean(),
                name="SMA 50",
                line={"color": p.sma50, "width": 1.6},
            ),
            row=1,
            col=1,
        )
    if bollinger:
        mid = close.rolling(20).mean()
        sd = close.rolling(20).std()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mid + 2 * sd,
                name="Bollinger upper",
                line={"color": p.muted, "width": 1, "dash": "dot"},
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mid - 2 * sd,
                name="Bollinger bands",
                fill="tonexty",
                fillcolor=p.hold_soft,
                line={"color": p.muted, "width": 1, "dash": "dot"},
            ),
            row=1,
            col=1,
        )
    if volume and "volume" in df.columns:
        colors = np.where(close >= df["open"].astype(float), p.up, p.down)
        fig.add_trace(
            go.Bar(
                x=x,
                y=df["volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.5,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    if forecast is not None:
        target, up_prob = forecast
        tx = pd.Timestamp(target)
        color = p.up if up_prob >= 0.55 else p.down if up_prob <= 0.45 else p.muted
        fig.add_vrect(
            x0=x.iloc[-1] + pd.Timedelta(hours=12),
            x1=tx + pd.Timedelta(hours=12),
            fillcolor=p.accent_soft,
            line_width=0,
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[tx],
                y=[float(close.iloc[-1])],
                mode="markers+text",
                marker={
                    "symbol": "triangle-up" if up_prob >= 0.5 else "triangle-down",
                    "size": 14,
                    "color": color,
                },
                text=[f"{up_prob:.0%} UP"],
                textposition="top left",
                textfont={"color": color, "family": MONO_FAMILY},
                name="Next session",
                hovertemplate="Next session %{x|%d %b}: "
                f"{up_prob:.0%} chance of UP<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.update_layout(xaxis_rangeslider_visible=False, dragmode="zoom")
    fig.update_xaxes(rangebreaks=[{"bounds": ["sat", "mon"]}])
    style_figure(fig, height=520 if volume else 440, mode=mode)
    fig.update_yaxes(title_text="PKR", row=1, col=1)
    return fig


def probability_gauge(up_prob: float, mode: Optional[Mode] = None) -> go.Figure:
    """Ring showing the chance the next close is higher."""
    p = palette(mode)
    color = p.up if up_prob >= 0.55 else p.down if up_prob <= 0.45 else p.muted
    fig = go.Figure(
        go.Pie(
            values=[up_prob, 1 - up_prob],
            hole=0.78,
            sort=False,
            direction="clockwise",
            marker={"colors": [color, p.surface2]},
            textinfo="none",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_annotation(
        text=f"<b>{up_prob:.0%}</b>",
        x=0.5,
        y=0.56,
        showarrow=False,
        font={"size": 26, "family": MONO_FAMILY, "color": p.text},
    )
    fig.add_annotation(
        text="chance of UP",
        x=0.5,
        y=0.36,
        showarrow=False,
        font={"size": 11, "color": p.muted},
    )
    style_figure(fig, height=170, mode=mode)
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    return fig


def drivers_chart(drivers: dict[str, float], mode: Optional[Mode] = None) -> go.Figure:
    """Horizontal bars of the top prediction drivers (positive pushes UP)."""
    p = palette(mode)
    items = sorted(drivers.items(), key=lambda kv: abs(float(kv[1])))[-6:]
    names = [k.replace("_", " ") for k, _ in items]
    vals = [float(v) for _, v in items]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=[p.up if v >= 0 else p.down for v in vals],
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    style_figure(fig, height=40 + 34 * max(1, len(items)), mode=mode)
    fig.update_layout(hovermode="closest", bargap=0.35)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont={"family": None})
    return fig


def equity_chart(
    strategy: pd.DataFrame,
    benchmark: Optional[pd.DataFrame] = None,
    strategy_name: str = "Strategy",
    mode: Optional[Mode] = None,
) -> go.Figure:
    """Portfolio value of a backtest against buy and hold."""
    p = palette(mode)
    fig = go.Figure()
    if strategy.empty:
        return style_figure(fig, height=340, mode=mode)
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(strategy[_date_col(strategy)]),
            y=strategy[_equity_col(strategy)],
            name=strategy_name,
            line={"color": p.accent, "width": 2.4},
            fill="tozeroy",
            fillcolor=p.accent_soft,
        )
    )
    if benchmark is not None and not benchmark.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(benchmark[_date_col(benchmark)]),
                y=benchmark[_equity_col(benchmark)],
                name="Buy and hold",
                line={"color": p.muted, "width": 1.4},
            )
        )
    style_figure(fig, height=340, mode=mode)
    vals = strategy[_equity_col(strategy)].astype(float)
    if benchmark is not None and not benchmark.empty:
        vals = pd.concat([vals, benchmark[_equity_col(benchmark)].astype(float)])
    lo, hi = float(vals.min()), float(vals.max())
    pad = (hi - lo) * 0.08 or hi * 0.02
    fig.update_yaxes(range=[lo - pad, hi + pad], title_text="PKR", tickformat=",.0f")
    return fig


def drawdown_chart(equity: pd.DataFrame, mode: Optional[Mode] = None) -> go.Figure:
    """Fall from the running peak, in percent."""
    p = palette(mode)
    fig = go.Figure()
    if equity.empty:
        return style_figure(fig, height=160, mode=mode)
    vals = equity[_equity_col(equity)].astype(float)
    dd = (vals / vals.cummax() - 1.0) * 100.0
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(equity[_date_col(equity)]),
            y=dd,
            name="Drawdown",
            line={"color": p.down, "width": 1.2},
            fill="tozeroy",
            fillcolor=p.down_soft,
            hovertemplate="%{y:.1f}%<extra></extra>",
        )
    )
    style_figure(fig, height=160, mode=mode)
    fig.update_yaxes(ticksuffix="%")
    return fig


def calibration_chart(
    resolved: pd.DataFrame, n_bins: int = 10, mode: Optional[Mode] = None
) -> go.Figure:
    """Predicted chance of UP against how often the stock actually rose, per bin."""
    p = palette(mode)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfectly honest",
            line={"color": p.muted, "dash": "dot", "width": 1},
            hoverinfo="skip",
        )
    )
    if not resolved.empty:
        pred, actual, counts = compute_calibration_curve(
            resolved["up_probability"].astype(float).to_numpy(),
            resolved["actual_up"].to_numpy(dtype=float),
            n_bins=n_bins,
        )
        mask = counts > 0
        fig.add_trace(
            go.Scatter(
                x=pred[mask],
                y=actual[mask],
                mode="lines+markers",
                name="This model",
                line={"color": p.accent, "width": 2},
                marker={
                    "size": 6 + np.sqrt(counts[mask]) * 1.5,
                    "color": p.accent,
                    "line": {"width": 0},
                },
                customdata=counts[mask],
                hovertemplate="Predicted %{x:.0%} · actual %{y:.0%} · %{customdata} "
                "predictions<extra></extra>",
            )
        )
    style_figure(fig, height=340, mode=mode)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(
        range=[-0.02, 1.02],
        tickformat=".0%",
        title_text="Predicted chance of UP",
        showgrid=True,
        gridcolor=p.border,
        griddash="dot",
    )
    fig.update_yaxes(range=[-0.04, 1.04], tickformat=".0%", title_text="Share that went UP")
    return fig


def sector_bar(sectors: pd.DataFrame, mode: Optional[Mode] = None) -> go.Figure:
    """Average 1-day move per sector."""
    p = palette(mode)
    fig = go.Figure(
        go.Bar(
            x=sectors["chg_1d"],
            y=sectors["sector"],
            orientation="h",
            marker_color=[p.up if v >= 0 else p.down for v in sectors["chg_1d"]],
            hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
        )
    )
    style_figure(fig, height=40 + 30 * max(1, len(sectors)), mode=mode)
    fig.update_layout(hovermode="closest", bargap=0.35)
    fig.update_xaxes(ticksuffix="%", showgrid=True, gridcolor=p.border, griddash="dot")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", tickfont={"family": None})
    return fig
