"""Unit tests for Phase 18: Streamlit Research Dashboard and Plotly Visualizations."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from typer.testing import CliRunner

from psx_predictor.cli.main import app as cli_app
from psx_predictor.dashboard.app import main as dashboard_main
from psx_predictor.dashboard.components.charts import (
    create_candlestick_chart,
    create_drawdown_chart,
    create_equity_curve_chart,
)
from psx_predictor.dashboard.components.metrics_table import (
    render_metrics_summary,
    render_trades_table,
)
from psx_predictor.dashboard.components.prediction_card import render_prediction_card


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate 60 sessions of synthetic daily OHLCV price series."""
    dates = pd.date_range(start="2024-01-01", periods=60, freq="B")
    np.random.seed(42)
    closes = 100.0 + np.cumsum(np.random.randn(60) * 1.5)

    return pd.DataFrame(
        {
            "trade_date": [d.strftime("%Y-%m-%d") for d in dates],
            "open": closes - 0.5,
            "high": closes + 1.5,
            "low": closes - 1.5,
            "close": closes,
            "volume": np.random.randint(500_000, 2_000_000, 60),
            "adjusted_close": closes,
        }
    )


def test_dashboard_imports_cleanly() -> None:
    """Verify all dashboard components and entry points import cleanly without side-effects."""
    assert callable(dashboard_main)
    assert callable(create_candlestick_chart)
    assert callable(create_equity_curve_chart)
    assert callable(create_drawdown_chart)
    assert callable(render_prediction_card)
    assert callable(render_metrics_summary)
    assert callable(render_trades_table)


def test_candlestick_chart_generation(sample_ohlcv_data: pd.DataFrame) -> None:
    """Test multi-panel candlestick figure generation with moving averages and volume."""
    fig = create_candlestick_chart(
        df=sample_ohlcv_data,
        symbol="OGDC",
        show_volume=True,
        show_ma=True,
        show_bb=True,
    )
    assert isinstance(fig, go.Figure)

    trace_names = [t.name for t in fig.data]
    assert "OGDC Price" in trace_names
    assert "SMA 20" in trace_names
    assert "SMA 50" in trace_names
    assert "BB Upper" in trace_names
    assert "BB Lower" in trace_names
    assert "Volume" in trace_names


def test_candlestick_chart_empty_dataframe() -> None:
    """Test candlestick figure gracefully handles empty inputs without crashing."""
    fig = create_candlestick_chart(
        df=pd.DataFrame(),
        symbol="EMPTY",
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_equity_curve_chart_generation(sample_ohlcv_data: pd.DataFrame) -> None:
    """Test equity curve comparison chart generation in PKR."""
    dates = sample_ohlcv_data["trade_date"]
    values = 1_000_000.0 * np.cumprod(1.0 + np.random.randn(len(dates)) * 0.01)

    eq_df = pd.DataFrame({"date": dates, "portfolio_value": values})
    bench_df = pd.DataFrame({"date": dates, "portfolio_value": values * 0.98})

    fig = create_equity_curve_chart(
        equity_df=eq_df,
        symbol="OGDC",
        benchmark_df=bench_df,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    assert fig.data[0].name == "OGDC Strategy"
    assert fig.data[1].name == "Buy & Hold Benchmark"


def test_drawdown_chart_generation(sample_ohlcv_data: pd.DataFrame) -> None:
    """Test underwater drawdown area figure generation."""
    dates = sample_ohlcv_data["trade_date"]
    drawdown = pd.Series(np.linspace(0.0, -0.15, len(dates)))

    fig = create_drawdown_chart(drawdown, dates)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].name == "Drawdown"


def test_cli_dashboard_dry_run() -> None:
    """Test CLI command `psx dashboard --dry-run` validates entry point and exits 0."""
    runner = CliRunner()
    result = runner.invoke(
        cli_app, ["dashboard", "--host", "127.0.0.1", "--port", "8501", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "Initializing PSX Predictor Streamlit Dashboard" in result.output
    assert "Dashboard entrypoint verified" in result.output
