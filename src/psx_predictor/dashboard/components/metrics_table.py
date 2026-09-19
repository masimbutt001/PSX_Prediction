"""Streamlit UI components for rendering backtest performance metrics and summaries."""

from typing import Any

import pandas as pd
import streamlit as st


def render_metrics_summary(metrics: Any) -> None:
    """Render key performance indicator cards for strategy backtesting."""
    col1, col2, col3, col4, col5 = st.columns(5)

    m_data = metrics.to_dict() if hasattr(metrics, "to_dict") else dict(metrics)

    tot_ret = float(m_data.get("total_return", 0.0)) * 100.0
    cagr = float(m_data.get("cagr", 0.0)) * 100.0
    sharpe = float(m_data.get("sharpe_ratio", 0.0))
    max_dd = float(m_data.get("max_drawdown", 0.0)) * 100.0
    win_rate = float(m_data.get("win_rate", 0.0)) * 100.0

    with col1:
        st.metric(
            label="Total Return",
            value=f"{tot_ret:+.2f}%",
            delta=f"{tot_ret:+.2f}%",
        )
    with col2:
        st.metric(
            label="Sharpe Ratio",
            value=f"{sharpe:.2f}",
        )
    with col3:
        st.metric(
            label="CAGR",
            value=f"{cagr:+.2f}%",
        )
    with col4:
        st.metric(
            label="Max Drawdown",
            value=f"{max_dd:.2f}%",
            delta=f"-{abs(max_dd):.2f}%",
            delta_color="inverse",
        )
    with col5:
        st.metric(
            label="Win Rate",
            value=f"{win_rate:.1f}%",
        )


def render_trades_table(trades_df: pd.DataFrame) -> None:
    """Render interactive dataframe of simulated trade executions."""
    if trades_df.empty:
        st.info("No trades were executed during this backtest simulation window.")
        return

    st.subheader("Simulated Trade Execution Log")
    st.dataframe(
        trades_df,
        use_container_width=True,
        hide_index=True,
    )
