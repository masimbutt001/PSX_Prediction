"""Streamlit UI component for rendering directional prediction cards and feature drivers."""

from typing import Any

import plotly.graph_objects as go
import streamlit as st


def render_prediction_card(
    symbol: str,
    up_prob: float,
    signal: str,
    confidence: float,
    target_date: str,
    top_drivers: list[dict[str, Any]],
    model_name: str = "Multi-Modal Ensemble",
) -> None:
    """Render forward-looking prediction summary card with confidence and drivers."""
    sig_upper = signal.strip().upper()

    if sig_upper == "BUY":
        bg_color = "#064e3b"
        border_color = "#10b981"
        badge_text = "🟢 BUY (Bullish)"
    elif sig_upper == "SELL":
        bg_color = "#4c0519"
        border_color = "#f43f5e"
        badge_text = "🔴 SELL (Bearish)"
    else:
        bg_color = "#1e293b"
        border_color = "#64748b"
        badge_text = "🟡 HOLD (Neutral)"

    st.markdown(
        f"""
        <div style="background-color: {bg_color}; border: 1px solid {border_color};
                    border-radius: 10px; padding: 18px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: #f8fafc;">{symbol} Forecast — {target_date}</h3>
                <span style="font-size: 1.15rem; font-weight: bold; color: {border_color};">
                    {badge_text}
                </span>
            </div>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">Model: {model_name}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="UP Probability",
            value=f"{up_prob * 100:.1f}%",
            delta=f"{(up_prob - 0.5) * 100:+.1f}% vs Neutral",
        )
    with col2:
        st.metric(label="Decision Signal", value=signal)
    with col3:
        st.metric(label="Model Confidence", value=f"{confidence * 100:.1f}%")

    clamped_prob = min(max(up_prob, 0.0), 1.0)
    st.progress(clamped_prob, text=f"Calibrated Bullish Probability: {up_prob * 100:.1f}%")

    if top_drivers:
        st.subheader("Top Predictive Feature Drivers")
        names = [str(d.get("name", "")) for d in top_drivers][:8]
        importances = [float(d.get("importance", 0.0)) for d in top_drivers][:8]

        fig = go.Figure(
            go.Bar(
                x=importances[::-1],
                y=names[::-1],
                orientation="h",
                marker_color="#38bdf8",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
            height=260,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#161b22",
            xaxis_title="Relative Feature Weight / Importance",
        )
        st.plotly_chart(fig, use_container_width=True)
