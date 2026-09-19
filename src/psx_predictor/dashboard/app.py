"""PSX Predictor Platform — Streamlit Interactive Research Dashboard Entrypoint."""

import streamlit as st

from psx_predictor import __version__
from psx_predictor.config.loader import load_config
from psx_predictor.predictions.registry import PredictionRegistry
from psx_predictor.storage.paths import get_storage_paths


def main() -> None:
    """Main dashboard home page displaying platform diagnostics and overview."""
    st.set_page_config(
        page_title="PSX Predictor Platform",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📈 Pakistan Stock Exchange (PSX) Prediction Platform")
    st.markdown(
        """
        Welcome to the **PSX Prediction & Quantitative Analytics Suite** — an institutional-grade,
        multi-modal machine learning platform combining technical price action, automated news &
        corporate disclosure sentiment, and retrospective macroeconomic state tracking.
        """
    )

    cfg = load_config()
    paths = get_storage_paths(cfg.settings.data_dir)
    registry = PredictionRegistry(storage_paths=paths)

    total_stocks = len(cfg.stocks)
    enabled_stocks = len(cfg.get_enabled_stocks())

    # Count stocks with price parquet files
    stocks_with_data = sum(
        1 for s in cfg.stocks if (paths["processed_prices"] / f"{s.symbol}.parquet").exists()
    )

    # Prediction registry records
    pred_df = registry.get_predictions()
    total_preds = len(pred_df) if not pred_df.empty else 0

    st.markdown("### 🏛️ Platform Status & Universe Summary")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="Universe Size", value=f"{total_stocks} Stocks")
    with kpi2:
        st.metric(label="Active Tracking", value=f"{enabled_stocks} Stocks")
    with kpi3:
        st.metric(label="Data Ready", value=f"{stocks_with_data}/{total_stocks} Parquets")
    with kpi4:
        st.metric(label="Logged Predictions", value=f"{total_preds} Records")

    st.divider()

    st.markdown("### 🧭 Interactive Dashboard Modules")
    nav1, nav2, nav3 = st.columns(3)

    with nav1:
        st.markdown(
            """
            #### 🌐 1. Market Overview
            - Universal snapshot across all tracked PSX stocks.
            - Real-time/historical closing prices and 1-day percentage change.
            - Machine learning directional signals (**BUY**, **HOLD**, **SELL**) and confidence.
            - Sector breakdown and signal distribution.
            """
        )

    with nav2:
        st.markdown(
            """
            #### 📈 2. Stock Analysis
            - Interactive multi-panel Plotly candlestick charts with volume bars.
            - Dynamic overlays: SMA 20, SMA 50, and Bollinger Bands.
            - Forward-looking prediction card with calibrated probability.
            - Explainable AI feature driver importance breakdown.
            """
        )

    with nav3:
        st.markdown(
            """
            #### 🧪 3. Model Performance
            - Interactive strategy backtesting engine with realistic PSX transaction costs.
            - Strategy evaluation: Moving Average Crossover, Momentum, Buy & Hold.
            - Comparative equity curves and underwater drawdown visualizers.
            - Quantitative metrics: Sharpe ratio, CAGR, max drawdown, win rate.
            """
        )

    st.divider()

    footer_text = f"PSX Predictor Platform v{__version__} | FastAPI, Streamlit, and XGBoost"
    st.markdown(
        f"""
        <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 10px;">
            {footer_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
