"""Page 2: Stock Analysis — In-depth technical charts and explainable ML forecasts."""

import json

import streamlit as st

from psx_predictor.config.loader import load_config
from psx_predictor.dashboard.components.charts import create_candlestick_chart
from psx_predictor.dashboard.components.prediction_card import render_prediction_card
from psx_predictor.predictions.predictor import LivePredictor
from psx_predictor.predictions.registry import PredictionRegistry
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import get_storage_paths

st.set_page_config(page_title="Stock Analysis — PSX Predictor", page_icon="📈", layout="wide")

st.title("📈 Individual Stock Analysis & Predictive Profile")
st.markdown(
    "Detailed candlestick technical analysis, volume dynamics, and machine learning predictions."
)

cfg = load_config()
paths = get_storage_paths(cfg.settings.data_dir)
symbols = [s.symbol for s in cfg.stocks]

# Stock Selector
selected_symbol = st.selectbox("Select PSX Stock", symbols, index=0)
stock_cfg = cfg.get_stock(selected_symbol)

if stock_cfg:
    st.markdown(
        f"**Company:** {stock_cfg.name} | **Sector:** {stock_cfg.sector} | "
        f"**Ticker:** `{stock_cfg.symbol}` (Yahoo: `{stock_cfg.yahoo_ticker}`)"
    )

price_file = paths["processed_prices"] / f"{selected_symbol}.parquet"

if not price_file.exists():
    st.warning(
        f"Processed price data for '{selected_symbol}' was not found. "
        "Run `psx update --symbol " + selected_symbol + "` to fetch daily data."
    )
else:
    df = read_parquet(price_file)

    # Technical Chart Controls
    st.subheader(f"{selected_symbol} Technical Candlestick Analysis")
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    with ctrl1:
        max_win = min(len(df), 500)
        def_win = min(len(df), 120)
        days_window = st.slider(
            "Lookback Window (Sessions)",
            min_value=20,
            max_value=max_win,
            value=def_win,
        )
    with ctrl2:
        show_ma = st.checkbox("Show Moving Averages (20 & 50)", value=True)
    with ctrl3:
        show_bb = st.checkbox("Show Bollinger Bands", value=True)
    with ctrl4:
        show_vol = st.checkbox("Show Volume Subplot", value=True)

    slice_df = df.tail(days_window).copy()
    fig_candlestick = create_candlestick_chart(
        df=slice_df,
        symbol=selected_symbol,
        show_volume=show_vol,
        show_ma=show_ma,
        show_bb=show_bb,
    )
    st.plotly_chart(fig_candlestick, use_container_width=True)

    st.divider()

    # Machine Learning Forecast Section
    st.subheader(f"{selected_symbol} Machine Learning Directional Forecast")

    registry = PredictionRegistry(storage_paths=paths)
    preds = registry.get_predictions(symbol=selected_symbol)

    if not preds.empty:
        preds = preds.sort_values("target_date", ascending=False)
        latest = preds.iloc[0]

        # Parse feature drivers
        drivers = []
        raw_drivers = latest.get("drivers_json", "{}")
        if isinstance(raw_drivers, str):
            try:
                d_dict = json.loads(raw_drivers)
                drivers = [{"name": k, "importance": v} for k, v in d_dict.items()]
            except Exception:
                drivers = []

        render_prediction_card(
            symbol=selected_symbol,
            up_prob=float(latest["up_probability"]),
            signal=str(latest.get("signal", "HOLD")),
            confidence=float(latest.get("confidence", 0.5)),
            target_date=str(latest["target_date"]),
            top_drivers=drivers,
            model_name=str(latest.get("model_name", "Multi-Modal Stacking Ensemble")),
        )
    else:
        st.info(f"No prediction record logged yet in the audit registry for {selected_symbol}.")
        if st.button("Generate On-Demand Prediction", type="primary"):
            with st.spinner(f"Computing forward prediction for {selected_symbol}..."):
                try:
                    predictor = LivePredictor(storage_paths=paths)
                    rec = predictor.generate_prediction(selected_symbol, log_to_registry=True)
                    parsed_drivers: dict[str, float] = {}
                    if rec.drivers_json:
                        try:
                            parsed_drivers = json.loads(rec.drivers_json)
                        except Exception:
                            parsed_drivers = {}
                    drv_list = [
                        {"name": k, "importance": float(v)} for k, v in parsed_drivers.items()
                    ]
                    render_prediction_card(
                        symbol=selected_symbol,
                        up_prob=rec.up_probability,
                        signal=rec.signal,
                        confidence=rec.confidence,
                        target_date=rec.target_date,
                        top_drivers=drv_list,
                        model_name=rec.model_name,
                    )
                except Exception as exc:
                    st.error(f"Failed to generate prediction: {exc}")
