"""Page 1: Market Overview — Universal snapshot of PSX universe and model signals."""

import pandas as pd
import plotly.express as px
import streamlit as st

from psx_predictor.config.loader import load_config
from psx_predictor.predictions.registry import PredictionRegistry
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import get_storage_paths

st.set_page_config(page_title="Market Overview — PSX Predictor", page_icon="🌐", layout="wide")

st.title("🌐 Market Overview & Universe Signals")
st.markdown(
    "Universal monitoring of tracked Pakistan Stock Exchange (PSX) assets "
    "and machine learning signals."
)

cfg = load_config()
paths = get_storage_paths(cfg.settings.data_dir)
registry = PredictionRegistry(storage_paths=paths)

# Query registered predictions
all_preds = registry.get_predictions()

rows = []
for stock in cfg.stocks:
    sym = stock.symbol
    price_file = paths["processed_prices"] / f"{sym}.parquet"
    has_data = price_file.exists()

    latest_close: float | None = None
    ret_1d: float | None = None
    last_date: str | None = None

    if has_data:
        try:
            df = read_parquet(price_file)
            if not df.empty:
                d_col = "trade_date" if "trade_date" in df.columns else "date"
                last_date = str(df[d_col].iloc[-1])
                latest_close = float(df["close"].iloc[-1])
                if len(df) >= 2:
                    prev_close = float(df["close"].iloc[-2])
                    ret_1d = ((latest_close - prev_close) / prev_close) * 100.0
                else:
                    ret_1d = 0.0
        except Exception:
            has_data = False

    # Lookup latest prediction for symbol
    signal_val = "N/A"
    up_prob_val: float | None = None

    if not all_preds.empty and "symbol" in all_preds.columns:
        sym_preds = all_preds[all_preds["symbol"] == sym]
        if not sym_preds.empty:
            sym_preds = sym_preds.sort_values("target_date", ascending=False)
            latest_rec = sym_preds.iloc[0]
            signal_val = str(latest_rec.get("signal", "HOLD"))
            up_prob_val = float(latest_rec.get("up_probability", 0.5)) * 100.0

    rows.append(
        {
            "Symbol": sym,
            "Company Name": stock.name,
            "Sector": stock.sector,
            "Last Session": last_date or "N/A",
            "Latest Close (PKR)": f"{latest_close:.2f}" if latest_close is not None else "—",
            "1-Day Change": f"{ret_1d:+.2f}%" if ret_1d is not None else "—",
            "Signal": signal_val,
            "UP Prob": f"{up_prob_val:.1f}%" if up_prob_val is not None else "—",
            "Data Status": "Ready" if has_data else "Missing",
        }
    )

overview_df = pd.DataFrame(rows)

# Metric summary row
c1, c2, c3, c4 = st.columns(4)
buy_count = sum(1 for r in rows if r["Signal"] == "BUY")
hold_count = sum(1 for r in rows if r["Signal"] == "HOLD")
sell_count = sum(1 for r in rows if r["Signal"] == "SELL")
ready_count = sum(1 for r in rows if r["Data Status"] == "Ready")

with c1:
    st.metric("Total Universe", f"{len(rows)} Tickers")
with c2:
    st.metric("Bullish Signals (BUY)", f"{buy_count} Stocks")
with c3:
    st.metric("Neutral Signals (HOLD)", f"{hold_count} Stocks")
with c4:
    st.metric("Data Complete", f"{ready_count}/{len(rows)} Stocks")

st.divider()

# Filter controls
col_search, col_sector = st.columns([2, 2])
with col_search:
    search_q = st.text_input("Filter by Symbol or Name", "")
with col_sector:
    sectors = ["All"] + sorted(list({s.sector for s in cfg.stocks}))
    selected_sector = st.selectbox("Filter by Sector", sectors)

filtered_df = overview_df.copy()
if search_q:
    q = search_q.strip().lower()
    filtered_df = filtered_df[
        filtered_df["Symbol"].str.lower().str.contains(q)
        | filtered_df["Company Name"].str.lower().str.contains(q)
    ]
if selected_sector and selected_sector != "All":
    filtered_df = filtered_df[filtered_df["Sector"] == selected_sector]

st.dataframe(filtered_df, use_container_width=True, hide_index=True)

st.divider()

# Visual charts: Sector Breakdown & Signal Distribution
st.subheader("Market Composition & Signal Breakdown")
chart_c1, chart_c2 = st.columns(2)

with chart_c1:
    sector_counts = overview_df["Sector"].value_counts().reset_index()
    sector_counts.columns = ["Sector", "Count"]
    fig_sector = px.pie(
        sector_counts,
        names="Sector",
        values="Count",
        title="Universe Sector Distribution",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Teal,
    )
    fig_sector.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    st.plotly_chart(fig_sector, use_container_width=True)

with chart_c2:
    signal_counts = overview_df["Signal"].value_counts().reset_index()
    signal_counts.columns = ["Signal", "Count"]
    color_map = {"BUY": "#10b981", "SELL": "#ef4444", "HOLD": "#64748b", "N/A": "#334155"}
    fig_signal = px.bar(
        signal_counts,
        x="Signal",
        y="Count",
        color="Signal",
        color_discrete_map=color_map,
        title="Platform Decision Signal Distribution",
    )
    fig_signal.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        showlegend=False,
    )
    st.plotly_chart(fig_signal, use_container_width=True)
