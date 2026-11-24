"""Stock detail: switcher, interactive price chart, prediction card, history, news, research."""

import pandas as pd
import streamlit as st

from psx_predictor.dashboard import data, state
from psx_predictor.dashboard.components.figures import (
    drivers_chart,
    price_chart,
    probability_gauge,
)
from psx_predictor.dashboard.components.jobs_panel import run_button
from psx_predictor.dashboard.components.ui import (
    disclaimer,
    fmt_pct,
    sentiment_tag,
    signal_chip,
    tone_of,
)
from psx_predictor.dashboard.theme import palette, text

RANGES = {"1M": 22, "3M": 66, "6M": 128, "1Y": 252, "5Y": 1260, "All": None}

cfg = state.get_config()
p = palette()
symbol = state.selected_symbol()

picked = st.pills(
    "Stock",
    state.symbols(),
    default=symbol,
    selection_mode="single",
    label_visibility="collapsed",
)
if picked and picked != symbol:
    state.select_symbol(picked)
    st.rerun()

stock = cfg.get_stock(symbol)
df = state.prices(symbol)
preds = state.predictions()
latest = data.latest_prediction_per_symbol(preds).get(symbol)

# ---- Title row ----
t1, t2 = st.columns([3, 2], vertical_alignment="bottom")
with t1:
    name = stock.name if stock else ""
    sector = stock.sector if stock else ""
    if df.empty:
        price_html = '<span class="psx-muted">No price data</span>'
    else:
        close = float(df["close"].iloc[-1])
        chg = (close / float(df["close"].iloc[-2]) - 1) * 100 if len(df) > 1 else None
        price_html = (
            f'<span class="psx-value" style="font-size:34px">{close:,.2f}</span> '
            f'<span class="psx-note psx-{tone_of(chg)}" style="font-size:16px">'
            f"{fmt_pct(chg)}</span> "
            f'<span class="psx-muted" style="font-size:12px">PKR · close '
            f"{pd.Timestamp(df['trade_date'].iloc[-1]):%a %d %b %Y}</span>"
        )
    st.html(
        f'<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">'
        f'<span style="font-size:28px;font-weight:700">{text(symbol)}</span>'
        f'<span class="psx-muted">{text(name)}</span>'
        f'<span class="psx-pill" style="height:24px">{text(sector)}</span></div>'
        f'<div style="margin-top:4px">{price_html}</div>'
    )
with t2:
    b1, b3, b2 = st.columns(3)
    with b3:
        run_button(
            "Update prices",
            "update_prices",
            symbols=[symbol],
            key="sd_update",
            icon=":material/download_for_offline:",
            help=f"Fetch the newest sessions for {symbol} in the background",
            width="stretch",
        )
    if b1.button(
        "Refresh prediction",
        icon=":material/bolt:",
        type="primary",
        width="stretch",
        help="Run the live predictor for this stock and log the result",
    ):
        with st.spinner(f"Predicting the next session for {symbol}…"):
            try:
                from psx_predictor.predictions.predictor import LivePredictor

                LivePredictor(storage_paths=state.get_paths()).generate_prediction(
                    symbol, log_to_registry=True
                )
                state.refresh()
                st.toast(f"New prediction logged for {symbol}", icon=":material/check:")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not generate a prediction: {exc}")
    if not df.empty:
        b2.download_button(
            "Prices CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"{symbol}_prices.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )

if df.empty:
    with st.container(border=True):
        st.warning(f"No price data for {symbol} yet.")
        run_button(
            f"Update {symbol} now",
            "update_prices",
            symbols=[symbol],
            key="sd_update_missing",
            primary=True,
            icon=":material/download:",
            help=f"Runs `psx data update --symbols {symbol}` in the background",
        )
        st.caption(
            f"Same as running `uv run psx data update --symbols {symbol}` in a terminal. "
            "Progress shows at the top of the page."
        )
    st.stop()

left, right = st.columns([2.3, 1], gap="large")

# ---- Chart ----
with left:
    with st.container(border=True):
        c1, c2 = st.columns([1, 1.4], vertical_alignment="center")
        rng = (
            c1.segmented_control(
                "Range", list(RANGES), default="6M", label_visibility="collapsed", key="sd_range"
            )
            or "6M"
        )
        inds = (
            c2.pills(
                "Indicators",
                ["SMA 20", "SMA 50", "Bollinger", "Volume"],
                default=["SMA 20", "SMA 50", "Volume"],
                selection_mode="multi",
                label_visibility="collapsed",
                key="sd_inds",
            )
            or []
        )
        n = RANGES[rng]
        view = df if n is None else df.tail(n + 50)  # extra rows so SMA 50 starts on screen
        forecast = None
        if latest and latest.get("target_date"):
            forecast = (str(latest["target_date"]), float(latest["up_probability"]))
        fig = price_chart(
            view,
            symbol,
            sma20="SMA 20" in inds,
            sma50="SMA 50" in inds,
            bollinger="Bollinger" in inds,
            volume="Volume" in inds,
            forecast=forecast,
        )
        if n is not None and len(df) > n:
            start = pd.Timestamp(df["trade_date"].iloc[-n])
            end = pd.Timestamp(forecast[0] if forecast else df["trade_date"].iloc[-1])
            fig.update_xaxes(range=[start, end + pd.Timedelta(days=3)])
            vis = df.tail(n)
            lo, hi = float(vis["low"].min()), float(vis["high"].max())
            fig.update_yaxes(range=[lo * 0.97, hi * 1.03], row=1, col=1)
        st.plotly_chart(
            fig,
            width="stretch",
            config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        )
        st.caption(
            "Drag to zoom, double-click to reset. The marker after the last candle is "
            "the next-session forecast."
        )

    # stats strip
    closes = df["close"].astype(float)
    tail = df.tail(252)
    vol20 = closes.pct_change().tail(20).std() * 100
    avg_vol = float(df["volume"].tail(20).mean()) if "volume" in df.columns else None
    stats = [
        ("52-week high", f"{float(tail['high'].max()):,.2f}"),
        ("52-week low", f"{float(tail['low'].min()):,.2f}"),
        ("Daily volatility, 20d", f"{vol20:.2f}%"),
        ("Avg volume, 20d", f"{avg_vol / 1e6:,.2f}M" if avg_vol else "—"),
        ("1-year change", fmt_pct((closes.iloc[-1] / float(tail["close"].iloc[0]) - 1) * 100)),
        ("Sessions on file", f"{len(df):,}"),
    ]
    cols = st.columns(len(stats))
    for col, (lab, val) in zip(cols, stats, strict=True):
        col.html(
            f'<div class="psx-label">{text(lab)}</div>'
            f'<div class="psx-value" style="font-size:16px">{text(val)}</div>'
        )

# ---- Prediction card ----
with right:
    with st.container(border=True):
        if not latest:
            st.markdown("##### Next session")
            st.info("No prediction logged for this stock yet. Use **Refresh prediction**.")
        else:
            up = float(latest["up_probability"])
            target = pd.Timestamp(str(latest["target_date"]))
            st.html(
                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                f'<span style="font-weight:600;font-size:16px">Next session · {target:%a %d %b}'
                f"</span>{signal_chip(latest.get('signal'))}</div>"
            )
            g, info = st.columns([1, 1.1], vertical_alignment="center")
            g.plotly_chart(
                probability_gauge(up),
                width="stretch",
                config={"displayModeBar": False, "staticPlot": True},
            )
            conf = latest.get("confidence")
            rec = data.track_record(preds, last_n=60)
            rec_row = rec[rec["symbol"] == symbol]
            track = (
                f"{float(rec_row['hit_rate'].iloc[0]):.0%} of {int(rec_row['predictions'].iloc[0])}"
                if not rec_row.empty
                else "—"
            )
            research = data.parse_json(latest.get("research_json"))
            conf_txt = f"{float(conf):.0%}" if conf is not None and not pd.isna(conf) else "—"
            facts = [
                ("Confidence", conf_txt),
                ("Model", str(latest.get("model_name", "—"))),
                ("Research", str(research.get("stance", "—")).title()),
                ("Hit rate, 60 sessions", track),
            ]
            info.html(
                '<div style="display:flex;flex-direction:column;gap:8px;font-size:13px">'
                + "".join(
                    f'<div><span class="psx-muted">{text(k)}</span> <b>{text(v)}</b></div>'
                    for k, v in facts
                )
                + "</div>"
            )
            drivers = {
                k: float(v)
                for k, v in data.parse_json(latest.get("drivers_json")).items()
                if isinstance(v, (int, float))
            }
            if drivers:
                st.markdown("**Why the model thinks this**")
                st.plotly_chart(
                    drivers_chart(drivers), width="stretch", config={"displayModeBar": False}
                )
                st.caption("Bars show each input's weight in this prediction.")
            if research.get("agrees_with_model") is False:
                st.warning("The latest research disagrees with the model's direction.")

# ---- Tabs ----
tab_hist, tab_news, tab_research = st.tabs(
    [
        ":material/history: Prediction history",
        ":material/newspaper: News",
        ":material/menu_book: Research",
    ]
)
with tab_hist:
    hist = data.prediction_history(preds, symbol, limit=40)
    if hist.empty:
        st.caption("No predictions logged for this stock yet.")
    else:
        hist["Chance of UP"] = hist["Chance of UP"] * 100

        def _res_color(v: object) -> str:
            return {
                "Correct": f"color: {p.up}; font-weight: 600",
                "Missed": f"color: {p.down}; font-weight: 600",
            }.get(str(v), f"color: {p.muted}")

        st.dataframe(
            hist.style.map(_res_color, subset=["Result"]),
            hide_index=True,
            width="stretch",
            column_config={
                "Chance of UP": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=100.0, format="%.0f%%"
                ),
                "Actual move": st.column_config.NumberColumn(format="%+.2f%%"),
            },
        )
with tab_news:
    items = state.news(symbol=symbol, limit=15)
    if items.empty:
        st.caption(f"No news matched to {symbol} yet.")
    else:
        rows = []
        for _, n in items.iterrows():
            score = n.get("sentiment_score")
            score_txt = f" · sentiment {float(score):+.2f}" if score is not None else ""
            rows.append(
                f'<div class="psx-row" style="align-items:flex-start">'
                f"{sentiment_tag(n.get('sentiment_label'))}<div><div>"
                f"{text(n.get('headline', ''))}</div><div class='psx-muted' "
                f"style='font-size:12px'>{text(n.get('source', ''))} · "
                f"{text(data.format_pkt(n.get('published_at'), '%d %b %Y'))}"
                f"{text(score_txt)}</div></div></div>"
            )
        st.html("".join(rows))
with tab_research:
    snap_r = data.parse_json(latest.get("research_json")) if latest else {}
    if not snap_r:
        st.caption(
            f"No research applied to the latest {symbol} prediction. "
            "Add a research file from the Research page."
        )
    else:
        keys = [
            ("stance", "Stance"),
            ("outlook_short_term", "Outlook 1-4 weeks"),
            ("risk_score", "Risk"),
            ("weight", "Weight today"),
            ("research_date", "Research date"),
        ]
        cols = st.columns(len(keys))
        for col, (k, lab) in zip(cols, keys, strict=True):
            col.metric(lab, str(snap_r.get(k, "—")))
    st.page_link("views/research.py", label="Open the Research page", icon=":material/menu_book:")

disclaimer()
