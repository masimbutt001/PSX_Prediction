"""Overview: KPIs, the filterable watchlist with signals, sector moves, last run and news."""

import pandas as pd
import streamlit as st

from psx_predictor.dashboard import data, state
from psx_predictor.dashboard.components.figures import sector_bar
from psx_predictor.dashboard.components.help import section_title
from psx_predictor.dashboard.components.jobs_panel import run_button
from psx_predictor.dashboard.components.ui import (
    disclaimer,
    fmt_pct,
    kpi_row,
    page_header,
    sentiment_tag,
    step_row,
    tone_of,
)
from psx_predictor.dashboard.theme import palette, text

STOCK_PAGE = "views/stock.py"

snap = state.snapshot()
preds = state.predictions()
macro = state.macro()
p = palette()

targets = snap["target_date"].dropna()
next_session = str(targets.max()) if not targets.empty else None
subtitle = f"{len(snap)} stocks tracked"
if next_session:
    subtitle = f"Signals for the session on {pd.Timestamp(next_session):%a %d %b %Y} · " + subtitle
page_header("Market overview", subtitle)

# ---- KPI row ----
cards = []
if not macro.empty and "kse100_index" in macro.columns:
    last = macro.iloc[-1]
    kse_ret = float(last.get("kse100_return_1d", 0.0) or 0.0) * 100.0
    cards.append(
        (
            "KSE-100",
            f"{float(last['kse100_index']):,.0f}",
            fmt_pct(kse_ret) + " today",
            tone_of(kse_ret),
        )
    )
else:
    cards.append(("KSE-100", "—", "No macro data yet", "muted"))

sig_counts = snap["signal"].value_counts()
cards.append(
    (
        "Signals",
        f"{sig_counts.get('BUY', 0)} · {sig_counts.get('HOLD', 0)} · {sig_counts.get('SELL', 0)}",
        "BUY · HOLD · SELL",
        "accent",
    )
)

s30 = data.overall_scores(preds, days=30)
s_all = data.overall_scores(preds)
if s30["hit_rate"] is not None:
    cards.append(
        (
            "Hit rate, 30 days",
            f"{s30['hit_rate']:.1%}",
            f"{s30['n']} scored predictions",
            "up" if s30["hit_rate"] >= 0.5 else "down",
        )
    )
    cards.append(
        (
            "Brier score, 30 days",
            f"{s30['brier']:.3f}",
            "Coin flip scores 0.250",
            "up" if s30["brier"] < 0.25 else "down",
        )
    )
elif s_all["hit_rate"] is not None:
    cards.append(
        (
            "Hit rate, all time",
            f"{s_all['hit_rate']:.1%}",
            f"{s_all['n']} scored predictions",
            "muted",
        )
    )
    cards.append(("Brier score", f"{s_all['brier']:.3f}", "Coin flip scores 0.250", "muted"))
else:
    cards.append(("Hit rate", "—", "No scored predictions yet", "muted"))
    cards.append(("Brier score", "—", "No scored predictions yet", "muted"))

if not macro.empty and "usd_pkr" in macro.columns:
    last = macro.iloc[-1]
    fx_ret = float(last.get("usd_pkr_return_1d", 0.0) or 0.0) * 100.0
    cards.append(
        ("USD / PKR", f"{float(last['usd_pkr']):.2f}", fmt_pct(fx_ret) + " today", tone_of(-fx_ret))
    )
helps = ["ov_kse100", "ov_signals", "ov_hit_rate", "ov_brier", "ov_usdpkr"][: len(cards)]
kpi_row(cards, helps)

st.write("")
left, right = st.columns([2.3, 1], gap="large")

# ---- Watchlist ----
with left:
    with st.container(border=True):
        h1, h2 = st.columns([1, 1], vertical_alignment="center")
        with h1:
            section_title("Watchlist and signals", "ov_watchlist")
        sig_filter = h2.segmented_control(
            "Signal",
            ["All", "BUY", "HOLD", "SELL"],
            default="All",
            label_visibility="collapsed",
            key="ov_sig",
        )
        sectors = sorted(snap["sector"].dropna().unique().tolist())
        sector_filter = st.pills(
            "Sector",
            sectors,
            selection_mode="multi",
            label_visibility="collapsed",
            key="ov_sector",
        )

        view = snap.copy()
        if sig_filter and sig_filter != "All":
            view = view[view["signal"] == sig_filter]
        if sector_filter:
            view = view[view["sector"].isin(sector_filter)]

        table = pd.DataFrame(
            {
                "Stock": view["symbol"],
                "Company": view["name"],
                "Close": view["close"],
                "1D %": view["chg_1d"],
                "5D %": view["chg_5d"],
                "30 sessions": view["trend"],
                "Signal": view["signal"].fillna("—"),
                "Chance of UP": view["up_prob"] * 100,
                "Research": view["research"].fillna("—"),
            }
        ).reset_index(drop=True)

        def _move_color(v: float | None) -> str:
            if v is None or pd.isna(v):
                return ""
            if v > 0:
                return f"color: {p.up}"
            return f"color: {p.down}" if v < 0 else ""

        def _sig_color(v: object) -> str:
            return {
                "BUY": f"color: {p.up}; font-weight: 700",
                "SELL": f"color: {p.down}; font-weight: 700",
            }.get(str(v), f"color: {p.muted}")

        styled = table.style.map(_move_color, subset=["1D %", "5D %"]).map(
            _sig_color, subset=["Signal"]
        )
        event = st.dataframe(
            styled,
            hide_index=True,
            width="stretch",
            height=min(36 * (max(len(table), 1) + 1) + 3, 540),
            row_height=36,
            on_select="rerun",
            selection_mode="single-row",
            key="ov_table",
            column_config={
                "Stock": st.column_config.TextColumn(width="small", pinned=True),
                "Company": st.column_config.TextColumn(width="medium"),
                "Close": st.column_config.NumberColumn("Close (PKR)", format="%.2f"),
                "1D %": st.column_config.NumberColumn(format="%+.2f%%"),
                "5D %": st.column_config.NumberColumn(format="%+.2f%%"),
                "30 sessions": st.column_config.LineChartColumn(width="small"),
                "Chance of UP": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=100.0, format="%.0f%%", width="small"
                ),
            },
        )
        sel = event.selection.rows if event and hasattr(event, "selection") else []
        if sel:
            state.select_symbol(str(table.iloc[sel[0]]["Stock"]))
            st.switch_page(STOCK_PAGE)

        f1, f2 = st.columns([3, 1], vertical_alignment="center")
        f1.caption("Click a row to open that stock. Filters combine.")
        export = snap.drop(columns=["trend"]).rename(columns={"up_prob": "up_probability"})
        f2.download_button(
            "Export CSV",
            export.to_csv(index=False).encode("utf-8"),
            file_name=f"psx_signals_{next_session or 'latest'}.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )
        if table.empty:
            st.info("No stocks match these filters.")

# ---- Right column ----
with right:
    sect = data.sector_moves(snap)
    with st.container(border=True):
        section_title("Sector moves, last session", "ov_sectors")
        if sect.empty:
            st.caption("No price data yet.")
        else:
            st.plotly_chart(sector_bar(sect), width="stretch", config={"displayModeBar": False})

    runs = state.pipeline_runs(limit=10)
    run, run_tasks = data.latest_steps(runs)
    with st.container(border=True):
        section_title("Last daily run", "ov_last_run")
        if run is None:
            st.caption("No daily run logged yet. Run `uv run psx schedule run-daily`.")
        else:
            st.caption(
                f"{data.format_pkt(run['started_at'])} · "
                f"{float(run.get('duration_seconds', 0)):.0f}s · "
                + ("finished" if run.get("success") else "had failures")
            )
            rows = "".join(
                step_row(
                    data.STEP_LABELS.get(t.get("step", ""), t.get("step", "")),
                    t.get("status", ""),
                    "OK" if t.get("status") == "SUCCESS" else t.get("status", "").title(),
                )
                for t in run_tasks
            )
            st.html(rows)
        l1, l2 = st.columns([1.2, 1], vertical_alignment="center")
        with l1:
            st.page_link("views/pipeline.py", label="Open pipeline details", icon=":material/sync:")
        with l2:
            run_button("Run now", "run_pipeline", key="ov_run_pipeline", width="stretch")

    headlines = state.news(limit=5)
    with st.container(border=True):
        section_title("Latest news", "ov_news")
        if headlines.empty:
            st.caption("No processed news yet.")
        else:
            items = []
            for _, n in headlines.iterrows():
                sym = n.get("primary_symbol")
                sym = sym if isinstance(sym, str) and sym else "Market"
                try:
                    when = f"{pd.Timestamp(n['published_at']):%d %b}"
                except (ValueError, TypeError):
                    when = ""
                items.append(
                    f'<div class="psx-row" style="align-items:flex-start">'
                    f"{sentiment_tag(n.get('sentiment_label'))}"
                    f"<div><div>{text(n.get('headline', ''))}</div>"
                    f'<div class="psx-muted" style="font-size:12px">'
                    f"{text(n.get('source', ''))} · {text(sym)} · {text(when)}</div></div></div>"
                )
            st.html("".join(items))

disclaimer()
