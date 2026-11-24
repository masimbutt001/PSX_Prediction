"""Research: stock research files from data/research/ and how they feed predictions."""

import json

import pandas as pd
import streamlit as st

from psx_predictor.dashboard import state
from psx_predictor.dashboard.components.research_view import (
    create_score_history_chart,
    files_table,
    overview_table,
    score_history,
)
from psx_predictor.dashboard.components.ui import disclaimer, page_header
from psx_predictor.dashboard.theme import style_figure
from psx_predictor.predictions.registry import PredictionRegistry
from psx_predictor.research.features import research_context_for_date
from psx_predictor.research.store import ResearchStore
from psx_predictor.research.videos import find_research_video, list_research_videos

page_header(
    "Stock research",
    "Research files in data/research/, written with the Antigravity research prompt in "
    "docs/research/.",
)

cfg = state.get_config()
paths = state.get_paths()
store = ResearchStore(research_dir=paths["root"] / "research")
all_files = store.scan()
all_videos = list_research_videos(store.research_dir)
video_keys = {v.key for v in all_videos}

if not all_files:
    st.info(
        f"No research files yet in `{store.research_dir}`. Run the research prompt in "
        "Antigravity, then refresh this page."
    )
    st.stop()

# ---- Overview across all stocks ----
st.subheader("Latest research across stocks")
st.caption(
    "Ranked by short-term outlook × recency weight × confidence. "
    "Research older than 30 days is marked stale."
)
overview = overview_table(all_files)
if overview.empty:
    st.warning("No valid research files. See the validation details below.")
else:
    st.dataframe(overview, width="stretch", hide_index=True)

invalid = [f for f in all_files if not f.is_valid]
if invalid:
    with st.expander(f"⚠️ {len(invalid)} invalid file(s)"):
        for rf in invalid:
            st.markdown(f"**{rf.path.name}**")
            for err in rf.errors:
                st.markdown(f"- {err}")

st.divider()

# ---- One stock ----
researched = sorted({f.symbol for f in all_files})
config_symbols = [s.symbol for s in cfg.stocks]
options = researched + [s for s in config_symbols if s not in researched]
current = state.selected_symbol()
symbol = st.selectbox("Stock", options, index=options.index(current) if current in options else 0)
state.select_symbol(symbol)

sym_files = [f for f in all_files if f.symbol == symbol]
if not sym_files:
    st.info(f"No research for {symbol} yet.")
    st.stop()

st.markdown(f"**{len(sym_files)} research file(s) for {symbol}**")
st.dataframe(files_table(sym_files, video_keys), width="stretch", hide_index=True)

history = score_history(sym_files)
if len(history) >= 2:
    st.plotly_chart(
        style_figure(create_score_history_chart(history, symbol), height=360), width="stretch"
    )

date_labels = [str(f.research_date or f.path.stem) for f in sym_files]
chosen_label = st.selectbox("Research date", date_labels, index=0)
rf = sym_files[date_labels.index(chosen_label)]

if not rf.is_valid or rf.report is None:
    st.error(f"{rf.path.name} is invalid and is not used for predictions.")
    for err in rf.errors:
        st.markdown(f"- {err}")
    md = rf.read_markdown()
    if md:
        st.markdown(md)
    st.stop()

rep = rf.report
assert rep is not None
s = rep.scores
rec = rep.recommendation

st.subheader(f"{rep.symbol} · {rep.company_name or ''}")
st.caption(
    f"Research date {rep.research_date} · data as of {rep.data_as_of or '?'} · "
    f"by {rep.generated_by or '?'}"
)

m = st.columns(6)
m[0].metric("Stance", rec.stance or "-")
m[1].metric("Fair value (PKR)", f"{rec.fair_value_pkr:,.2f}" if rec.fair_value_pkr else "-")
m[2].metric("Upside", f"{rec.upside_pct:.1f}%" if rec.upside_pct is not None else "-")
m[3].metric("Sentiment", f"{s.overall_sentiment:+.2f}" if s.overall_sentiment is not None else "-")
m[4].metric("Risk", f"{s.risk_score:.0f}/100" if s.risk_score is not None else "-")
m[5].metric("Confidence", f"{s.confidence:.0%}" if s.confidence is not None else "-")

video = find_research_video(store.research_dir, rep.symbol, rep.research_date)
if video is not None:
    st.markdown("**Research video**")
    details = [
        x
        for x in (
            f"{video.duration_sec / 60:.1f} min" if video.duration_sec else None,
            video.style,
            video.language,
            video.video_path.name,
        )
        if x
    ]
    st.caption(" · ".join(details))
    try:
        st.video(
            str(video.video_path),
            subtitles=str(video.subtitles_path) if video.subtitles_path else None,
        )
    except Exception as exc:  # a broken or locked file should not break the page
        st.warning(f"Could not play {video.video_path.name}: {exc}")
else:
    st.caption(
        "No video for this research yet. Run docs/research/antigravity_video_prompt.md in "
        "Antigravity to make one."
    )

if rec.rationale:
    st.markdown(f"> {rec.rationale}")
if rep.key_takeaways:
    st.markdown("\n".join(f"- {t}" for t in rep.key_takeaways))

# How this research reaches predictions
ctx = research_context_for_date([rep], pd.Timestamp.today().date())
if ctx:
    st.info(
        f"Weight in today's predictions: **{ctx['weight']:.2f}** "
        f"(age {ctx['age_days']} days; influence halves every 30 days and stops after 120). "
        "Predictions use the newest research dated on or before each session."
    )

registry = PredictionRegistry(storage_paths=paths)
preds = registry.get_predictions(symbol=symbol)
if not preds.empty and "research_json" in preds.columns:
    latest_pred = preds.sort_values("target_date", ascending=False).iloc[0]
    try:
        snap = json.loads(latest_pred.get("research_json") or "{}")
    except (TypeError, ValueError):
        snap = {}
    if snap:
        agree = snap.get("agrees_with_model")
        if agree is True:
            verdict = "agrees with"
        elif agree is False:
            verdict = "disagrees with"
        else:
            verdict = "is neutral to"
        st.markdown(
            f"Latest prediction for **{latest_pred['target_date']}** is "
            f"**{latest_pred['signal']}** ({float(latest_pred['up_probability']):.0%} up). "
            f"Research from {snap.get('research_date')} {verdict} it."
        )

tabs = st.tabs(
    ["Report", "Scores", "Catalysts & risks", "News", "Financials", "Events", "Sources", "Raw JSON"]
)

with tabs[0]:
    md = rf.read_markdown()
    st.markdown(md if md else "_No matching .md report found._")

with tabs[1]:
    st.dataframe(
        pd.DataFrame([{"Score": k, "Value": v} for k, v in s.model_dump().items()]).astype(
            {"Value": "float64"}
        ),
        width="stretch",
        hide_index=True,
    )

with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Catalysts**")
        if rep.catalysts:
            st.dataframe(
                pd.DataFrame([c.model_dump() for c in rep.catalysts]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("None listed.")
    with c2:
        st.markdown("**Risks**")
        if rep.risks:
            st.dataframe(
                pd.DataFrame([r.model_dump() for r in rep.risks]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("None listed.")

with tabs[3]:
    if rep.news:
        news_df = pd.DataFrame([n.model_dump() for n in rep.news]).sort_values(
            "date", ascending=False
        )
        st.dataframe(news_df, width="stretch", hide_index=True)
    else:
        st.caption("No news listed.")

with tabs[4]:
    st.markdown("**Fundamentals**")
    fund = {k: v for k, v in rep.fundamentals.model_dump().items() if k != "source_ids"}
    fund_rows = [{"Field": k, "Value": "-" if v is None else str(v)} for k, v in fund.items()]
    st.dataframe(
        pd.DataFrame(fund_rows),
        width="stretch",
        hide_index=True,
    )
    if rep.earnings:
        st.markdown("**Earnings**")
        st.dataframe(
            pd.DataFrame([e.model_dump() for e in rep.earnings]),
            width="stretch",
            hide_index=True,
        )
    if rep.dividends.history:
        st.markdown(
            f"**Dividends** · yield {rep.dividends.dividend_yield_pct}% · "
            f"TTM DPS {rep.dividends.trailing_12m_dps_pkr} PKR"
        )
        st.dataframe(
            pd.DataFrame([d.model_dump() for d in rep.dividends.history]),
            width="stretch",
            hide_index=True,
        )
    if rep.analyst_views:
        st.markdown("**Analyst views**")
        st.dataframe(
            pd.DataFrame([a.model_dump() for a in rep.analyst_views]),
            width="stretch",
            hide_index=True,
        )

with tabs[5]:
    if rep.upcoming_events:
        st.dataframe(
            pd.DataFrame([e.model_dump() for e in rep.upcoming_events]),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No upcoming events listed.")

with tabs[6]:
    for src in rep.sources:
        title = src.title or src.url or src.id
        link = f"[{title}]({src.url})" if src.url else title
        st.markdown(
            f"- **[{src.id}]** {link} · {src.publisher or ''} · {src.published_date or 'n.d.'}"
        )
    if rep.data_gaps:
        st.markdown("**Data gaps**")
        st.markdown("\n".join(f"- {g}" for g in rep.data_gaps))

with tabs[7]:
    for warn in rf.warnings:
        st.warning(warn)
    st.json(json.loads(rf.path.read_text(encoding="utf-8-sig")))

disclaimer()
