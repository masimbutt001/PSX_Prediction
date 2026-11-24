"""Pipeline health: last daily runs, failed steps, and how fresh each data set is."""

import pandas as pd
import streamlit as st

from psx_predictor.dashboard import data, state
from psx_predictor.dashboard.components.ui import disclaimer, kpi_row, page_header, step_row
from psx_predictor.dashboard.theme import palette

p = palette()
page_header("Pipeline health", "What the daily run did, what failed, and how fresh the data is.")

runs = state.pipeline_runs(limit=30)
live_runs = [r for r in runs if not r.get("dry_run")]
last = data.latest_live_run(runs)
snap = state.snapshot()
stale = data.stale_symbols(snap)
macro = state.macro()
preds = state.predictions()

last_ok = next((r for r in live_runs if r.get("success")), None)
failed_steps = [t for t in (last or {}).get("tasks", []) if t.get("status") == "FAILED"]
kpi_row(
    [
        (
            "Last run",
            data.format_pkt(last["started_at"], "%a %d %b") if last else "—",
            ("Finished" if last.get("success") else f"{len(failed_steps)} step(s) failed")
            if last
            else "No run logged yet",
            ("up" if last.get("success") else "down") if last else "muted",
        ),
        (
            "Last full success",
            data.format_pkt(last_ok["started_at"], "%a %d %b") if last_ok else "—",
            "All steps OK" if last_ok else "None logged",
            "up" if last_ok else "muted",
        ),
        (
            "Prices up to date",
            f"{len(snap) - len(stale)} / {len(snap)}",
            ("Behind: " + ", ".join(stale)) if stale else "Every stock has the newest session",
            "warn" if stale else "up",
        ),
        (
            "Macro data",
            str(macro["session_date"].iloc[-1]) if not macro.empty else "—",
            "Newest macro session" if not macro.empty else "No macro data yet",
            "muted",
        ),
        (
            "Predictions logged",
            f"{len(data.dedupe_predictions(preds)):,}",
            f"{len(preds) - len(data.dedupe_predictions(preds))} duplicates ignored"
            if not preds.empty
            else "None yet",
            "muted",
        ),
    ]
)
st.write("")

left, right = st.columns([1, 1.4], gap="large")
with left:
    with st.container(border=True):
        st.markdown("##### Latest run, step by step")
        if not last:
            st.caption("No daily run logged yet.")
        else:
            st.caption(
                f"{data.format_pkt(last['started_at'])} · "
                f"{float(last.get('duration_seconds', 0)):.0f}s"
            )
            st.html(
                "".join(
                    step_row(
                        data.STEP_LABELS.get(t.get("step", ""), t.get("step", "")),
                        t.get("status", ""),
                        f"{float(t.get('duration_seconds', 0)):.1f}s",
                    )
                    for t in last.get("tasks", [])
                )
            )
            for t in failed_steps:
                st.error(
                    f"**{data.STEP_LABELS.get(t['step'], t['step'])}:** "
                    f"{t.get('error') or t.get('message') or 'failed'}"
                )
    with st.container(border=True):
        st.markdown("##### Run it by hand")
        st.code("uv run psx schedule run-daily --catch-up", language="bash")
        st.caption(
            "The Windows scheduled task runs this every hour after the market closes "
            "and stops once the day has a successful run."
        )

with right:
    with st.container(border=True):
        st.markdown("##### Price data by stock")
        fresh = pd.DataFrame(
            {
                "Stock": snap["symbol"],
                "Last session": snap["last_date"].fillna("missing"),
                "Status": ["Behind" if s in stale else "Up to date" for s in snap["symbol"]],
                "Has prediction": snap["signal"].notna(),
            }
        )

        def _status_color(v: object) -> str:
            return f"color: {p.up}" if v == "Up to date" else f"color: {p.warn}; font-weight: 600"

        st.dataframe(
            fresh.style.map(_status_color, subset=["Status"]),
            hide_index=True,
            width="stretch",
            height=36 * (len(fresh) + 1) + 3,
            row_height=36,
        )
    with st.container(border=True):
        st.markdown("##### Recent runs")
        if not runs:
            st.caption("Nothing in data/pipeline.log yet.")
        else:
            hist = pd.DataFrame(
                [
                    {
                        "Started": data.format_pkt(r["started_at"], "%d %b %H:%M"),
                        "Result": "Dry run"
                        if r.get("dry_run")
                        else ("Success" if r.get("success") else "Failed"),
                        "Failed steps": ", ".join(
                            data.STEP_LABELS.get(t["step"], t["step"])
                            for t in r.get("tasks", [])
                            if t.get("status") == "FAILED"
                        )
                        or "—",
                        "Seconds": round(float(r.get("duration_seconds", 0))),
                    }
                    for r in runs
                ]
            )

            def _res_color(v: object) -> str:
                return {"Success": f"color: {p.up}", "Failed": f"color: {p.down}"}.get(
                    str(v), f"color: {p.muted}"
                )

            st.dataframe(
                hist.style.map(_res_color, subset=["Result"]), hide_index=True, width="stretch"
            )

disclaimer()
