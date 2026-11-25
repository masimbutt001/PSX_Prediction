"""Pipeline health: last daily runs, failed steps, and how fresh each data set is."""

import pandas as pd
import streamlit as st

from psx_predictor.dashboard import data, state
from psx_predictor.dashboard.components.help import section_title
from psx_predictor.dashboard.components.jobs_panel import run_button
from psx_predictor.dashboard.components.ui import disclaimer, kpi_row, page_header, step_row
from psx_predictor.dashboard.theme import palette

p = palette()
page_header("Pipeline health", "What the daily run did, what failed, and how fresh the data is.")

runs = state.pipeline_runs(limit=30)
live_runs = [r for r in runs if not r.get("dry_run")]
last = data.latest_live_run(runs)
base_run, step_tasks = data.latest_steps(runs)
snap = state.snapshot()
stale = data.stale_symbols(snap)
macro = state.macro()
preds = state.predictions()

last_ok = next((r for r in live_runs if r.get("success") and not r.get("partial")), None)
failed_steps = [t for t in step_tasks if t.get("status") == "FAILED"]
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
    ],
    ["pl_last_run", "pl_last_ok", "pl_prices", "pl_macro", "pl_predictions"],
)
st.write("")

left, right = st.columns([1, 1.4], gap="large")
with left:
    with st.container(border=True):
        section_title("Run now", "pl_run_now")
        r1, r2, r3 = st.columns(3)
        with r1:
            run_button(
                "Run full pipeline",
                "run_pipeline",
                key="pl_run_all",
                primary=True,
                help="All 6 steps, continuing past a failed step",
                width="stretch",
            )
        with r2:
            run_button(
                "Update all prices",
                "update_prices",
                key="pl_update_all",
                icon=":material/download:",
                width="stretch",
            )
        with r3:
            run_button(
                "Score predictions",
                "audit_predictions",
                key="pl_audit",
                icon=":material/fact_check:",
                help="Check past predictions against the closes now on file",
                width="stretch",
            )
        st.caption(
            "Jobs run in the background on this computer; progress shows at the top of the "
            "page. The Windows scheduled task still runs "
            "`uv run psx schedule run-daily --catch-up` every hour after the market closes."
        )

    with st.container(border=True):
        section_title("Latest run, step by step", "pl_steps")
        if not base_run:
            st.caption("No daily run logged yet.")
        else:
            st.caption(
                f"{data.format_pkt(base_run['started_at'])} · "
                f"{float(base_run.get('duration_seconds', 0)):.0f}s"
                + (
                    " · later reruns shown per step"
                    if any(t.get("rerun") for t in step_tasks)
                    else ""
                )
            )
            for t in step_tasks:
                step = str(t.get("step", ""))
                c_row, c_btn = st.columns([5, 1.3], vertical_alignment="center")
                c_row.html(
                    step_row(
                        data.STEP_LABELS.get(step, step),
                        t.get("status", ""),
                        (
                            f"rerun {data.format_pkt(t['rerun_at'], '%d %b %H:%M')}"
                            if t.get("rerun")
                            else f"{float(t.get('duration_seconds', 0)):.1f}s"
                        ),
                    )
                )
                with c_btn:
                    run_button(
                        "Rerun",
                        "run_step",
                        step=step,
                        key=f"pl_rerun_{step}",
                        icon=":material/replay:",
                        help=f"Run only the {data.STEP_LABELS.get(step, step)} step again",
                        width="stretch",
                    )
            for t in failed_steps:
                st.error(
                    f"**{data.STEP_LABELS.get(t['step'], t['step'])}:** "
                    f"{t.get('error') or t.get('message') or 'failed'}"
                )


def _result_label(run: dict) -> str:
    if run.get("dry_run"):
        return "Dry run"
    outcome = "Success" if run.get("success") else "Failed"
    return f"{outcome} (rerun)" if run.get("partial") else outcome


with right:
    with st.container(border=True):
        section_title("Price data by stock", "pl_price_table")
        if stale:
            run_button(
                f"Update {len(stale)} stock(s) that are behind",
                "update_prices",
                symbols=stale,
                key="pl_update_stale",
                icon=":material/download:",
                help="Fetch the newest sessions for: " + ", ".join(stale),
            )
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
        section_title("Recent runs", "pl_recent")
        if not runs:
            st.caption("Nothing in data/pipeline.log yet.")
        else:
            hist = pd.DataFrame(
                [
                    {
                        "Started": data.format_pkt(r["started_at"], "%d %b %H:%M"),
                        "Result": _result_label(r),
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
                return {
                    "Success": f"color: {p.up}",
                    "Failed": f"color: {p.down}",
                    "Success (rerun)": f"color: {p.up}",
                    "Failed (rerun)": f"color: {p.down}",
                }.get(str(v), f"color: {p.muted}")

            st.dataframe(
                hist.style.map(_res_color, subset=["Result"]), hide_index=True, width="stretch"
            )

disclaimer()
