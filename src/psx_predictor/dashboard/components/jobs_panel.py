"""Run buttons and the live job panel shown under the header on every page."""

from typing import Optional

import streamlit as st

from psx_predictor.dashboard import state
from psx_predictor.dashboard.jobs import Job, JobError

ACK_KEY = "psx_job_seen_done"
POLL_SECONDS = 2


def run_button(
    label: str,
    action: str,
    *,
    symbols: Optional[list[str]] = None,
    step: Optional[str] = None,
    key: str,
    primary: bool = False,
    icon: str = ":material/play_arrow:",
    help: Optional[str] = None,
    width: str = "content",
) -> None:
    """A button that starts one allowed background job."""
    allowed = state.commands_allowed()
    busy = state.job_manager().current() is not None
    tip = help
    if not allowed:
        tip = "Commands can only be run from the computer the dashboard runs on."
    elif busy:
        tip = "Another job is running. Wait for it to finish."
    if st.button(
        label,
        key=key,
        type="primary" if primary else "secondary",
        icon=icon,
        disabled=not allowed or busy,
        help=tip,
        width=width,  # type: ignore[arg-type]
    ):
        try:
            state.job_manager().start(
                action,
                symbols=symbols,
                step=step,
                allowed_symbols=set(state.symbols()),
            )
        except JobError as exc:
            st.toast(str(exc), icon=":material/error:")
            return
        st.session_state.pop(ACK_KEY, None)
        st.rerun()


def _status_line(job: Job) -> tuple[str, str]:
    secs = job.elapsed_seconds()
    took = f"{secs / 60:.1f} min" if secs >= 90 else f"{secs:.0f}s"
    if job.running:
        return f"**{job.label}** is running · {took}", "running"
    if job.status == "succeeded":
        return f"**{job.label}** finished in {took}", "complete"
    if job.status == "lost":
        return (
            f"**{job.label}**: the dashboard restarted while it ran, so its result is unknown",
            "error",
        )
    return f"**{job.label}** failed (exit code {job.returncode}) after {took}", "error"


def _render(job: Job) -> None:
    text, st_state = _status_line(job)
    with st.status(text, state=st_state, expanded=job.running or job.status == "failed"):  # type: ignore[arg-type]
        log = state.job_manager().read_log(job.id, tail=40)
        st.code(log or "(no output yet)", language=None, height=300)
        if not job.running:
            st.caption("Full log: " + str(state.job_manager().log_path(job.id)))


@st.fragment(run_every=POLL_SECONDS)
def _live_panel(job_id: str) -> None:
    job = state.job_manager().get(job_id)
    if job is None:
        return
    _render(job)
    if not job.running:
        # Finished: reload data from disk and redraw the whole page once.
        st.session_state[ACK_KEY] = job.id
        state.refresh()
        st.rerun(scope="app")


def job_panel() -> None:
    """Live status of the running job, or the result of the last one until dismissed."""
    mgr = state.job_manager()
    job = mgr.current()
    if job is not None:
        _live_panel(job.id)
        return
    recent = mgr.recent(limit=1)
    if not recent:
        return
    last = recent[0]
    if st.session_state.get(ACK_KEY) != last.id:
        return
    c1, c2 = st.columns([12, 1], vertical_alignment="top")
    with c1:
        _render(last)
    with c2:
        if st.button("Hide", key="job_hide", help="Hide this result"):
            st.session_state[ACK_KEY] = None
            st.rerun()
