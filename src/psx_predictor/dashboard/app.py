"""PSX Predictor dashboard entry point: theme, top navigation and the shared header.

Run with `uv run psx dashboard` from the repository root (so `.streamlit/config.toml`
and `data/` are found).
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from psx_predictor import __version__
from psx_predictor.dashboard import data, state
from psx_predictor.dashboard.components.help import help_button, inject_speech
from psx_predictor.dashboard.components.jobs_panel import job_panel
from psx_predictor.dashboard.components.ui import status_pill
from psx_predictor.dashboard.theme import inject_css

PAGES_DIR = Path(__file__).parent / "views"
ASSETS_DIR = Path(__file__).parent / "assets"
STOCK_PAGE = str(PAGES_DIR / "stock.py")


def build_pages() -> list["st.Page"]:
    """Pages in navigation order."""
    return [
        st.Page(
            str(PAGES_DIR / "overview.py"),
            title="Overview",
            icon=":material/space_dashboard:",
            default=True,
        ),
        st.Page(STOCK_PAGE, title="Stocks", icon=":material/candlestick_chart:"),
        st.Page(
            str(PAGES_DIR / "performance.py"), title="Performance", icon=":material/query_stats:"
        ),
        st.Page(str(PAGES_DIR / "research.py"), title="Research", icon=":material/menu_book:"),
        st.Page(str(PAGES_DIR / "pipeline.py"), title="Pipeline", icon=":material/sync:"),
    ]


def freshness_label(snap: pd.DataFrame) -> tuple[str, bool, bool]:
    """(text, ok, warn) for the data-freshness pill in the header."""
    if snap.empty or not snap["has_data"].any():
        return "No price data yet · update prices on the Pipeline page", False, False
    newest = str(snap["last_date"].dropna().max())
    stale = data.stale_symbols(snap)
    try:
        day = pd.Timestamp(newest).strftime("%a %d %b %Y")
    except ValueError:
        day = newest
    if stale:
        return f"Prices to {day} · {len(stale)} behind: {', '.join(stale)}", True, True
    return f"Prices to {day} · all {len(snap)} up to date", True, False


def _on_search() -> None:
    choice = st.session_state.get("header_search")
    if choice:
        state.select_symbol(choice)
        st.session_state["header_search"] = None
        st.session_state["_goto_stock"] = True


def render_header() -> None:
    """Brand, quick stock search, data freshness and refresh, above every page."""
    snap = state.snapshot()
    c_search, c_status, c_btn = st.columns([3, 5.5, 1.1], vertical_alignment="center")
    with c_search:
        syms = state.symbols()
        names = {s.symbol: s.name for s in state.get_config().stocks}
        st.selectbox(
            "Search stocks",
            options=syms,
            index=None,
            placeholder="Search a stock…",
            format_func=lambda s: f"{s} · {names.get(s, '')}",
            label_visibility="collapsed",
            key="header_search",
            on_change=_on_search,
        )
        if st.session_state.pop("_goto_stock", False):
            st.switch_page(STOCK_PAGE)
    with c_status:
        label, ok, warn = freshness_label(snap)
        with st.container(horizontal=True, vertical_alignment="center", gap="small", wrap=False):
            st.html(status_pill(label, ok=ok, warn=warn), width="content")
            help_button("hdr_freshness")
    with c_btn:
        if st.button(
            "Refresh",
            icon=":material/refresh:",
            width="stretch",
            help="Re-read the data files from disk",
        ):
            state.refresh()
            st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="PSX Predictor",
        page_icon=":material/monitoring:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.logo(str(ASSETS_DIR / "logo.svg"), icon_image=str(ASSETS_DIR / "icon.svg"), size="large")
    inject_css()
    inject_speech()
    nav = st.navigation(build_pages(), position="top")
    render_header()
    job_panel()
    nav.run()
    st.caption(f"PSX Predictor v{__version__} · runs locally on your computer")


if __name__ == "__main__":
    main()
