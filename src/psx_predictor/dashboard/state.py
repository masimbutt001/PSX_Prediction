"""Cached data loaders and the shared stock selection for the dashboard pages."""

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from psx_predictor.config.loader import load_config
from psx_predictor.config.schema import AppConfig
from psx_predictor.dashboard import data
from psx_predictor.storage.paths import get_storage_paths

CACHE_TTL_SECONDS = 300
SYMBOL_KEY = "psx_symbol"


@st.cache_resource
def get_config() -> AppConfig:
    return load_config()


def get_paths() -> dict[str, Path]:
    return get_storage_paths(get_config().settings.data_dir)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def predictions() -> pd.DataFrame:
    return data.load_predictions(get_paths())


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def prices(symbol: str) -> pd.DataFrame:
    return data.load_prices(get_paths(), symbol)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def research_stances() -> dict[str, str]:
    """Latest valid research stance per symbol (empty when no research files)."""
    try:
        from psx_predictor.research.store import ResearchStore

        files = ResearchStore(research_dir=get_paths()["root"] / "research").scan()
    except Exception:
        return {}
    latest: dict[str, Any] = {}
    for rf in files:
        if not rf.is_valid or rf.report is None or rf.research_date is None:
            continue
        cur = latest.get(rf.symbol)
        if cur is None or cur.research_date < rf.research_date:
            latest[rf.symbol] = rf
    return {sym: str(rf.report.recommendation.stance or "—").title() for sym, rf in latest.items()}


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def snapshot() -> pd.DataFrame:
    return data.universe_snapshot(get_config(), get_paths(), predictions(), research_stances())


@st.cache_data(ttl=60, show_spinner=False)
def pipeline_runs(limit: int = 20) -> list[dict[str, Any]]:
    return data.read_pipeline_runs(get_paths()["root"] / "pipeline.log", limit=limit)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def news(symbol: Optional[str] = None, limit: int = 20) -> pd.DataFrame:
    return data.load_news(get_paths(), symbol=symbol, limit=limit)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def macro() -> pd.DataFrame:
    return data.load_macro(get_paths())


def refresh() -> None:
    """Drop cached data so the next render reads files from disk again."""
    st.cache_data.clear()


def symbols() -> list[str]:
    return [s.symbol for s in get_config().stocks]


def selected_symbol() -> str:
    """Stock chosen anywhere in the app (header search, watchlist row, stock page)."""
    syms = symbols()
    cur = st.session_state.get(SYMBOL_KEY)
    if cur not in syms:
        cur = syms[0] if syms else ""
        st.session_state[SYMBOL_KEY] = cur
    return str(cur)


def select_symbol(symbol: str) -> None:
    st.session_state[SYMBOL_KEY] = symbol
