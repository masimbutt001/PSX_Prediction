"""Tests for the explanation popups: every (?) button has text and the text reads aloud."""

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from psx_predictor.dashboard.components.help import listen_bar
from psx_predictor.dashboard.help_text import HELP, speech_text
from psx_predictor.storage.parquet_io import write_parquet_atomic
from psx_predictor.storage.paths import ensure_directories

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "src" / "psx_predictor" / "dashboard"
FIXTURE = ROOT / "tests" / "fixtures" / "research" / "OGDC_2026-11-20.json"

# help_button("key"), section_title("..", "key"), helped("key"), inline_help("..", "key")
# and the key lists passed to kpi_row.
CALL = re.compile(
    r"(?:help_button|helped)\(\s*\"(\w+)\"|(?:section_title|inline_help)\([^;]*?,\s*\"(\w+)\""
)
KEY_LIST = re.compile(r"\[\s*(?:\"(?:ov|pf|bt|pl|sd|rs|hdr)_\w+\",?\s*)+\]")
KEY = re.compile(r"\"(\w+)\"")


def _used_keys() -> set[str]:
    keys: set[str] = set()
    for f in [DASH / "app.py", *sorted((DASH / "views").glob("*.py"))]:
        src = f.read_text(encoding="utf-8")
        for a, b in CALL.findall(src):
            keys.add(a or b)
        for lst in KEY_LIST.findall(src):
            keys.update(KEY.findall(lst))
    return keys


def test_every_help_key_used_on_a_page_has_text() -> None:
    used = _used_keys()
    assert len(used) >= 50
    missing = sorted(used - set(HELP))
    assert not missing, f"Add these to dashboard/help_text.py: {missing}"


def test_no_orphan_help_text() -> None:
    orphans = sorted(set(HELP) - _used_keys())
    assert not orphans, f"Help text that no page shows: {orphans}"


@pytest.mark.parametrize("key", sorted(HELP))
def test_topic_text_is_plain_and_complete(key: str) -> None:
    t = HELP[key]
    assert t.title and len(t.body) > 80
    say = speech_text(t)
    assert len(say) > len(t.title) + 60
    assert not re.search(r"\*\*|`|^- ", say), say
    assert "%" not in say and "PKR" not in say
    assert say.endswith((".", "?", "!"))


def test_listen_bar_escapes_text() -> None:
    html = listen_bar("ov_signals")
    assert 'class="psx-listen"' in html and "data-say=" in html
    assert "<b>" not in html and "**" not in html
    assert "52 percent" in html


def _prices(n: int = 260) -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-11-20", periods=n)
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.015, n)))
    return pd.DataFrame(
        {
            "trade_date": dates.strftime("%Y-%m-%d"),
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(100_000, 900_000, n),
        }
    )


@pytest.mark.parametrize("page", ["stock.py", "research.py", "performance.py"])
def test_pages_with_data_render_their_help(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, page: str
) -> None:
    """Pages render with the (?) buttons when a stock has prices, predictions and research."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from psx_predictor.dashboard.state import SYMBOL_KEY

    st.cache_data.clear()
    st.cache_resource.clear()
    paths = ensure_directories(tmp_path)
    write_parquet_atomic(_prices(), paths["processed_prices"] / "OGDC.parquet")
    pred = {
        "prediction_id": "p1",
        "symbol": "OGDC",
        "generated_at": "2026-11-20T12:00",
        "target_date": "2026-11-21",
        "model_name": "XGBoost",
        "up_probability": 0.56,
        "signal": "BUY",
        "confidence": 0.12,
        "drivers_json": json.dumps({"rsi_14": 0.2, "atr_14": 0.1}),
        "research_json": json.dumps({"stance": "buy", "research_date": "2026-11-20"}),
        "realized_outcome": None,
        "is_correct": None,
    }
    write_parquet_atomic(pd.DataFrame([pred]), paths["predictions"] / "predictions.parquet")
    (tmp_path / "research").mkdir(exist_ok=True)
    shutil.copy(FIXTURE, tmp_path / "research" / FIXTURE.name)
    monkeypatch.setenv("PSX_DATA_DIR", str(tmp_path))

    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=120)
    at.session_state[SYMBOL_KEY] = "OGDC"
    at.run()
    at.switch_page(str(DASH / "views" / page))
    at.run()
    assert not at.exception, [e.value for e in at.exception]
