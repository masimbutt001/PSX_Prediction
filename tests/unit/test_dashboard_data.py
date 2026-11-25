"""Tests for the redesigned dashboard's data helpers, figures and page wiring."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from psx_predictor.config.loader import load_config
from psx_predictor.dashboard import data
from psx_predictor.dashboard.app import build_pages, freshness_label
from psx_predictor.dashboard.components.figures import (
    calibration_chart,
    drawdown_chart,
    equity_chart,
    price_chart,
    probability_gauge,
)
from psx_predictor.dashboard.components.ui import fmt_pct, tone_of
from psx_predictor.storage.parquet_io import write_parquet_atomic
from psx_predictor.storage.paths import ensure_directories

VIEWS_DIR = Path(__file__).resolve().parents[2] / "src" / "psx_predictor" / "dashboard" / "views"


def _prices(n: int = 60, end: str = "2026-11-20", start_price: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=n)
    rng = np.random.default_rng(1)
    close = start_price + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "trade_date": dates.strftime("%Y-%m-%d"),
            "open": close - 0.3,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": rng.integers(1_000, 5_000, n),
        }
    )


def _pred(symbol: str, target: str, up: float, generated: str, outcome=None, correct=None):
    return {
        "prediction_id": f"{symbol}-{target}-{generated}",
        "symbol": symbol,
        "generated_at": generated,
        "target_date": target,
        "model_name": "test",
        "up_probability": up,
        "signal": "BUY" if up >= 0.55 else "SELL" if up <= 0.45 else "HOLD",
        "confidence": abs(up - 0.5) * 2,
        "drivers_json": json.dumps({"rsi_14": 0.2}),
        "research_json": "{}",
        "realized_outcome": outcome,
        "is_correct": correct,
    }


@pytest.fixture
def preds() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _pred("OGDC", "2026-11-19", 0.6, "2026-11-18T12:00", 0.01, True),
            # re-run of the same session: the later record should win
            _pred("OGDC", "2026-11-20", 0.4, "2026-11-19T12:00", -0.02, True),
            _pred("OGDC", "2026-11-20", 0.7, "2026-11-19T13:00", -0.02, False),
            _pred("OGDC", "2026-11-24", 0.65, "2026-11-20T12:00"),
            _pred("PPL", "2026-11-20", 0.3, "2026-11-19T12:00", 0.01, False),
        ]
    )


def test_dedupe_keeps_latest_generated(preds: pd.DataFrame) -> None:
    out = data.dedupe_predictions(preds)
    assert len(out) == 4
    row = out[(out["symbol"] == "OGDC") & (out["target_date"] == "2026-11-20")].iloc[0]
    assert row["up_probability"] == 0.7


def test_latest_prediction_per_symbol(preds: pd.DataFrame) -> None:
    latest = data.latest_prediction_per_symbol(preds)
    assert latest["OGDC"]["target_date"] == "2026-11-24"
    assert latest["PPL"]["target_date"] == "2026-11-20"


def test_track_record_and_scores(preds: pd.DataFrame) -> None:
    rec = data.track_record(preds).set_index("symbol")
    assert rec.loc["OGDC", "predictions"] == 2
    assert rec.loc["OGDC", "hit_rate"] == pytest.approx(0.5)
    # OGDC: (0.6-1)^2 and (0.7-0)^2
    assert rec.loc["OGDC", "brier"] == pytest.approx((0.16 + 0.49) / 2)
    scores = data.overall_scores(preds)
    assert scores["n"] == 3
    assert data.overall_scores(pd.DataFrame())["hit_rate"] is None


def test_prediction_history_marks_pending(preds: pd.DataFrame) -> None:
    hist = data.prediction_history(preds, "OGDC")
    assert list(hist["Session"]) == ["2026-11-24", "2026-11-20", "2026-11-19"]
    assert list(hist["Result"]) == ["Pending", "Missed", "Correct"]
    assert np.isnan(hist["Actual move"].iloc[0])


def test_universe_snapshot_and_staleness(tmp_path: Path, preds: pd.DataFrame) -> None:
    cfg = load_config()
    paths = ensure_directories(tmp_path)
    write_parquet_atomic(_prices(), paths["processed_prices"] / "OGDC.parquet")
    write_parquet_atomic(_prices(end="2026-11-18"), paths["processed_prices"] / "PPL.parquet")

    snap = data.universe_snapshot(cfg, paths, preds, {"OGDC": "Buy"})
    assert len(snap) == len(cfg.stocks)
    ogdc = snap.set_index("symbol").loc["OGDC"]
    assert ogdc["last_date"] == "2026-11-20"
    assert len(ogdc["trend"]) == data.SPARK_SESSIONS
    assert ogdc["signal"] == "BUY" and ogdc["research"] == "Buy"
    stale = data.stale_symbols(snap)
    assert "PPL" in stale and "OGDC" not in stale and "ENGROH" in stale

    text, ok, warn = freshness_label(snap)
    assert ok and warn and "behind" in text

    moves = data.sector_moves(snap)
    assert set(moves["sector"]) <= set(snap["sector"])


def test_read_pipeline_runs(tmp_path: Path) -> None:
    log = tmp_path / "pipeline.log"
    log.write_text(
        "\n".join(
            [
                json.dumps({"started_at": "2026-11-19T12:00:00+00:00", "success": True}),
                "not json",
                json.dumps(
                    {"started_at": "2026-11-20T12:00:00+00:00", "success": False, "dry_run": True}
                ),
                json.dumps({"started_at": "2026-11-21T12:00:00+00:00", "success": False}),
            ]
        ),
        encoding="utf-8",
    )
    runs = data.read_pipeline_runs(log)
    assert [r["started_at"][:10] for r in runs] == ["2026-11-21", "2026-11-20", "2026-11-19"]
    assert data.latest_live_run(runs)["started_at"].startswith("2026-11-21")
    assert data.read_pipeline_runs(tmp_path / "missing.log") == []


def test_format_pkt() -> None:
    assert data.format_pkt("2026-11-21T12:00:00+00:00", "%H:%M") == "17:00"
    assert data.format_pkt("2026-11-21T12:00:00", "%H:%M") == "17:00"


def test_parse_json() -> None:
    assert data.parse_json('{"a": 1}') == {"a": 1}
    assert data.parse_json("bad") == {}
    assert data.parse_json(None) == {}


def test_figures_build() -> None:
    df = _prices(80)
    fig = price_chart(df, "OGDC", bollinger=True, forecast=("2026-11-24", 0.62), mode="dark")
    assert isinstance(fig, go.Figure)
    names = {t.name for t in fig.data}
    assert {"OGDC", "SMA 20", "SMA 50", "Volume", "Next session"} <= names

    assert isinstance(probability_gauge(0.62, mode="light"), go.Figure)

    eq = pd.DataFrame(
        {"trade_date": df["trade_date"], "total_equity": np.linspace(1e6, 1.2e6, len(df))}
    )
    assert len(equity_chart(eq, eq, "Test", mode="dark").data) == 2
    assert len(drawdown_chart(eq, mode="dark").data) == 1

    res = pd.DataFrame({"up_probability": [0.2, 0.4, 0.6, 0.8], "actual_up": [0, 0, 1, 1]})
    assert len(calibration_chart(res, mode="dark").data) == 2


def test_formatting_helpers() -> None:
    assert fmt_pct(1.234) == "▲ +1.23%"
    assert fmt_pct(-0.5) == "▼ -0.50%"
    assert fmt_pct(None) == "—"
    assert tone_of(2.0) == "up" and tone_of(-1.0) == "down" and tone_of(None) == "muted"


def test_navigation_pages_exist() -> None:
    files = {"overview.py", "stock.py", "performance.py", "research.py", "pipeline.py"}
    assert files <= {p.name for p in VIEWS_DIR.glob("*.py")}
    assert len(build_pages()) == len(files)


APP_FILE = VIEWS_DIR.parent / "app.py"


@pytest.mark.parametrize(
    "page", ["overview.py", "stock.py", "performance.py", "research.py", "pipeline.py"]
)
def test_pages_render_without_errors(
    tmp_path: Path, preds: pd.DataFrame, monkeypatch: pytest.MonkeyPatch, page: str
) -> None:
    """Each page runs end to end on a small data folder without raising."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_data.clear()
    st.cache_resource.clear()
    paths = ensure_directories(tmp_path)
    for sym in ("OGDC", "PPL"):
        write_parquet_atomic(_prices(), paths["processed_prices"] / f"{sym}.parquet")
    write_parquet_atomic(preds, paths["predictions"] / "predictions.parquet")
    (tmp_path / "pipeline.log").write_text(
        json.dumps(
            {
                "started_at": "2026-11-20T12:00:00+00:00",
                "duration_seconds": 10,
                "success": False,
                "tasks": [
                    {"step": "update_prices", "status": "FAILED", "error": "ENGRO: no rows"},
                    {"step": "collect_news", "status": "SUCCESS"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PSX_DATA_DIR", str(tmp_path))

    at = AppTest.from_file(str(APP_FILE), default_timeout=60)
    at.run()
    if page != "overview.py":
        at.switch_page(str(VIEWS_DIR / page))
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    if page in ("overview.py", "pipeline.py"):
        assert len(at.dataframe) >= 1


def test_call_sessions_counts_scored_and_right(preds: pd.DataFrame) -> None:
    s = data.call_sessions(preds)
    assert list(s["target_date"]) == ["2026-11-24", "2026-11-20", "2026-11-19"]
    row = s.set_index("target_date").loc["2026-11-20"]
    # deduped: OGDC (latest re-run, wrong) and PPL (wrong)
    assert (row["calls"], row["scored"], row["correct"]) == (2, 2, 0)
    assert s.set_index("target_date").loc["2026-11-24", "scored"] == 0
    # the newest day is not closed yet, so the default is the newest scored day
    assert data.latest_scored_session(s) == "2026-11-20"
    assert data.latest_scored_session(data.call_sessions(pd.DataFrame())) is None


def test_session_calls_joins_closes_and_results(preds: pd.DataFrame) -> None:
    px = _prices(end="2026-11-20")
    calls = data.session_calls(preds, "2026-11-20", prices=lambda sym: px)
    assert list(calls["Stock"]) == ["OGDC", "PPL"]
    ogdc = calls.iloc[0]
    assert ogdc["Call"] == "BUY" and ogdc["Result"] == "Wrong" and ogdc["Went"] == "Down"
    assert ogdc["Actual move"] == pytest.approx(-2.0)
    assert ogdc["Close"] == pytest.approx(px["close"].iloc[-1])
    assert ogdc["Previous close"] == pytest.approx(px["close"].iloc[-2])
    assert calls.iloc[1]["Went"] == "Up"

    pending = data.session_calls(preds, "2026-11-24", prices=lambda sym: px)
    assert list(pending["Result"]) == ["Pending"]
    assert pending.iloc[0]["Went"] == "Not closed yet"
    assert np.isnan(pending.iloc[0]["Close"])

    right_first = data.session_calls(preds, "2026-11-19")
    assert list(right_first["Result"]) == ["Right"]
    assert np.isnan(right_first.iloc[0]["Previous close"])
