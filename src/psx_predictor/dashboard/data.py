"""Data access for the dashboard pages.

Plain functions (no Streamlit calls) so they can be unit tested; the pages wrap the
slow ones in `st.cache_data`.
"""

import datetime
import json
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from psx_predictor.config.schema import AppConfig
from psx_predictor.storage.parquet_io import read_parquet

SPARK_SESSIONS = 30
STEP_LABELS = {
    "update_prices": "Prices",
    "collect_news": "News and announcements",
    "update_macro": "Macro (USD/PKR, Brent, KSE-100)",
    "build_features": "Features",
    "generate_predictions": "Predictions",
    "audit_predictions": "Audit of past predictions",
}


def _date_col(df: pd.DataFrame) -> str:
    return "trade_date" if "trade_date" in df.columns else "date"


def load_prices(paths: dict[str, Path], symbol: str) -> pd.DataFrame:
    """Processed daily prices for `symbol`, oldest first; empty when missing or unreadable."""
    f = paths["processed_prices"] / f"{symbol}.parquet"
    if not f.exists():
        return pd.DataFrame()
    try:
        df = read_parquet(f)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    d = _date_col(df)
    df = df.sort_values(d).reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df[d]).dt.strftime("%Y-%m-%d")
    return df


def load_predictions(paths: dict[str, Path]) -> pd.DataFrame:
    """All logged predictions, or an empty frame."""
    f = paths["predictions"] / "predictions.parquet"
    if not f.exists():
        return pd.DataFrame()
    try:
        return read_parquet(f)
    except Exception:
        return pd.DataFrame()


def dedupe_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    """Keep one prediction per (symbol, target_date): the most recently generated one.

    Re-running the daily pipeline logs a second record for the same session; the
    dashboard should count it once.
    """
    if preds.empty or not {"symbol", "target_date"}.issubset(preds.columns):
        return preds
    sort_cols = ["target_date"] + (["generated_at"] if "generated_at" in preds.columns else [])
    out = preds.sort_values(sort_cols)
    return out.drop_duplicates(subset=["symbol", "target_date"], keep="last").reset_index(drop=True)


def latest_prediction_per_symbol(preds: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Most recent prediction (by target date, then generation time) for each symbol."""
    preds = dedupe_predictions(preds)
    if preds.empty:
        return {}
    latest = preds.sort_values("target_date").groupby("symbol").tail(1)
    return {str(r["symbol"]): r.to_dict() for _, r in latest.iterrows()}


def parse_json(raw: Any) -> dict[str, Any]:
    """Decode a JSON text column value; {} for blanks and bad JSON."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        val = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return val if isinstance(val, dict) else {}


def universe_snapshot(
    cfg: AppConfig,
    paths: dict[str, Path],
    preds: Optional[pd.DataFrame] = None,
    research_stance: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """One row per configured stock: last close, moves, 30-session trend and latest signal."""
    preds = load_predictions(paths) if preds is None else preds
    latest = latest_prediction_per_symbol(preds)
    research_stance = research_stance or {}
    rows: list[dict[str, Any]] = []
    for stock in cfg.stocks:
        df = load_prices(paths, stock.symbol)
        close = chg_1d = chg_5d = None
        last_date = None
        spark: list[float] = []
        if not df.empty:
            closes = df["close"].astype(float)
            close = float(closes.iloc[-1])
            last_date = str(df["trade_date"].iloc[-1])
            if len(closes) >= 2:
                chg_1d = (close / float(closes.iloc[-2]) - 1.0) * 100.0
            if len(closes) >= 6:
                chg_5d = (close / float(closes.iloc[-6]) - 1.0) * 100.0
            spark = [round(float(v), 2) for v in closes.tail(SPARK_SESSIONS)]
        rec = latest.get(stock.symbol, {})
        up = rec.get("up_probability")
        rows.append(
            {
                "symbol": stock.symbol,
                "name": stock.name,
                "sector": stock.sector,
                "last_date": last_date,
                "close": close,
                "chg_1d": chg_1d,
                "chg_5d": chg_5d,
                "trend": spark,
                "signal": str(rec.get("signal", "")) or None,
                "up_prob": float(up) if up is not None and not pd.isna(up) else None,
                "confidence": (
                    float(rec["confidence"])
                    if rec.get("confidence") is not None and not pd.isna(rec["confidence"])
                    else None
                ),
                "target_date": rec.get("target_date"),
                "research": research_stance.get(stock.symbol),
                "has_data": not df.empty,
            }
        )
    return pd.DataFrame(rows)


def resolved_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    """Deduplicated predictions whose outcome is known, with an `actual_up` 0/1 column."""
    preds = dedupe_predictions(preds)
    if preds.empty or "realized_outcome" not in preds.columns:
        return pd.DataFrame()
    res = preds[preds["realized_outcome"].notna()].copy()
    if res.empty:
        return res
    res["actual_up"] = (res["realized_outcome"].astype(float) > 0).astype(int)
    res["is_correct"] = res["is_correct"].astype(bool)
    return res


def track_record(preds: pd.DataFrame, last_n: Optional[int] = None) -> pd.DataFrame:
    """Hit rate, Brier score and counts per symbol over each symbol's last `last_n` sessions."""
    res = resolved_predictions(preds)
    cols = ["symbol", "predictions", "hit_rate", "brier", "last_target"]
    if res.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for sym, g in res.sort_values("target_date").groupby("symbol"):
        if last_n:
            g = g.tail(last_n)
        p = g["up_probability"].astype(float).to_numpy()
        y = g["actual_up"].to_numpy(dtype=float)
        out.append(
            {
                "symbol": sym,
                "predictions": len(g),
                "hit_rate": float(g["is_correct"].mean()),
                "brier": float(np.mean((p - y) ** 2)),
                "last_target": str(g["target_date"].iloc[-1]),
            }
        )
    return pd.DataFrame(out, columns=cols).sort_values("hit_rate", ascending=False)


def overall_scores(preds: pd.DataFrame, days: Optional[int] = None) -> dict[str, Any]:
    """Hit rate and Brier score across all stocks, optionally for the last `days` days."""
    res = resolved_predictions(preds)
    if not res.empty and days:
        cutoff = (datetime.date.today() - datetime.timedelta(days=days)).isoformat()
        res = res[res["target_date"].astype(str) >= cutoff]
    if res.empty:
        return {"n": 0, "hit_rate": None, "brier": None}
    p = res["up_probability"].astype(float).to_numpy()
    y = res["actual_up"].to_numpy(dtype=float)
    return {
        "n": len(res),
        "hit_rate": float(res["is_correct"].mean()),
        "brier": float(np.mean((p - y) ** 2)),
    }


def prediction_history(preds: pd.DataFrame, symbol: str, limit: int = 30) -> pd.DataFrame:
    """Recent predictions for one symbol, newest first, with their outcome when known."""
    preds = dedupe_predictions(preds)
    if preds.empty:
        return pd.DataFrame()
    g = preds[preds["symbol"] == symbol].sort_values("target_date", ascending=False).head(limit)
    rows = []
    for _, r in g.iterrows():
        outcome = r.get("realized_outcome")
        known = outcome is not None and not pd.isna(outcome)
        rows.append(
            {
                "Session": str(r["target_date"]),
                "Signal": str(r.get("signal", "")),
                "Chance of UP": float(r["up_probability"]),
                "Actual move": float(outcome) * 100.0 if known else float("nan"),
                "Result": ("Correct" if bool(r.get("is_correct")) else "Missed")
                if known
                else "Pending",
            }
        )
    return pd.DataFrame(rows)


def call_sessions(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per target session, newest first: how many calls it has and how many are scored.

    Columns: `target_date`, `calls`, `scored`, `correct`, `called_on` (newest generation
    time of that session's calls).
    """
    preds = dedupe_predictions(preds)
    cols = ["target_date", "calls", "scored", "correct", "called_on"]
    if preds.empty or "target_date" not in preds.columns:
        return pd.DataFrame(columns=cols)
    df = preds.copy()
    df["target_date"] = df["target_date"].astype(str)
    outcome = (
        df["realized_outcome"] if "realized_outcome" in df.columns else pd.Series(np.nan, df.index)
    )
    df["_scored"] = outcome.notna()
    df["_correct"] = df["_scored"] & (
        df["is_correct"].fillna(False).astype(bool) if "is_correct" in df.columns else False
    )
    gen = df["generated_at"] if "generated_at" in df.columns else pd.Series("", df.index)
    df["_gen"] = gen.astype(str)
    out = (
        df.groupby("target_date")
        .agg(
            calls=("symbol", "size"),
            scored=("_scored", "sum"),
            correct=("_correct", "sum"),
            called_on=("_gen", "max"),
        )
        .reset_index()
        .sort_values("target_date", ascending=False)
        .reset_index(drop=True)
    )
    out[["calls", "scored", "correct"]] = out[["calls", "scored", "correct"]].astype(int)
    return out[cols]


def latest_scored_session(sessions: pd.DataFrame) -> Optional[str]:
    """Newest target session with at least one scored call, else the newest session."""
    if sessions.empty:
        return None
    scored = sessions[sessions["scored"] > 0]
    row = scored.iloc[0] if not scored.empty else sessions.iloc[0]
    return str(row["target_date"])


def _closes_around(prices: pd.DataFrame, target: str) -> tuple[float, float]:
    """(previous close, close) for the `target` session; NaN when the session is missing."""
    nan = float("nan")
    if prices is None or prices.empty or "close" not in prices.columns:
        return nan, nan
    d = _date_col(prices)
    dates = prices[d].astype(str).str[:10].tolist()
    if target not in dates:
        return nan, nan
    i = dates.index(target)
    close = float(prices["close"].iloc[i])
    prev = float(prices["close"].iloc[i - 1]) if i > 0 else nan
    return prev, close


def session_calls(
    preds: pd.DataFrame,
    target_date: str,
    prices: Optional[Callable[[str], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Every call made for `target_date` next to what the stock really did that session.

    The result follows the audit's rules: BUY is right when the close is higher than the
    previous close, SELL when it is flat or lower, HOLD when the move is within 0.75%.
    `prices(symbol)` returns that stock's price history and fills the two close columns.
    Rows come right first, then wrong, then pending, each sorted by stock.
    """
    cols = [
        "Stock",
        "Call",
        "Chance of UP",
        "Previous close",
        "Close",
        "Actual move",
        "Went",
        "Result",
    ]
    preds = dedupe_predictions(preds)
    if preds.empty:
        return pd.DataFrame(columns=cols)
    g = preds[preds["target_date"].astype(str) == str(target_date)]
    rows = []
    for _, r in g.iterrows():
        sym = str(r["symbol"])
        outcome = r.get("realized_outcome")
        known = outcome is not None and not pd.isna(outcome)
        nan = float("nan")
        prev, close = _closes_around(prices(sym), str(target_date)) if prices else (nan, nan)
        move = float(outcome) if known else float("nan")
        if known:
            went = "Up" if move > 0 else "Down" if move < 0 else "Flat"
            result = "Right" if bool(r.get("is_correct")) else "Wrong"
        else:
            went, result = "Not closed yet", "Pending"
        rows.append(
            {
                "Stock": sym,
                "Call": str(r.get("signal", "")).upper(),
                "Chance of UP": float(r["up_probability"]),
                "Previous close": prev,
                "Close": close if known else float("nan"),
                "Actual move": move * 100.0 if known else float("nan"),
                "Went": went,
                "Result": result,
            }
        )
    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out
    order = {"Right": 0, "Wrong": 1, "Pending": 2}
    out["_o"] = out["Result"].map(order)
    return out.sort_values(["_o", "Stock"]).drop(columns="_o").reset_index(drop=True)


def read_pipeline_runs(log_file: Path, limit: int = 20) -> list[dict[str, Any]]:
    """Newest-first entries from data/pipeline.log (one JSON run report per line)."""
    if not log_file.exists():
        return []
    runs: list[dict[str, Any]] = []
    for line in log_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and "started_at" in entry:
            runs.append(entry)
    runs.sort(key=lambda r: str(r.get("started_at", "")), reverse=True)
    return runs[:limit]


def latest_live_run(runs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Most recent run that was not a dry run."""
    return next((r for r in runs if not r.get("dry_run")), None)


def stale_symbols(snapshot: pd.DataFrame) -> list[str]:
    """Symbols whose last price is older than the newest price in the universe, or missing."""
    if snapshot.empty:
        return []
    newest = snapshot["last_date"].dropna().max() if snapshot["last_date"].notna().any() else None
    out = []
    for _, r in snapshot.iterrows():
        if (
            r["last_date"] is None
            or pd.isna(r["last_date"])
            or (newest and r["last_date"] < newest)
        ):
            out.append(str(r["symbol"]))
    return out


def load_news(
    paths: dict[str, Path], symbol: Optional[str] = None, limit: int = 20
) -> pd.DataFrame:
    """Processed news signals (headline, source, sentiment), newest first."""
    f = paths["processed_news"] / "news_signals.parquet"
    if not f.exists():
        return pd.DataFrame()
    try:
        df = read_parquet(f)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    if symbol:

        def mentions(row: pd.Series) -> bool:
            syms = row.get("matched_symbols")
            in_list = syms is not None and symbol in list(syms)
            return bool(row.get("primary_symbol") == symbol or in_list)

        df = df[df.apply(mentions, axis=1)]
    sort_key = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df.assign(_ts=sort_key).sort_values("_ts", ascending=False).drop(columns="_ts")
    return df.head(limit).reset_index(drop=True)


def load_macro(paths: dict[str, Path]) -> pd.DataFrame:
    """Daily macro table (USD/PKR, Brent, KSE-100, SBP rates), oldest first."""
    f = paths["processed_macro"] / "macro_daily.parquet"
    if not f.exists():
        return pd.DataFrame()
    try:
        df = read_parquet(f)
    except Exception:
        return pd.DataFrame()
    if df.empty or "session_date" not in df.columns:
        return df
    return df.sort_values("session_date").reset_index(drop=True)


def sector_moves(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Average 1-day move per sector, strongest first."""
    if snapshot.empty or snapshot["chg_1d"].isna().all():
        return pd.DataFrame(columns=["sector", "chg_1d", "stocks"])
    g = (
        snapshot.dropna(subset=["chg_1d"])
        .groupby("sector")
        .agg(chg_1d=("chg_1d", "mean"), stocks=("symbol", "count"))
        .reset_index()
    )
    return g.sort_values("chg_1d", ascending=False).reset_index(drop=True)


def format_pkt(iso_ts: Any, fmt: str = "%a %d %b · %H:%M PKT") -> str:
    """Format an ISO timestamp in Pakistan time (naive timestamps are taken as UTC)."""
    try:
        ts = pd.Timestamp(iso_ts)
    except (ValueError, TypeError):
        return str(iso_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("Asia/Karachi").strftime(fmt)


def latest_steps(
    runs: list[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    """Newest full live run, with each step replaced by any later rerun of that step.

    Returns (full_run, tasks); each task gains `rerun: True` when it comes from a rerun.
    """
    live = [r for r in runs if not r.get("dry_run")]  # runs are newest first
    base_idx = next((i for i, r in enumerate(live) if not r.get("partial")), None)
    if base_idx is None:
        return (live[0], list(live[0].get("tasks", []))) if live else (None, [])
    base = live[base_idx]
    tasks = {t.get("step"): dict(t) for t in base.get("tasks", [])}
    for rerun in reversed(live[:base_idx]):  # oldest rerun first, newest wins
        for t in rerun.get("tasks", []):
            tasks[t.get("step")] = {**t, "rerun": True, "rerun_at": rerun.get("started_at")}
    order = [t.get("step") for t in base.get("tasks", [])]
    order += [s for s in tasks if s not in order]
    return base, [tasks[s] for s in order]
