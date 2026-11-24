"""Detect splits and bonus issues that price sources leave out, and back-adjust for them.

SCSTrade and the PSX Data Portal report raw traded prices with `split_ratio = 1.0` and
`adjusted_close = close`, so a 1:5 split looks like an 80% crash. A PSX stock cannot
move more than about 10% in one session (circuit breakers), so an opening gap of more
than `MAX_SESSION_GAP` from the previous close can only be a corporate action. These
helpers record such gaps as `split_ratio` (shares after / shares before) so the
standard backward adjustment in `compute_continuous_adjusted_close` handles them.

Limit: small bonus issues (for example 10%) leave gaps inside the normal daily range
and are not detected.
"""

from typing import Any

import numpy as np
import pandas as pd

MAX_SESSION_GAP = 0.30
# Usual PSX split and bonus ratios (shares after / shares before). A detected ratio
# snaps to the nearest one within SNAP_TOLERANCE; the gap also holds that session's
# real price move, so the raw ratio is rarely exact.
COMMON_RATIOS: tuple[float, ...] = (
    1.1,
    1.125,
    1.15,
    1.2,
    1.25,
    1.3,
    4 / 3,
    1.4,
    1.5,
    1.6,
    1.75,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    10.0,
)
SNAP_TOLERANCE = 0.06


def _snap_ratio(raw: float) -> float:
    candidates = [*COMMON_RATIOS, *(1.0 / r for r in COMMON_RATIOS)]
    best = min(candidates, key=lambda r: abs(r / raw - 1.0))
    return best if abs(best / raw - 1.0) <= SNAP_TOLERANCE else raw


def infer_split_ratios(
    df: pd.DataFrame, max_gap: float = MAX_SESSION_GAP
) -> tuple[pd.DataFrame, list[tuple[Any, float]]]:
    """Fill `split_ratio` for sessions whose open gaps beyond what PSX allows.

    The ratio is previous close / open (5.0 for a 1:5 split, 1.25 for a 25% bonus),
    snapped to a clean fraction when close to one. Sessions that already carry a
    split ratio from the source are left alone.

    Args:
        df: Chronologically sorted daily prices with 'trade_date', 'open', 'close'.
        max_gap: Smallest opening gap (as a fraction) treated as a corporate action.

    Returns:
        (copy of df with `split_ratio` filled, list of (trade_date, ratio) found).
    """
    out = df.copy()
    if out.empty or "close" not in out.columns:
        return out, []
    if "split_ratio" not in out.columns:
        out["split_ratio"] = 1.0
    out["split_ratio"] = out["split_ratio"].fillna(1.0).astype(float)

    open_px = out["open"] if "open" in out.columns else out["close"]
    open_px = open_px.fillna(out["close"]).astype(float)
    prev_close = out["close"].astype(float).shift(1)
    gap = open_px / prev_close
    known = (out["split_ratio"] - 1.0).abs() > 1e-4
    hits = out.index[((gap < 1.0 - max_gap) | (gap > 1.0 / (1.0 - max_gap))) & ~known & gap.notna()]
    events: list[tuple[Any, float]] = []
    for idx in hits:
        raw = float(prev_close.loc[idx] / open_px.loc[idx])
        if not np.isfinite(raw) or raw <= 0:
            continue
        ratio = _snap_ratio(raw)
        out.at[idx, "split_ratio"] = ratio
        events.append((out.at[idx, "trade_date"], ratio))
    return out, events


def apply_corporate_actions(
    df: pd.DataFrame, max_gap: float = MAX_SESSION_GAP
) -> tuple[pd.DataFrame, list[tuple[Any, float]]]:
    """Infer missing splits, then recompute `adjusted_close` for splits and dividends.

    Raw open/high/low/close stay as traded; only `split_ratio` and `adjusted_close`
    change. Safe to run repeatedly: already-recorded ratios are not inferred twice.
    """
    from psx_predictor.processing.pipeline import compute_continuous_adjusted_close

    if df.empty or "close" not in df.columns:
        return df.copy(), []
    work = df.copy()
    if "trade_date" in work.columns and not work["trade_date"].is_monotonic_increasing:
        work = work.sort_values("trade_date").reset_index(drop=True)
    work, events = infer_split_ratios(work, max_gap=max_gap)
    has_splits = ((work["split_ratio"] - 1.0).abs() > 1e-4).any()
    has_divs = "dividend_amount" in work.columns and (work["dividend_amount"].fillna(0) > 0).any()
    if has_splits or has_divs:
        work["adjusted_close"] = compute_continuous_adjusted_close(work)
    elif "adjusted_close" not in work.columns:
        work["adjusted_close"] = work["close"]
    return work, events


def adjusted_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Scale open/high/low/close onto the `adjusted_close` basis; volume inversely.

    Use this wherever bars are compared across time (indicators, backtests) so that a
    split does not show up as a crash. Rows without a usable factor keep raw values.
    """
    out = df.copy()
    if "adjusted_close" not in out.columns or "close" not in out.columns:
        return out
    close = out["close"].astype(float)
    factor = (out["adjusted_close"].astype(float) / close.replace(0.0, np.nan)).fillna(1.0)
    if np.allclose(factor.to_numpy(), 1.0):
        return out
    for col in ("open", "high", "low"):
        if col in out.columns:
            out[col] = out[col].astype(float) * factor
    out["close"] = close * factor
    if "volume" in out.columns:
        out["volume"] = out["volume"].astype(float) / factor
    return out
