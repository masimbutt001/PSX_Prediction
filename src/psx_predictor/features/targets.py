"""Supervised learning prediction targets for classification and regression models."""

import numpy as np
import pandas as pd


def compute_binary_direction_target(series: pd.Series) -> pd.Series:
    """Compute next-day binary price direction target.

    Formula:
        y_t = 1.0 if P_{t+1} > P_t else 0.0
        y_T = NaN (the final session has no known future price)

    Args:
        series: Historical price series (e.g. adjusted_close).

    Returns:
        pd.Series containing binary direction labels with NaN on the last row.
    """
    future_price = series.shift(-1)
    target = (future_price > series).astype(float)
    target[future_price.isna()] = np.nan
    return target.rename("target_next_day_dir")


def compute_threshold_3class_target(
    series: pd.Series,
    threshold: float = 0.0075,
) -> pd.Series:
    """Compute noise-filtered 3-class direction target.

    Filters bid-ask spread and transaction cost friction (default threshold: 0.75%):
        y_t = 1.0 (UP) if (P_{t+1} / P_t - 1) > +threshold
        y_t = -1.0 (DOWN) if (P_{t+1} / P_t - 1) < -threshold
        y_t = 0.0 (NEUTRAL) if abs(P_{t+1} / P_t - 1) <= threshold
        y_T = NaN

    Args:
        series: Historical price series.
        threshold: Minimum percentage price movement to trigger UP/DOWN (default: 0.0075).

    Returns:
        pd.Series containing {-1.0, 0.0, 1.0} labels with NaN on the last row.
    """
    future_price = series.shift(-1)
    future_return = ((future_price - series) / series).round(8)

    target = pd.Series(0.0, index=series.index, dtype=float)
    target[future_return > threshold] = 1.0
    target[future_return < -threshold] = -1.0
    target[future_price.isna()] = np.nan

    return target.rename("target_next_day_3class")


def compute_future_return_target(
    series: pd.Series,
    horizon: int = 5,
) -> pd.Series:
    """Compute cumulative forward return over a multi-session horizon.

    Formula:
        R_{t, t+h} = (P_{t+h} / P_t) - 1.0
        Last h sessions must be NaN.

    Args:
        series: Historical price series.
        horizon: Forward lookahead horizon in trading sessions (default: 5).

    Returns:
        pd.Series containing future return values with NaN on the final h rows.
    """
    future_price = series.shift(-horizon)
    target = (future_price / series) - 1.0
    target[future_price.isna()] = np.nan
    return target.rename(f"target_return_{horizon}d")


def compute_all_prediction_targets(
    df: pd.DataFrame,
    threshold: float = 0.0075,
    horizon: int = 5,
) -> pd.DataFrame:
    """Generate the full set of supervised learning targets aligned with DataFrame index.

    Ensures strict chronological sorting so shift(-1) and shift(-horizon) accurately
    index subsequent temporal sessions.

    Args:
        df: DataFrame with 'adjusted_close' or 'close' column and 'trade_date'.
        threshold: 3-class breakout threshold (default: 0.0075).
        horizon: Multi-session return horizon (default: 5).

    Returns:
        DataFrame containing target columns:
            ['target_next_day_dir', 'target_next_day_3class', f'target_return_{horizon}d'].
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "target_next_day_dir",
                "target_next_day_3class",
                f"target_return_{horizon}d",
            ]
        )

    # Use adjusted_close where available
    price = df["adjusted_close"] if "adjusted_close" in df.columns else df["close"]

    t_dir = compute_binary_direction_target(price)
    t_3class = compute_threshold_3class_target(price, threshold=threshold)
    t_return = compute_future_return_target(price, horizon=horizon)

    targets_df = pd.concat([t_dir, t_3class, t_return], axis=1)
    return targets_df
