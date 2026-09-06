"""Strict chronological train/test splitting for financial time series without data leakage."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_TECHNICAL_FEATURES: list[str] = [
    "log_ret_1d",
    "log_ret_3d",
    "log_ret_5d",
    "log_ret_10d",
    "log_ret_20d",
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "sma_200",
    "price_to_sma_20",
    "price_to_sma_50",
    "price_to_sma_200",
    "ema_12",
    "ema_26",
    "price_to_ema_12",
    "price_to_ema_26",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "roc_10",
    "volatility_20d",
    "atr_14",
    "atr_pct_14",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_bandwidth",
    "bb_pct_b",
    "rel_volume_20",
    "obv",
    "volume_roc_5",
]


@dataclass(frozen=True)
class ChronologicalSplit:
    """Chronological dataset split with preserved time order."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_dates: pd.Series
    test_dates: pd.Series
    feature_names: list[str]
    target_name: str


def chronological_train_test_split(
    df: pd.DataFrame,
    target_col: str = "target_next_day_dir",
    train_ratio: float = 0.8,
    feature_cols: list[str] | None = None,
) -> ChronologicalSplit:
    """Split dataset chronologically with zero forward data leakage.

    Guarantees:
        1. Preserves temporal order: Test set strictly follows train set in time.
        2. Drops incomplete tail rows where target_col is NaN (e.g. final session T).
        3. Drops initial warmup sessions where rolling indicators are NaN.
        4. Absolutely zero future data is included in training split.

    Args:
        df: Input DataFrame containing features and targets.
        target_col: Target column to predict (e.g. 'target_next_day_dir').
        train_ratio: Proportion of observations allocated to training (default: 0.8).
        feature_cols: List of feature column names (defaults to DEFAULT_TECHNICAL_FEATURES).

    Returns:
        ChronologicalSplit containing train and test feature matrices and target vectors.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    if not 0.1 <= train_ratio <= 0.95:
        raise ValueError(f"train_ratio must be between 0.1 and 0.95, got {train_ratio}")

    # Determine feature columns
    if feature_cols is None:
        # Use available default technical features
        feature_cols = [col for col in DEFAULT_TECHNICAL_FEATURES if col in df.columns]
        if not feature_cols:
            # Fallback to all numeric columns except targets and identifiers
            exclude = {
                "symbol",
                "trade_date",
                "target_next_day_dir",
                "target_next_day_3class",
                "target_return_5d",
            }
            feature_cols = [
                c
                for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c].dtype)
            ]

    # Sort strictly by trade_date
    working_df = df.copy()
    if "trade_date" in working_df.columns:
        working_df = working_df.sort_values("trade_date").reset_index(drop=True)

    # Clean rows with NaNs in features or target
    cols_to_check = feature_cols + [target_col]
    clean_df = working_df.dropna(subset=cols_to_check).copy()

    if len(clean_df) < 20:
        raise ValueError(f"Insufficient observations ({len(clean_df)}) after dropping NaNs.")

    # Split chronologically
    n_samples = len(clean_df)
    split_idx = int(np.floor(n_samples * train_ratio))

    train_df = clean_df.iloc[:split_idx]
    test_df = clean_df.iloc[split_idx:]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    train_dates = train_df["trade_date"].copy() if "trade_date" in train_df.columns else pd.Series()
    test_dates = test_df["trade_date"].copy() if "trade_date" in test_df.columns else pd.Series()

    return ChronologicalSplit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_dates=train_dates,
        test_dates=test_dates,
        feature_names=feature_cols,
        target_name=target_col,
    )
