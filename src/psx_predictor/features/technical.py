"""Deterministic, vectorized technical indicator computations without look-ahead bias."""

import numpy as np
import pandas as pd


def compute_log_returns(
    series: pd.Series, windows: tuple[int, ...] = (1, 3, 5, 10, 20)
) -> dict[str, pd.Series]:
    """Compute logarithmic returns over various lookback horizons.

    Formula: r_k = ln(P_t / P_{t-k})

    Args:
        series: Price series (e.g. adjusted_close).
        windows: Tuple of integer lookback horizons in trading days.

    Returns:
        Dictionary mapping column name to computed log return Series.
    """
    results: dict[str, pd.Series] = {}
    for w in windows:
        shifted = series.shift(w)
        # Avoid division by zero or log of non-positive
        ratio = series / shifted
        # Replace non-positive ratios with NaN before log
        valid_ratio = ratio.where(ratio > 0, np.nan)
        results[f"return_{w}d"] = np.log(valid_ratio)
    return results


def compute_moving_averages(
    series: pd.Series,
    sma_windows: tuple[int, ...] = (5, 10, 20, 50, 200),
    ema_windows: tuple[int, ...] = (12, 26),
) -> dict[str, pd.Series]:
    """Compute Simple and Exponential Moving Averages and normalized price distances.

    Args:
        series: Price series (e.g. adjusted_close).
        sma_windows: Simple moving average window lengths.
        ema_windows: Exponential moving average span lengths.

    Returns:
        Dictionary mapping indicator name to computed Series.
    """
    results: dict[str, pd.Series] = {}

    for w in sma_windows:
        sma = series.rolling(window=w, min_periods=w).mean()
        results[f"sma_{w}"] = sma
        # Normalized distance: (Price - SMA) / SMA
        results[f"dist_sma_{w}"] = (series - sma) / sma

    for span in ema_windows:
        ema = series.ewm(span=span, adjust=False).mean()
        results[f"ema_{span}"] = ema

    return results


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index using Wilder's smoothing method.

    Args:
        series: Price series.
        window: Lookback period (default: 14).

    Returns:
        RSI Series bounded in [0, 100].
    """
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's exponential smoothing: alpha = 1 / window
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Where loss is 0 and gain > 0, RSI is 100
    zero_loss_mask = (avg_loss == 0.0) & (avg_gain > 0.0)
    rsi = rsi.mask(zero_loss_mask, 100.0)

    # Where both gain and loss are 0, RSI is 50
    flat_mask = (avg_loss == 0.0) & (avg_gain == 0.0)
    rsi = rsi.mask(flat_mask, 50.0)

    return rsi.rename(f"rsi_{window}")


def compute_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> dict[str, pd.Series]:
    """Compute Moving Average Convergence Divergence (MACD).

    Args:
        series: Price series.
        fast_period: Fast EMA span (default: 12).
        slow_period: Slow EMA span (default: 26).
        signal_period: Signal line EMA span (default: 9).

    Returns:
        Dictionary with 'macd_line', 'macd_signal', 'macd_hist'.
    """
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    return {
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
    }


def compute_rate_of_change(series: pd.Series, window: int = 10) -> pd.Series:
    """Compute percentage Rate of Change (ROC).

    Formula: ROC_k = ((P_t - P_{t-k}) / P_{t-k}) * 100

    Args:
        series: Price series.
        window: Lookback period in days (default: 10).

    Returns:
        ROC Series.
    """
    shifted = series.shift(window)
    roc = ((series - shifted) / shifted.replace(0.0, np.nan)) * 100.0
    return roc.rename(f"roc_{window}")


def compute_volatility(returns_1d: pd.Series, window: int = 20) -> pd.Series:
    """Compute annualized rolling standard deviation of 1-day log returns.

    Assumes 252 trading sessions per annum for Pakistan Stock Exchange.

    Args:
        returns_1d: 1-day log returns Series.
        window: Rolling window size in sessions (default: 20).

    Returns:
        Annualized volatility Series.
    """
    vol = returns_1d.rolling(window=window, min_periods=window).std() * np.sqrt(252)
    return vol.rename(f"volatility_{window}d")


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> dict[str, pd.Series]:
    """Compute Average True Range (ATR) and normalized ATR percentage.

    Args:
        high: High price Series.
        low: Low price Series.
        close: Close price Series.
        window: Smoothing period (default: 14).

    Returns:
        Dictionary with 'atr_14' and 'atr_percent_14'.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    atr_pct = (atr / close) * 100.0

    return {
        f"atr_{window}": atr,
        f"atr_percent_{window}": atr_pct,
    }


def compute_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> dict[str, pd.Series]:
    """Compute Bollinger Bands (Upper, Lower, Bandwidth, and %B).

    Args:
        series: Price series.
        window: Rolling window (default: 20).
        num_std: Multiplier for standard deviation (default: 2.0).

    Returns:
        Dictionary with 'bb_upper', 'bb_middle', 'bb_lower', 'bb_bandwidth', 'bb_percent_b'.
    """
    middle = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()

    upper = middle + (num_std * rolling_std)
    lower = middle - (num_std * rolling_std)
    bandwidth = (upper - lower) / middle.replace(0.0, np.nan)
    percent_b = (series - lower) / (upper - lower).replace(0.0, np.nan)

    return {
        "bb_middle": middle,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_bandwidth": bandwidth,
        "bb_percent_b": percent_b,
    }


def compute_volume_dynamics(
    volume: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> dict[str, pd.Series]:
    """Compute relative volume, On-Balance Volume (OBV), and volume rate of change.

    Args:
        volume: Traded volume Series.
        close: Closing price Series.
        window: Relative volume smoothing window (default: 20).

    Returns:
        Dictionary with 'volume_relative_20', 'obv', 'volume_roc_5'.
    """
    vol_sma = volume.rolling(window=window, min_periods=window).mean()
    vol_relative = volume / vol_sma.replace(0.0, np.nan)

    # On-Balance Volume (OBV)
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()

    # 5-day volume rate of change
    vol_shift5 = volume.shift(5)
    vol_roc_5 = ((volume - vol_shift5) / vol_shift5.replace(0.0, np.nan)) * 100.0

    return {
        f"volume_relative_{window}": vol_relative,
        "obv": obv,
        "volume_roc_5": vol_roc_5,
    }


def compute_all_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the full suite of technical indicators without look-ahead bias.

    Preserves original columns and appends all derived technical indicator features.

    Args:
        df: Chronologically ordered OHLCV DataFrame with columns:
            ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume'].

    Returns:
        Enriched DataFrame containing all computed technical indicators.
    """
    if df.empty:
        return df.copy()

    # Ensure chronological sort
    work_df = df.copy()
    if "trade_date" in work_df.columns and not work_df["trade_date"].is_monotonic_increasing:
        work_df = work_df.sort_values(by="trade_date").reset_index(drop=True)

    # Use adjusted_close where available for returns, trend, and momentum
    adj_price = (
        work_df["adjusted_close"] if "adjusted_close" in work_df.columns else work_df["close"]
    )
    unadj_close = work_df["close"]
    high = work_df["high"]
    low = work_df["low"]
    volume = work_df["volume"].astype(float)

    feature_dict: dict[str, pd.Series] = {}

    # 1. Log Returns
    returns = compute_log_returns(adj_price)
    feature_dict.update(returns)

    # 2. Moving Averages & Normalized Distance
    ma_dict = compute_moving_averages(adj_price)
    feature_dict.update(ma_dict)

    # 3. Momentum
    feature_dict["rsi_14"] = compute_rsi(adj_price, window=14)
    feature_dict.update(compute_macd(adj_price))
    feature_dict["roc_10"] = compute_rate_of_change(adj_price, window=10)

    # 4. Volatility
    r_1d = returns.get("return_1d", pd.Series(np.nan, index=work_df.index))
    feature_dict["volatility_20d"] = compute_volatility(r_1d, window=20)
    feature_dict.update(compute_atr(high, low, unadj_close, window=14))
    feature_dict.update(compute_bollinger_bands(adj_price, window=20, num_std=2.0))

    # 5. Volume Dynamics
    feature_dict.update(compute_volume_dynamics(volume, adj_price, window=20))

    # Assemble final DataFrame
    features_df = pd.DataFrame(feature_dict, index=work_df.index)
    result_df = pd.concat([work_df, features_df], axis=1)

    return result_df
