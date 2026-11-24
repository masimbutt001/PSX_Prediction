"""Deterministic, vectorized technical indicator computations without look-ahead bias."""

import numpy as np
import pandas as pd

from psx_predictor.processing.corporate_actions import adjusted_ohlcv


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
        results[f"log_ret_{w}d"] = np.log(valid_ratio)
    return results


def compute_moving_averages(
    series: pd.Series,
    sma_windows: tuple[int, ...] = (5, 10, 20, 50, 200),
    ema_windows: tuple[int, ...] = (12, 20, 26, 50),
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
        results[f"price_to_sma_{w}"] = (series - sma) / sma

    for span in ema_windows:
        ema = series.ewm(span=span, adjust=False).mean()
        results[f"ema_{span}"] = ema
        # Normalized distance: (Price - EMA) / EMA
        results[f"price_to_ema_{span}"] = (series - ema) / ema.replace(0.0, np.nan)

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
        Dictionary with 'atr_14' and 'atr_pct_14'.
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
        f"atr_pct_{window}": atr_pct,
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
        Dictionary with 'bb_upper', 'bb_middle', 'bb_lower', 'bb_bandwidth', 'bb_pct_b'.
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
        "bb_pct_b": percent_b,
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
        Dictionary with 'rel_volume_20', 'obv', 'volume_roc_5'.
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
        f"rel_volume_{window}": vol_relative,
        "obv": obv,
        "volume_roc_5": vol_roc_5,
    }


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range: the largest of today's range and the gaps from yesterday's close."""
    prev_close = close.shift(1)
    return pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )


def _wilder(series: pd.Series, window: int) -> pd.Series:
    """Wilder smoothing (an EMA with alpha = 1 / window)."""
    return series.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()


def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> dict[str, pd.Series]:
    """Average Directional Index with the +DI / -DI lines (Wilder).

    ADX measures trend strength regardless of direction: above 25 is a trend,
    below 20 is a choppy market.
    """
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = _wilder(_true_range(high, low, close), window)
    safe_atr = atr.replace(0.0, np.nan)
    plus_di = (100.0 * _wilder(plus_dm, window) / safe_atr).fillna(0.0).where(atr.notna())
    minus_di = (100.0 * _wilder(minus_dm, window) / safe_atr).fillna(0.0).where(atr.notna())
    di_sum = plus_di + minus_di
    dx = (100.0 * (plus_di - minus_di).abs() / di_sum.replace(0.0, np.nan)).fillna(0.0)
    dx = dx.where(atr.notna())
    adx = _wilder(dx, window)
    return {
        f"adx_{window}": adx,
        f"plus_di_{window}": plus_di,
        f"minus_di_{window}": minus_di,
    }


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> dict[str, pd.Series]:
    """Slow Stochastic oscillator (14, 3, 3): %K and its signal line %D, in [0, 100]."""
    lowest = low.rolling(window=window, min_periods=window).min()
    highest = high.rolling(window=window, min_periods=window).max()
    rng = highest - lowest
    raw_k = (100.0 * (close - lowest) / rng.replace(0.0, np.nan)).where(rng != 0.0, 50.0)
    raw_k = raw_k.where(rng.notna())
    k = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(window=smooth_d, min_periods=smooth_d).mean()
    return {f"stoch_k_{window}": k, f"stoch_d_{window}": d}


def compute_stoch_rsi(
    series: pd.Series,
    rsi_window: int = 14,
    stoch_window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> dict[str, pd.Series]:
    """Stochastic RSI (14, 14, 3, 3): where RSI sits in its own recent range, in [0, 100]."""
    rsi = compute_rsi(series, window=rsi_window)
    lowest = rsi.rolling(window=stoch_window, min_periods=stoch_window).min()
    highest = rsi.rolling(window=stoch_window, min_periods=stoch_window).max()
    rng = highest - lowest
    raw = (100.0 * (rsi - lowest) / rng.replace(0.0, np.nan)).where(rng != 0.0, 50.0)
    raw = raw.where(rng.notna())
    k = raw.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(window=smooth_d, min_periods=smooth_d).mean()
    return {f"stochrsi_k_{rsi_window}": k, f"stochrsi_d_{rsi_window}": d}


def compute_ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
    displacement: int = 26,
) -> dict[str, pd.Series]:
    """Ichimoku Cloud lines as they stand on each session, without look-ahead.

    The cloud plotted over today was computed `displacement` sessions ago, so the
    spans are shifted forward. The lagging span (Chikou) uses future prices when
    aligned to today and is deliberately left out.
    """

    def midpoint(n: int) -> pd.Series:
        hi = high.rolling(window=n, min_periods=n).max()
        lo = low.rolling(window=n, min_periods=n).min()
        return (hi + lo) / 2.0

    tenkan_line = midpoint(tenkan)
    kijun_line = midpoint(kijun)
    span_a = ((tenkan_line + kijun_line) / 2.0).shift(displacement)
    span_b = midpoint(senkou_b).shift(displacement)

    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1, skipna=False)
    cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1, skipna=False)
    safe_close = close.replace(0.0, np.nan)
    # Signed distance from the cloud: positive above it, negative below, 0 inside.
    above = (close - cloud_top).clip(lower=0.0)
    below = (close - cloud_bottom).clip(upper=0.0)
    price_to_cloud = (above + below) / safe_close

    return {
        f"tenkan_{tenkan}": tenkan_line,
        f"kijun_{kijun}": kijun_line,
        "senkou_a": span_a,
        "senkou_b": span_b,
        f"price_to_tenkan_{tenkan}": (close - tenkan_line) / tenkan_line.replace(0.0, np.nan),
        f"price_to_kijun_{kijun}": (close - kijun_line) / kijun_line.replace(0.0, np.nan),
        "tenkan_kijun_pct": (tenkan_line - kijun_line) / kijun_line.replace(0.0, np.nan),
        "price_to_cloud": price_to_cloud,
        "cloud_thickness_pct": (span_a - span_b) / safe_close,
    }


def compute_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
    trade_dates: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Volume-weighted average price on daily bars.

    - `vwap_20`: rolling 20-session VWAP of the typical price.
    - `avwap_ytd`: VWAP anchored to the first session of each calendar year.
    """
    typical = (high + low + close) / 3.0
    pv = typical * volume
    vol_sum = volume.rolling(window=window, min_periods=window).sum()
    vwap = pv.rolling(window=window, min_periods=window).sum() / vol_sum.replace(0.0, np.nan)
    vwap = vwap.fillna(typical.rolling(window=window, min_periods=window).mean())

    if trade_dates is not None:
        years = pd.to_datetime(trade_dates).dt.year.to_numpy()
    else:
        years = np.zeros(len(close), dtype=int)
    groups = pd.Series(years, index=close.index)
    cum_pv = pv.groupby(groups).cumsum()
    cum_vol = volume.groupby(groups).cumsum()
    avwap = (cum_pv / cum_vol.replace(0.0, np.nan)).fillna(typical)

    return {
        f"vwap_{window}": vwap,
        f"price_to_vwap_{window}": (close - vwap) / vwap.replace(0.0, np.nan),
        "avwap_ytd": avwap,
        "price_to_avwap_ytd": (close - avwap) / avwap.replace(0.0, np.nan),
    }


def compute_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 10,
    multiplier: float = 3.0,
) -> dict[str, pd.Series]:
    """Supertrend (10, 3): an ATR band that flips with the trend; doubles as a trailing stop.

    `supertrend_dir` is +1 in an uptrend (line below price) and -1 in a downtrend.
    """
    atr = _wilder(_true_range(high, low, close), window).to_numpy()
    hl2 = ((high + low) / 2.0).to_numpy()
    c = close.to_numpy(dtype=float)
    n = len(c)
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    line = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    final_upper = np.nan
    final_lower = np.nan
    cur_dir = 1.0
    started = False
    for i in range(n):
        if np.isnan(atr[i]):
            continue
        if not started:
            final_upper, final_lower = upper_basic[i], lower_basic[i]
            cur_dir = 1.0 if c[i] >= hl2[i] else -1.0
            started = True
        else:
            prev_close = c[i - 1]
            if upper_basic[i] < final_upper or prev_close > final_upper:
                final_upper = upper_basic[i]
            if lower_basic[i] > final_lower or prev_close < final_lower:
                final_lower = lower_basic[i]
            if cur_dir > 0 and c[i] < final_lower:
                cur_dir = -1.0
            elif cur_dir < 0 and c[i] > final_upper:
                cur_dir = 1.0
        direction[i] = cur_dir
        line[i] = final_lower if cur_dir > 0 else final_upper

    st_line = pd.Series(line, index=close.index)
    return {
        f"supertrend_{window}_{int(multiplier)}": st_line,
        "supertrend_dir": pd.Series(direction, index=close.index),
        "price_to_supertrend": (close - st_line) / st_line.replace(0.0, np.nan),
    }


def compute_donchian(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> dict[str, pd.Series]:
    """Donchian Channel (20): highest high and lowest low of the last `window` sessions.

    `donchian_pos_20` is where the close sits in the channel (0 = at the low, 1 = at the high).
    """
    upper = high.rolling(window=window, min_periods=window).max()
    lower = low.rolling(window=window, min_periods=window).min()
    rng = upper - lower
    pos = ((close - lower) / rng.replace(0.0, np.nan)).where(rng != 0.0, 0.5).where(rng.notna())
    return {
        f"donchian_upper_{window}": upper,
        f"donchian_lower_{window}": lower,
        f"donchian_mid_{window}": (upper + lower) / 2.0,
        f"donchian_pos_{window}": pos,
        f"donchian_width_pct_{window}": rng / close.replace(0.0, np.nan),
    }


def compute_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Money Flow Index (14): a volume-weighted RSI, in [0, 100]."""
    typical = (high + low + close) / 3.0
    flow = typical * volume
    change = typical.diff()
    pos = flow.where(change > 0, 0.0)
    neg = flow.where(change < 0, 0.0)
    pos_sum = pos.rolling(window=window, min_periods=window).sum()
    neg_sum = neg.rolling(window=window, min_periods=window).sum()
    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    mfi = 100.0 - 100.0 / (1.0 + ratio)
    mfi = mfi.mask((neg_sum == 0.0) & (pos_sum > 0.0), 100.0)
    mfi = mfi.mask((neg_sum == 0.0) & (pos_sum == 0.0), 50.0)
    # The first diff is undefined, so the first full window starts one session later.
    mfi.iloc[:window] = np.nan
    return mfi.rename(f"mfi_{window}")


def compute_cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Chaikin Money Flow (20), in [-1, 1]: positive when buyers are accumulating."""
    rng = (high - low).replace(0.0, np.nan)
    mf_mult = (((close - low) - (high - close)) / rng).fillna(0.0)
    mf_vol = mf_mult * volume
    vol_sum = volume.rolling(window=window, min_periods=window).sum()
    cmf = mf_vol.rolling(window=window, min_periods=window).sum() / vol_sum.replace(0.0, np.nan)
    cmf = cmf.fillna(0.0).where(vol_sum.notna())
    return cmf.rename(f"cmf_{window}")


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
    # Bar-based indicators (ATR, ADX, Stochastic, Ichimoku, ...) use high/low/close
    # scaled onto the adjusted basis, so a split or bonus issue is not read as a crash.
    bars = adjusted_ohlcv(work_df)
    unadj_close = bars["close"]
    high = bars["high"]
    low = bars["low"]
    volume = bars["volume"].astype(float)

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
    r_1d = returns.get("log_ret_1d", pd.Series(np.nan, index=work_df.index))
    feature_dict["volatility_20d"] = compute_volatility(r_1d, window=20)
    feature_dict.update(compute_atr(high, low, unadj_close, window=14))
    feature_dict.update(compute_bollinger_bands(adj_price, window=20, num_std=2.0))

    # 5. Volume Dynamics
    feature_dict.update(compute_volume_dynamics(volume, adj_price, window=20))

    # 6. Trend strength, oscillators, cloud, channels and money flow (on unadjusted OHLC,
    #    like ATR, so highs, lows and closes are on the same scale)
    feature_dict.update(compute_adx(high, low, unadj_close, window=14))
    feature_dict.update(compute_stochastic(high, low, unadj_close, window=14))
    feature_dict.update(compute_stoch_rsi(adj_price))
    feature_dict.update(compute_ichimoku(high, low, unadj_close))
    feature_dict.update(
        compute_vwap(
            high,
            low,
            unadj_close,
            volume,
            window=20,
            trade_dates=work_df["trade_date"] if "trade_date" in work_df.columns else None,
        )
    )
    feature_dict.update(compute_supertrend(high, low, unadj_close, window=10, multiplier=3.0))
    feature_dict.update(compute_donchian(high, low, unadj_close, window=20))
    feature_dict["mfi_14"] = compute_mfi(high, low, unadj_close, volume, window=14)
    feature_dict["cmf_20"] = compute_cmf(high, low, unadj_close, volume, window=20)

    # Assemble final DataFrame
    features_df = pd.DataFrame(feature_dict, index=work_df.index)
    result_df = pd.concat([work_df, features_df], axis=1)

    return result_df
