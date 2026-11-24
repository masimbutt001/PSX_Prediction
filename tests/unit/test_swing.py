"""Tests for the new technical indicators and the rule-based swing strategy backtest."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from psx_predictor.backtesting.swing import (
    SwingStrategyConfig,
    blackout_windows_from_news,
    entry_signals,
    market_uptrend,
    prepare_swing_frame,
    run_swing_backtest,
)
from psx_predictor.features.technical import (
    compute_adx,
    compute_all_technical_features,
    compute_cmf,
    compute_donchian,
    compute_ichimoku,
    compute_mfi,
    compute_stochastic,
    compute_supertrend,
    compute_vwap,
)


def _ohlcv(close: np.ndarray, start: str = "2022-01-03", vol: float = 1_000_000.0) -> pd.DataFrame:
    n = len(close)
    return pd.DataFrame(
        {
            "symbol": "TEST",
            "trade_date": pd.bdate_range(start, periods=n).date,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adjusted_close": close,
            "volume": np.full(n, vol),
        }
    )


def _random_walk(n: int = 600, seed: int = 3, drift: float = 0.0008) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    df = _ohlcv(close)
    df["open"] = close * (1 + rng.normal(0, 0.003, n))
    df["high"] = np.maximum(df["open"], close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    df["low"] = np.minimum(df["open"], close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    df["volume"] = rng.lognormal(13.5, 0.4, n)
    return df


# ---------------------------------------------------------------- indicators


def test_adx_high_in_steady_trend_and_di_points_the_right_way() -> None:
    up = _ohlcv(np.linspace(100, 200, 120))
    out = compute_adx(up["high"], up["low"], up["close"])
    assert out["adx_14"].iloc[-1] > 40
    assert out["plus_di_14"].iloc[-1] > out["minus_di_14"].iloc[-1]
    down = _ohlcv(np.linspace(200, 100, 120))
    out_d = compute_adx(down["high"], down["low"], down["close"])
    assert out_d["minus_di_14"].iloc[-1] > out_d["plus_di_14"].iloc[-1]


def test_stochastic_bounded_and_flat_price_is_neutral() -> None:
    df = _random_walk(200)
    st = compute_stochastic(df["high"], df["low"], df["close"])
    k = st["stoch_k_14"].dropna()
    assert k.between(0, 100).all()
    flat = pd.Series(np.full(40, 50.0))
    st_flat = compute_stochastic(flat, flat, flat)
    assert st_flat["stoch_k_14"].dropna().eq(50.0).all()


def test_ichimoku_has_no_lookahead() -> None:
    """Changing a future bar must not change any value on or before an earlier session."""
    df = _random_walk(200)
    base = compute_ichimoku(df["high"], df["low"], df["close"])
    bumped = df.copy()
    bumped.loc[150:, ["high", "low", "close"]] *= 1.5
    after = compute_ichimoku(bumped["high"], bumped["low"], bumped["close"])
    for key in base:
        pd.testing.assert_series_equal(base[key].iloc[:150], after[key].iloc[:150])


def test_all_new_indicators_are_causal() -> None:
    df = _random_walk(400)
    base = compute_all_technical_features(df)
    bumped = df.copy()
    bumped.loc[300:, ["open", "high", "low", "close", "adjusted_close"]] *= 0.7
    bumped.loc[300:, "volume"] *= 3
    after = compute_all_technical_features(bumped)
    pd.testing.assert_frame_equal(base.iloc[:300], after.iloc[:300])


def test_vwap_constant_price_equals_price_and_resets_each_year() -> None:
    df = _ohlcv(np.full(300, 50.0), start="2023-11-01")
    out = compute_vwap(
        df["high"], df["low"], df["close"], df["volume"], trade_dates=df["trade_date"]
    )
    assert np.allclose(out["vwap_20"].dropna(), 50.0)
    df2 = _ohlcv(np.r_[np.full(30, 10.0), np.full(270, 20.0)], start="2023-11-01")
    out2 = compute_vwap(
        df2["high"], df2["low"], df2["close"], df2["volume"], trade_dates=df2["trade_date"]
    )
    first_2024 = int(np.argmax(pd.to_datetime(df2["trade_date"]).dt.year.to_numpy() == 2024))
    assert out2["avwap_ytd"].iloc[first_2024] == pytest.approx(20.0)


def test_supertrend_follows_trend_and_flips() -> None:
    close = np.r_[np.linspace(100, 150, 80), np.linspace(150, 90, 80)]
    df = _ohlcv(close)
    out = compute_supertrend(df["high"], df["low"], df["close"])
    assert out["supertrend_dir"].iloc[75] == 1
    assert out["supertrend_10_3"].iloc[75] < close[75]
    assert out["supertrend_dir"].iloc[-1] == -1
    assert out["supertrend_10_3"].iloc[-1] > close[-1]


def test_donchian_mfi_cmf_ranges() -> None:
    df = _random_walk(200)
    d = compute_donchian(df["high"], df["low"], df["close"])
    assert d["donchian_pos_20"].dropna().between(0, 1).all()
    assert (d["donchian_upper_20"].dropna() >= d["donchian_lower_20"].dropna()).all()
    mfi = compute_mfi(df["high"], df["low"], df["close"], df["volume"]).dropna()
    assert mfi.between(0, 100).all()
    cmf = compute_cmf(df["high"], df["low"], df["close"], df["volume"]).dropna()
    assert cmf.between(-1, 1).all()
    # Closing at the high every day is pure accumulation.
    at_high = _ohlcv(np.linspace(10, 20, 40))
    at_high["close"] = at_high["high"]
    assert compute_cmf(at_high["high"], at_high["low"], at_high["close"], at_high["volume"]).iloc[
        -1
    ] == pytest.approx(1.0)


# ---------------------------------------------------------------- strategy inputs


def test_swing_backtest_ignores_unadjusted_split() -> None:
    """A 1:5 split in raw prices must not trigger the stop or show as a loss."""
    df = _random_walk(600)
    raw = df.copy()
    cols = ["open", "high", "low", "close", "adjusted_close"]
    raw.loc[:399, cols] = raw.loc[:399, cols] * 5.0
    raw.loc[:399, "volume"] = raw.loc[:399, "volume"] / 5.0
    res_clean = run_swing_backtest(df, "TEST")
    res_raw = run_swing_backtest(raw, "TEST")
    assert any("1 share became 5" in n for n in res_raw.notes)
    assert res_raw.benchmark_metrics.max_drawdown == pytest.approx(
        res_clean.benchmark_metrics.max_drawdown, abs=1e-3
    )
    assert len(res_raw.trades) == len(res_clean.trades)


def test_blackout_windows_from_results_and_dividend_notices() -> None:
    news = pd.DataFrame(
        {
            "published_at": [
                "2026-01-05T10:00:00+05:00",
                "2026-02-01T10:00:00+05:00",
                "2026-03-01T10:00:00+05:00",
            ],
            "headline": ["OGDC board meeting to consider results", "OGDC new well", "PPL dividend"],
            "primary_symbol": ["OGDC", "OGDC", "PPL"],
            "matched_symbols": [["OGDC"], ["OGDC"], ["PPL"]],
            "event_category": ["OTHER", "OTHER", "DIVIDEND"],
        }
    )
    w = blackout_windows_from_news(news, "OGDC", days=7)
    assert w == [(dt.date(2026, 1, 5), dt.date(2026, 1, 12))]
    assert blackout_windows_from_news(news, "PPL")[0][0] == dt.date(2026, 3, 1)
    assert blackout_windows_from_news(pd.DataFrame(), "OGDC") == []


def test_market_uptrend_needs_real_history() -> None:
    dates = pd.bdate_range("2026-01-01", periods=120)
    flat = pd.DataFrame({"session_date": dates, "kse100_index": 100_000.0})
    assert market_uptrend(flat) is None  # back-filled constant is not history
    rising = pd.DataFrame({"session_date": dates, "kse100_index": np.linspace(1e5, 1.2e5, 120)})
    up = market_uptrend(rising)
    assert up is not None and bool(up.iloc[-1])


# ---------------------------------------------------------------- strategy and backtest


def test_entry_signals_need_every_rule() -> None:
    df = prepare_swing_frame(_random_walk(600))
    sig = entry_signals(df)
    assert (sig["entry"] <= (sig["trend"] & sig["pullback"] & sig["trigger"] & sig["liquid"])).all()
    strict = entry_signals(df, SwingStrategyConfig(adx_min=99))
    assert not strict["entry"].any()


def _scripted_trade(path_after_signal: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """A steady uptrend, one textbook pullback-and-bounce, then the given OHLC bars."""
    n = 260
    close = list(100 * np.exp(np.linspace(0, 0.6, n)))
    df = _ohlcv(np.array(close))
    rows = df.to_dict("records")
    last_date = rows[-1]["trade_date"]
    for i, (o, h, lo, c) in enumerate(path_after_signal, start=1):
        rows.append(
            {
                **rows[-1],
                "trade_date": last_date + dt.timedelta(days=i),
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "adjusted_close": c,
            }
        )
    return pd.DataFrame(rows)


def _force_entry_on(df: pd.DataFrame, idx: int, monkeypatch: pytest.MonkeyPatch) -> None:
    import psx_predictor.backtesting.swing as swing

    real = swing.entry_signals

    def fake(frame: pd.DataFrame, config: SwingStrategyConfig | None = None) -> pd.DataFrame:
        out = real(frame, config)
        out["entry"] = False
        out.loc[idx, "entry"] = True
        return out

    monkeypatch.setattr(swing, "entry_signals", fake)


def test_stop_loss_limits_loss_to_about_one_r(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _scripted_trade([(182.0, 183.0, 181.0, 182.0), (182.0, 182.5, 150.0, 151.0)])
    sig_idx = len(df) - 3  # last uptrend bar
    _force_entry_on(df, sig_idx, monkeypatch)
    no_rsi_exit = SwingStrategyConfig(rsi_exit=101.0)
    res = run_swing_backtest(df, "TEST", config=no_rsi_exit, initial_cash=1_000_000.0)
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.exit_reason == "Stop loss"
    assert t.entry_price == pytest.approx(182.0)
    assert -1.2 < t.r_multiple < -0.95  # one R plus costs
    # 1% risk sizing: loss is about 1% of capital.
    assert res.metrics.total_return == pytest.approx(-0.01, abs=0.003)
    assert res.metrics.win_rate == 0.0 and res.metrics.total_trades == 1


def test_partial_take_profit_then_trailing_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    bars = [(182.0, 183.0, 181.5, 182.5)]
    bars += [(200.0, 215.0, 199.0, 214.0)]  # hits the 2R target
    bars += [(214.0, 216.0, 213.0, 215.0)] * 3
    bars += [(214.0, 214.5, 150.0, 151.0)]  # collapses through the raised stop
    df = _scripted_trade(bars)
    _force_entry_on(df, len(df) - len(bars) - 1, monkeypatch)
    res = run_swing_backtest(
        df, "TEST", config=SwingStrategyConfig(rsi_exit=101.0), initial_cash=1_000_000.0
    )
    t = res.trades[0]
    assert t.partial_taken
    assert t.exit_reason in ("Trailing stop", "Supertrend flipped down")
    assert t.pnl > 0 and t.r_multiple > 0.9
    sells = [r for r in res.simulation.trades if r.side.value == "SELL"]
    assert len(sells) == 2 and sells[0].shares == t.shares // 2


def test_no_entry_inside_blackout_or_market_downtrend(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _scripted_trade([(182.0, 183.0, 181.0, 182.0), (182.0, 183.0, 181.0, 182.0)])
    sig_idx = len(df) - 3
    _force_entry_on(df, sig_idx, monkeypatch)
    day = pd.to_datetime(df["trade_date"]).dt.date.iloc[sig_idx]
    res = run_swing_backtest(df, "TEST", blackout_windows=[(day, day)])
    assert res.trades == []
    down = pd.Series({day: False})
    res2 = run_swing_backtest(df, "TEST", market_filter=down)
    assert res2.trades == []


def test_upper_lock_blocks_the_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _scripted_trade([(182.0, 183.0, 181.0, 182.0)])
    df["is_upper_lock"] = False
    df.loc[len(df) - 1, "is_upper_lock"] = True
    _force_entry_on(df, len(df) - 2, monkeypatch)
    res = run_swing_backtest(df, "TEST")
    assert res.trades == [] and res.simulation.rejections


def test_benchmark_and_strategy_cover_the_same_sessions() -> None:
    res = run_swing_backtest(_random_walk(600), "TEST")
    eq, bh = res.simulation.equity_curve, res.benchmark.equity_curve
    assert len(eq) == len(bh) and eq["trade_date"].iloc[0] == bh["trade_date"].iloc[0]
    assert 0.0 <= res.exposure <= 1.0
    assert res.summary()["sessions"] == len(eq)


def test_too_little_history_raises() -> None:
    with pytest.raises(ValueError, match="Not enough price history"):
        run_swing_backtest(_random_walk(150), "TEST")


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        SwingStrategyConfig(trail="nope")
    with pytest.raises(ValueError):
        SwingStrategyConfig(risk_per_trade=0.5)
