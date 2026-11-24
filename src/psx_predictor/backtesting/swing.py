"""Rule-based swing trading strategy (holding roughly 3 to 15 sessions) with a daily-bar backtest.

Rules (all decided on a session's close, orders filled at the next session's open):

1. Trend filter: close above SMA 200, EMA 20 above EMA 50, ADX 14 above 20.
2. Pullback entry: within the last few sessions the low touched EMA 20 or the middle
   Bollinger band and RSI 14 dipped into 40-50. Buy when the stock closes back up
   (higher close, above EMA 20) with the MACD histogram rising and relative volume
   of at least 1.2.
3. Stop loss: 1.5 x ATR 14 below the entry (or below the recent swing low).
4. Take profit: sell half at 2 x the risk (2R), move the stop on the rest to break-even
   and trail it with Supertrend (10, 3) or EMA 20. Exit fully if RSI went above 75 and
   then turns down.
5. Position size: risk 1% of equity per trade: shares = equity x 1% / stop distance.
6. PSX filters: skip illiquid stocks, skip entries in a window after a results /
   board-meeting / dividend notice, and stay out while KSE-100 is below its SMA 50.

Fills respect PSX circuit locks (no buying at the upper lock, no selling at the lower
lock) and pay the same per-side costs as the signal simulator.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from psx_predictor.backtesting.metrics import BacktestMetrics, calculate_backtest_metrics
from psx_predictor.backtesting.simulator import (
    OrderSide,
    SimulationResult,
    TradeRecord,
    TransactionCostModel,
)
from psx_predictor.features.technical import compute_all_technical_features
from psx_predictor.processing.corporate_actions import adjusted_ohlcv, apply_corporate_actions

STRATEGY_NAME = "Swing pullback"

# Headlines that announce an upcoming results meeting, dividend or book closure.
EVENT_KEYWORDS: tuple[str, ...] = (
    "board meeting",
    "financial result",
    "financial statement",
    "quarterly report",
    "annual report",
    "half year",
    "dividend",
    "book closure",
    "bonus",
    "right shares",
    "entitlement",
)

# News classifier categories that mean results or a payout are coming.
EVENT_CATEGORIES: frozenset[str] = frozenset({"EARNINGS", "DIVIDEND"})


@dataclass(frozen=True)
class SwingStrategyConfig:
    """Parameters of the swing strategy. Defaults follow the written rules."""

    # 1. Trend filter
    trend_sma: int = 200
    adx_min: float = 20.0
    # 2. Pullback entry
    setup_lookback: int = 3
    pullback_tolerance: float = 0.005  # low within 0.5% of EMA 20 / BB middle counts
    rsi_low: float = 40.0
    rsi_high: float = 50.0
    min_rel_volume: float = 1.2
    # 3. Stop loss
    atr_stop_mult: float = 1.5
    stop_mode: str = "atr"  # "atr" or "swing_low"
    swing_low_lookback: int = 10
    # 4. Take profit and trailing
    take_profit_r: float = 2.0
    partial_fraction: float = 0.5
    trail: str = "supertrend"  # "supertrend" or "ema20"
    rsi_exit: float = 75.0
    max_hold_sessions: int = 0  # 0 = no time stop
    # 5. Position sizing
    risk_per_trade: float = 0.01
    # 6. PSX filters
    min_avg_traded_value: float = 5_000_000.0  # 20-session average of close x volume (PKR)
    event_blackout_days: int = 7  # calendar days after a results/dividend notice
    market_sma: int = 50
    use_market_filter: bool = True

    def __post_init__(self) -> None:
        if self.stop_mode not in ("atr", "swing_low"):
            raise ValueError("stop_mode must be 'atr' or 'swing_low'.")
        if self.trail not in ("supertrend", "ema20"):
            raise ValueError("trail must be 'supertrend' or 'ema20'.")
        if not 0.0 < self.risk_per_trade <= 0.1:
            raise ValueError("risk_per_trade must be between 0 and 10%.")
        if not 0.0 < self.partial_fraction < 1.0:
            raise ValueError("partial_fraction must be between 0 and 1.")


@dataclass
class SwingTrade:
    """One position from entry to final exit."""

    entry_date: Any
    entry_price: float
    shares: int
    stop_price: float
    target_price: float
    exit_date: Any = None
    exit_price: float = float("nan")  # average exit price over all sells
    exit_reason: str = ""
    sessions_held: int = 0
    pnl: float = 0.0  # PKR, after all costs
    r_multiple: float = 0.0
    partial_taken: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class SwingBacktestResult:
    """Swing backtest output: the simulation, its metrics, trades and a buy-and-hold benchmark."""

    simulation: SimulationResult
    metrics: BacktestMetrics
    trades: list[SwingTrade]
    benchmark: SimulationResult
    benchmark_metrics: BacktestMetrics
    exposure: float  # share of sessions with an open position
    notes: list[str] = field(default_factory=list)

    @property
    def avg_r(self) -> float:
        closed = [t.r_multiple for t in self.trades if t.exit_date is not None]
        return float(np.mean(closed)) if closed else 0.0

    def summary(self) -> dict[str, Any]:
        m, b = self.metrics, self.benchmark_metrics
        return {
            "symbol": self.simulation.symbol,
            "sessions": len(self.simulation.equity_curve),
            "trades": m.total_trades,
            "win_rate": m.win_rate,
            "total_return": m.total_return,
            "max_drawdown": m.max_drawdown,
            "profit_factor": m.profit_factor,
            "avg_r": self.avg_r,
            "exposure": self.exposure,
            "bh_return": b.total_return,
            "bh_max_drawdown": b.max_drawdown,
        }


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------


def blackout_windows_from_news(
    news: pd.DataFrame,
    symbol: str,
    days: int = 7,
    keywords: Iterable[str] = EVENT_KEYWORDS,
) -> list[tuple[dt.date, dt.date]]:
    """(start, end) date windows after each results / dividend notice for `symbol`.

    PSX board-meeting and book-closure notices are published about a week before the
    event, so no new entries are taken from the notice date for `days` calendar days.
    """
    if news is None or news.empty or "published_at" not in news.columns:
        return []
    kw = tuple(k.lower() for k in keywords)
    windows: list[tuple[dt.date, dt.date]] = []
    for _, row in news.iterrows():
        syms = row.get("matched_symbols")
        sym_list = list(syms) if syms is not None and not isinstance(syms, str) else []
        if row.get("primary_symbol") != symbol and symbol not in sym_list:
            continue
        text = f"{row.get('headline', '')} {row.get('category', '')}".lower()
        is_event = str(row.get("event_category", "")).upper() in EVENT_CATEGORIES
        if not is_event and not any(k in text for k in kw):
            continue
        ts = pd.to_datetime(row.get("published_at"), errors="coerce", utc=True)
        if pd.isna(ts):
            continue
        start = ts.tz_convert("Asia/Karachi").date()
        windows.append((start, start + dt.timedelta(days=days)))
    return sorted(set(windows))


def market_uptrend(macro: Optional[pd.DataFrame], sma: int = 50) -> Optional[pd.Series]:
    """KSE-100 close above its SMA, indexed by session date; None when history is too short.

    The macro table back-fills KSE-100 before its first real observation, so only
    sessions where the index value actually changes count towards the history.
    """
    if macro is None or macro.empty or "kse100_index" not in macro.columns:
        return None
    m = macro[["session_date", "kse100_index"]].dropna()
    if m.empty:
        return None
    m = m.assign(session_date=pd.to_datetime(m["session_date"]).dt.date)
    m = m.drop_duplicates("session_date").sort_values("session_date")
    real = m[m["kse100_index"].diff().fillna(1.0) != 0.0]
    if len(real) < sma + 10:
        return None
    real = real.set_index("session_date")["kse100_index"].astype(float)
    above = real > real.rolling(window=sma, min_periods=sma).mean()
    return above.where(real.rolling(window=sma, min_periods=sma).mean().notna())


def prepare_swing_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily OHLCV plus every indicator the strategy reads, oldest first.

    Splits and bonus issues missing from the stored prices are back-adjusted first
    (see `processing.corporate_actions`), and the bars are put on the adjusted basis,
    so trades and indicators never see a split as a crash. Inferred actions are in
    `df.attrs["actions"]` as (date, ratio).
    """
    df = prices.copy()
    if "trade_date" not in df.columns:
        raise ValueError("Price data needs a 'trade_date' column.")
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df = df.sort_values("trade_date").reset_index(drop=True)
    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = df["close"]
    df["open"] = df["open"].fillna(df["close"])
    if "adjusted_close" not in df.columns:
        df["adjusted_close"] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["volume"] = df["volume"].fillna(0.0).astype(float)
    if "symbol" not in df.columns:
        df["symbol"] = "ASSET"
    df, actions = apply_corporate_actions(df)
    df = adjusted_ohlcv(df)
    df["adjusted_close"] = df["close"]
    # Indicators are always recomputed so they match the adjusted prices.
    df = compute_all_technical_features(df)
    df.attrs["actions"] = actions
    df["avg_traded_value_20"] = (
        (df["close"] * df["volume"]).rolling(window=20, min_periods=20).mean()
    )
    return df


# --------------------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------------------


def entry_signals(
    df: pd.DataFrame,
    config: Optional[SwingStrategyConfig] = None,
) -> pd.DataFrame:
    """Per-session booleans for each rule; `entry` is True on sessions that qualify.

    Everything here uses data up to and including that session's close only.
    """
    c = config or SwingStrategyConfig()
    close = df["close"]
    sma_col = f"sma_{c.trend_sma}"
    trend_sma = (
        df[sma_col]
        if sma_col in df.columns
        else df["adjusted_close"].rolling(window=c.trend_sma, min_periods=c.trend_sma).mean()
    )
    trend = (close > trend_sma) & (df["ema_20"] > df["ema_50"]) & (df["adx_14"] > c.adx_min)

    tol = 1.0 + c.pullback_tolerance
    touched = (df["low"] <= df["ema_20"] * tol) | (df["low"] <= df["bb_middle"] * tol)
    rsi_in_zone = (df["rsi_14"] >= c.rsi_low) & (df["rsi_14"] <= c.rsi_high)
    look = max(1, c.setup_lookback)
    pulled_back = touched.astype(float).rolling(look, min_periods=1).max() > 0
    rsi_dipped = rsi_in_zone.astype(float).rolling(look, min_periods=1).max() > 0

    closes_up = (close > close.shift(1)) & (close > df["ema_20"])
    macd_rising = df["macd_hist"] > df["macd_hist"].shift(1)
    volume_ok = df["rel_volume_20"] >= c.min_rel_volume
    liquid = df["avg_traded_value_20"] >= c.min_avg_traded_value

    out = pd.DataFrame(
        {
            "trend": trend.fillna(False),
            "pullback": (pulled_back & rsi_dipped).fillna(False),
            "trigger": (closes_up & macd_rising & volume_ok).fillna(False),
            "liquid": liquid.fillna(False),
        },
        index=df.index,
    )
    out["entry"] = out["trend"] & out["pullback"] & out["trigger"] & out["liquid"]
    return out


# --------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------


def _in_windows(day: dt.date, windows: list[tuple[dt.date, dt.date]]) -> bool:
    return any(start <= day <= end for start, end in windows)


def _finish_equity(rows: list[dict[str, Any]], initial_cash: float) -> pd.DataFrame:
    eq = pd.DataFrame(rows)
    if len(eq) > 0:
        eq["daily_return"] = eq["total_equity"].pct_change().fillna(0.0)
        eq["cum_return"] = eq["total_equity"] / initial_cash - 1.0
        eq["peak_equity"] = eq["total_equity"].cummax()
        eq["drawdown"] = (eq["total_equity"] - eq["peak_equity"]) / eq["peak_equity"]
    return eq


def _buy_and_hold(
    df: pd.DataFrame, initial_cash: float, cost_model: TransactionCostModel, symbol: str
) -> SimulationResult:
    """Buy at the first session's open, hold to the end (costs on the buy only)."""
    rows: list[dict[str, Any]] = []
    trades: list[TradeRecord] = []
    cash, shares = initial_cash, 0
    for i, row in df.iterrows():
        if i == 0:
            price = float(row["open"])
            shares = int(cash // (price * (1.0 + cost_model.total_one_way_pct)))
            gross = shares * price
            fee = cost_model.compute_cost(gross)
            cash -= gross + fee
            trades.append(
                TradeRecord(
                    row["trade_date"],
                    OrderSide.BUY,
                    shares,
                    price,
                    gross,
                    fee,
                    gross + fee,
                    cash,
                    "Buy and hold",
                )
            )
        close = float(row["close"])
        rows.append(
            {
                "trade_date": row["trade_date"],
                "close": close,
                "shares": shares,
                "cash": cash,
                "holdings_value": shares * close,
                "total_equity": cash + shares * close,
            }
        )
    eq = _finish_equity(rows, initial_cash)
    final = float(eq["total_equity"].iloc[-1]) if len(eq) else initial_cash
    return SimulationResult(
        symbol=symbol,
        strategy_name="Buy and hold",
        initial_cash=initial_cash,
        final_equity=final,
        total_return=final / initial_cash - 1.0,
        equity_curve=eq,
        trades=trades,
    )


def run_swing_backtest(
    prices: pd.DataFrame,
    symbol: str = "ASSET",
    config: Optional[SwingStrategyConfig] = None,
    initial_cash: float = 1_000_000.0,
    cost_model: Optional[TransactionCostModel] = None,
    blackout_windows: Optional[list[tuple[dt.date, dt.date]]] = None,
    market_filter: Optional[pd.Series] = None,
    risk_free_rate: float = 0.12,
    start_date: Optional[dt.date] = None,
) -> SwingBacktestResult:
    """Backtest the swing strategy on one stock's daily prices.

    Args:
        prices: Daily OHLCV (indicator columns are computed when missing).
        symbol: Ticker, for labels.
        config: Strategy parameters.
        initial_cash: Starting capital in PKR.
        cost_model: Per-side costs (default: PSX brokerage + fees + slippage).
        blackout_windows: Date windows with no new entries (results / dividend notices).
        market_filter: KSE-100 above its SMA 50, indexed by date; None skips the filter.
        risk_free_rate: Annual rate for the Sharpe ratio.
        start_date: Only report from this date (indicators still warm up on earlier data).
    """
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive.")
    config = config or SwingStrategyConfig()
    costs = cost_model or TransactionCostModel()
    full = prepare_swing_frame(prices)
    sig_full = entry_signals(full, config)

    # Start once the slowest indicator is ready, so both strategy and benchmark
    # cover the same sessions.
    sma_col = f"sma_{config.trend_sma}"
    ready = (
        full[sma_col].notna()
        if sma_col in full
        else pd.Series(np.arange(len(full)) >= config.trend_sma - 1, index=full.index)
    )
    first = int(ready.idxmax()) if ready.any() else len(full)
    if start_date is not None:
        after = full.index[full["trade_date"] >= start_date]
        first = max(first, int(after[0])) if len(after) else len(full)
    df = full.iloc[first:].reset_index(drop=True)
    sig = sig_full.iloc[first:].reset_index(drop=True)
    notes: list[str] = []
    if df.empty:
        raise ValueError(
            f"Not enough price history for {symbol}: the strategy needs "
            f"{config.trend_sma}+ sessions for its SMA {config.trend_sma} trend filter."
        )

    for day, ratio in full.attrs.get("actions", []):
        notes.append(
            f"Prices before {day} adjusted for a split or bonus issue (1 share became "
            f"{ratio:g}) missing from the stored data."
        )
    windows = blackout_windows or []
    use_market = config.use_market_filter and market_filter is not None
    if config.use_market_filter and market_filter is None:
        notes.append(
            "KSE-100 filter skipped: not enough KSE-100 history yet "
            f"(needs about {config.market_sma + 10} sessions)."
        )
    if not windows:
        notes.append("No results/dividend notices found, so no entry blackouts were applied.")

    cash = initial_cash
    shares = 0
    trade: Optional[SwingTrade] = None
    entry_cost = 0.0
    proceeds = 0.0
    sold_shares = 0
    risk_per_share = 0.0
    rsi_peaked = False
    pending_entry = False
    pending_exit: Optional[str] = None
    records: list[TradeRecord] = []
    rejections: list[dict[str, Any]] = []
    swing_trades: list[SwingTrade] = []
    rows: list[dict[str, Any]] = []
    in_market = 0

    def sell(day: Any, qty: int, price: float, note: str) -> None:
        nonlocal cash, shares, proceeds, sold_shares
        gross = qty * price
        fee = costs.compute_cost(gross)
        cash += gross - fee
        shares -= qty
        proceeds += gross - fee
        sold_shares += qty
        records.append(
            TradeRecord(day, OrderSide.SELL, qty, price, gross, fee, gross - fee, cash, note)
        )

    def close_trade(day: Any, reason: str, i: int) -> None:
        nonlocal trade
        assert trade is not None
        trade.exit_date = day
        trade.exit_reason = reason
        trade.exit_price = proceeds / sold_shares if sold_shares else float("nan")
        trade.pnl = proceeds - entry_cost
        risk_pkr = risk_per_share * trade.shares
        trade.r_multiple = trade.pnl / risk_pkr if risk_pkr > 0 else 0.0
        trade.sessions_held = i - entry_index
        swing_trades.append(trade)
        trade = None

    entry_index = 0
    for i, row in df.iterrows():
        day = row["trade_date"]
        o, h, lo, c = (float(row[k]) for k in ("open", "high", "low", "close"))
        upper_lock = bool(row.get("is_upper_lock", False))
        lower_lock = bool(row.get("is_lower_lock", False))

        # --- Orders decided on yesterday's close fill at today's open ---
        if pending_exit and trade is not None:
            if lower_lock:
                rejections.append(
                    {
                        "date": day,
                        "side": OrderSide.SELL,
                        "price": o,
                        "reason": "LOWER_CIRCUIT_BREAKER_LOCK",
                    }
                )
            else:
                sell(day, shares, o, pending_exit)
                close_trade(day, pending_exit, i)
                pending_exit = None
        elif pending_entry and trade is None:
            prev = df.iloc[i - 1]
            atr = float(prev["atr_14"])
            if config.stop_mode == "swing_low":
                lows = df["low"].iloc[max(0, i - config.swing_low_lookback) : i]
                stop = float(lows.min()) * 0.995
            else:
                stop = o - config.atr_stop_mult * atr
            risk = o - stop
            if upper_lock:
                rejections.append(
                    {
                        "date": day,
                        "side": OrderSide.BUY,
                        "price": o,
                        "reason": "UPPER_CIRCUIT_BREAKER_LOCK",
                    }
                )
            elif risk > 0 and math.isfinite(risk):
                equity = cash  # flat, so equity is all cash
                qty = int(equity * config.risk_per_trade // risk)
                qty = min(qty, int(cash // (o * (1.0 + costs.total_one_way_pct))))
                if qty > 0:
                    gross = qty * o
                    fee = costs.compute_cost(gross)
                    cash -= gross + fee
                    shares = qty
                    entry_cost = gross + fee
                    proceeds, sold_shares = 0.0, 0
                    risk_per_share = risk
                    rsi_peaked = False
                    entry_index = i
                    trade = SwingTrade(
                        entry_date=day,
                        entry_price=o,
                        shares=qty,
                        stop_price=stop,
                        target_price=o + config.take_profit_r * risk,
                    )
                    records.append(
                        TradeRecord(
                            day,
                            OrderSide.BUY,
                            qty,
                            o,
                            gross,
                            fee,
                            gross + fee,
                            cash,
                            f"Entry, stop {stop:.2f}, target {trade.target_price:.2f}",
                        )
                    )
        pending_entry = False

        # --- Intraday: stop loss, then take-profit (stop first when both are hit) ---
        if trade is not None and i > entry_index:
            stop = trade.stop_price
            if lo <= stop:
                if lower_lock:
                    rejections.append(
                        {
                            "date": day,
                            "side": OrderSide.SELL,
                            "price": stop,
                            "reason": "LOWER_CIRCUIT_BREAKER_LOCK",
                        }
                    )
                else:
                    reason = "Trailing stop" if trade.partial_taken else "Stop loss"
                    sell(day, shares, min(o, stop), reason)
                    close_trade(day, reason, i)
            elif not trade.partial_taken and h >= trade.target_price and not lower_lock:
                qty = int(shares * config.partial_fraction)
                if 0 < qty < shares:
                    sell(
                        day,
                        qty,
                        max(o, trade.target_price),
                        f"Take {config.partial_fraction:.0%} at {config.take_profit_r:g}R",
                    )
                trade.partial_taken = True
                trade.stop_price = max(trade.stop_price, trade.entry_price)

        # --- End of day: trail, RSI and time exits, decided on the close ---
        if trade is not None:
            rsi = float(row.get("rsi_14", np.nan))
            prev_rsi = float(df["rsi_14"].iloc[i - 1]) if i > 0 else np.nan
            if rsi > config.rsi_exit:
                rsi_peaked = True
            if trade.partial_taken:
                line = float(row["supertrend_10_3"])
                if config.trail == "supertrend" and row["supertrend_dir"] > 0 and line == line:
                    trade.stop_price = max(trade.stop_price, line)
                elif config.trail == "ema20" and c < float(row["ema_20"]):
                    pending_exit = "Closed below EMA 20"
            if rsi_peaked and rsi < prev_rsi and pending_exit is None:
                pending_exit = f"RSI above {config.rsi_exit:g} turned down"
            if (
                pending_exit is None
                and config.max_hold_sessions > 0
                and i - entry_index >= config.max_hold_sessions
            ):
                pending_exit = "Time stop"
            if (
                pending_exit is None
                and config.trail == "supertrend"
                and row.get("supertrend_dir", 1) < 0
                and trade.partial_taken
            ):
                pending_exit = "Supertrend flipped down"
        elif bool(sig.at[i, "entry"]) and i + 1 < len(df):
            allowed = not _in_windows(day, windows)
            if use_market and market_filter is not None:
                ok = market_filter.get(day)
                # Unknown KSE-100 state for this day (no reading yet) does not block.
                if ok is not None and not pd.isna(ok) and not bool(ok):
                    allowed = False
            pending_entry = allowed

        if shares > 0:
            in_market += 1
        rows.append(
            {
                "trade_date": day,
                "close": c,
                "signal": 1.0 if shares > 0 else 0.0,
                "position": "LONG" if shares > 0 else "FLAT",
                "shares": shares,
                "cash": cash,
                "holdings_value": shares * c,
                "total_equity": cash + shares * c,
            }
        )

    if trade is not None:
        notes.append(f"One trade still open at the end (entered {trade.entry_date}).")

    eq = _finish_equity(rows, initial_cash)
    final = float(eq["total_equity"].iloc[-1]) if len(eq) else initial_cash
    sim = SimulationResult(
        symbol=symbol,
        strategy_name=STRATEGY_NAME,
        initial_cash=initial_cash,
        final_equity=final,
        total_return=final / initial_cash - 1.0,
        equity_curve=eq,
        trades=records,
        rejections=rejections,
    )
    base = calculate_backtest_metrics(sim, risk_free_rate=risk_free_rate)
    wins = [t.pnl for t in swing_trades if t.pnl > 0]
    losses = [t.pnl for t in swing_trades if t.pnl <= 0]
    n = len(swing_trades)
    gross_loss = abs(sum(losses))
    pf = sum(wins) / gross_loss if gross_loss > 0 else (99.0 if wins else 0.0)
    metrics = dataclasses.replace(
        base,
        total_trades=n,
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=len(wins) / n if n else 0.0,
        profit_factor=pf,
    )
    if trade is not None:
        swing_trades.append(trade)

    bench = _buy_and_hold(df, initial_cash, costs, symbol)
    return SwingBacktestResult(
        simulation=sim,
        metrics=metrics,
        trades=swing_trades,
        benchmark=bench,
        benchmark_metrics=calculate_backtest_metrics(bench, risk_free_rate=risk_free_rate),
        exposure=in_market / len(df) if len(df) else 0.0,
        notes=notes,
    )
