"""Performance: live track record and calibration of logged predictions, plus backtests."""

import numpy as np
import pandas as pd
import streamlit as st

from psx_predictor.backtesting.metrics import calculate_backtest_metrics
from psx_predictor.backtesting.simulator import BacktestSimulator, TransactionCostModel
from psx_predictor.backtesting.swing import (
    STRATEGY_NAME as SWING,
)
from psx_predictor.backtesting.swing import (
    SwingStrategyConfig,
    blackout_windows_from_news,
    market_uptrend,
    run_swing_backtest,
)
from psx_predictor.dashboard import data, state
from psx_predictor.dashboard.components.figures import (
    calibration_chart,
    drawdown_chart,
    equity_chart,
)
from psx_predictor.dashboard.components.ui import disclaimer, kpi_row, page_header
from psx_predictor.dashboard.theme import palette
from psx_predictor.monitoring.calibration import compute_expected_calibration_error

p = palette()
page_header(
    "Model performance",
    "How the logged predictions turned out, and how simple strategies would have done "
    "after PSX costs.",
)

tab_live, tab_bt = st.tabs(
    [":material/fact_check: Live track record", ":material/science: Backtest a strategy"]
)

# ---- Live track record ----
with tab_live:
    preds = state.predictions()
    window = (
        st.segmented_control(
            "Window",
            ["30 days", "90 days", "All time"],
            default="90 days",
            label_visibility="collapsed",
            key="perf_window",
        )
        or "90 days"
    )
    days = {"30 days": 30, "90 days": 90, "All time": None}[window]
    res = data.resolved_predictions(preds)
    if days and not res.empty:
        cutoff = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        res = res[res["target_date"].astype(str) >= cutoff]

    if res.empty:
        st.info(
            "No scored predictions in this window yet. Predictions are scored once the "
            "target session's close is in the price data (the daily run's audit step)."
        )
    else:
        probs = res["up_probability"].astype(float).to_numpy()
        actual = res["actual_up"].to_numpy(dtype=float)
        brier = float(np.mean((probs - actual) ** 2))
        ece = compute_expected_calibration_error(probs, actual)
        hit = float(res["is_correct"].mean())
        base_up = float(actual.mean())
        kpi_row(
            [
                (
                    "Hit rate",
                    f"{hit:.1%}",
                    f"{len(res)} scored predictions",
                    "up" if hit >= 0.5 else "down",
                ),
                (
                    "Brier score",
                    f"{brier:.3f}",
                    "Coin flip scores 0.250",
                    "up" if brier < 0.25 else "down",
                ),
                (
                    "Calibration error",
                    f"{ece:.3f}",
                    "Gap between stated and real odds",
                    "up" if ece < 0.05 else "warn" if ece < 0.10 else "down",
                ),
                (
                    "Up days",
                    f"{base_up:.0%}",
                    "Sessions that closed higher",
                    "muted",
                ),
            ]
        )
        st.write("")
        c1, c2 = st.columns([1, 1.3], gap="large")
        with c1:
            with st.container(border=True):
                st.markdown("##### Are the probabilities honest?")
                st.caption(
                    "When the model says 60% UP, the stock should rise about 60% of the "
                    "time. Points near the dotted line mean it is well calibrated."
                )
                st.plotly_chart(
                    calibration_chart(res), width="stretch", config={"displayModeBar": False}
                )
        with c2:
            with st.container(border=True):
                st.markdown("##### Track record by stock")
                rec = data.track_record(res)
                table = rec.rename(
                    columns={
                        "symbol": "Stock",
                        "predictions": "Scored",
                        "hit_rate": "Hit rate",
                        "brier": "Brier",
                        "last_target": "Last scored",
                    }
                )
                table["Hit rate"] = table["Hit rate"] * 100
                st.dataframe(
                    table,
                    hide_index=True,
                    height=38 + 35 * len(table),
                    width="stretch",
                    column_config={
                        "Hit rate": st.column_config.ProgressColumn(
                            min_value=0.0, max_value=100.0, format="%.0f%%"
                        ),
                        "Brier": st.column_config.NumberColumn(format="%.3f"),
                    },
                )
                st.download_button(
                    "Export scored predictions",
                    res.to_csv(index=False).encode("utf-8"),
                    file_name="psx_scored_predictions.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )


def _swing_inputs(sym: str) -> tuple[list, object]:  # type: ignore[type-arg]
    """Results/dividend blackout windows for `sym` and the KSE-100 trend filter."""
    news = state.news(None, 100_000)
    return blackout_windows_from_news(news, sym), market_uptrend(state.macro())


@st.cache_data(ttl=300, show_spinner="Backtesting every stock…")
def _universe_table(cfg: SwingStrategyConfig, cash: float, rf: float) -> list[dict]:  # type: ignore[type-arg]
    """One summary row per stock with enough history, for the same swing rules."""
    rows = []
    for other in state.symbols():
        px = state.prices(other)
        if len(px) < cfg.trend_sma + 30:
            continue
        w, mk = _swing_inputs(other)
        r = run_swing_backtest(
            px,
            other,
            config=cfg,
            initial_cash=cash,
            blackout_windows=w,
            market_filter=mk,  # type: ignore[arg-type]
            risk_free_rate=rf,
        )
        s = r.summary()
        rows.append(
            {
                "Stock": other,
                "Trades": s["trades"],
                "Win rate": s["win_rate"] * 100,
                "Avg R": s["avg_r"],
                "Return": s["total_return"] * 100,
                "Max drawdown": s["max_drawdown"] * 100,
                "Buy and hold": s["bh_return"] * 100,
                "B&H drawdown": s["bh_max_drawdown"] * 100,
            }
        )
    return rows


def render_swing(
    sym: str, df: pd.DataFrame, cfg: SwingStrategyConfig, cash: float, rf: float
) -> None:
    """KPIs, equity, trades and an all-stocks table for the swing strategy."""
    windows, mkt = _swing_inputs(sym)
    res = run_swing_backtest(
        df,
        sym,
        config=cfg,
        initial_cash=cash,
        blackout_windows=windows,
        market_filter=mkt,  # type: ignore[arg-type]
        risk_free_rate=rf,
    )
    m, mb = res.metrics, res.benchmark_metrics
    diff = (m.total_return - mb.total_return) * 100
    kpi_row(
        [
            (
                "Total return",
                f"{m.total_return:+.1%}",
                f"Buy and hold {mb.total_return:+.1%}",
                "up" if m.total_return > 0 else "down",
            ),
            (
                "Win rate",
                f"{m.win_rate:.0%}",
                f"{m.total_trades} closed trades",
                "up" if m.win_rate >= 0.5 else "muted",
            ),
            (
                "Average trade",
                f"{res.avg_r:+.2f}R",
                "Profit in units of risk",
                "up" if res.avg_r > 0 else "down",
            ),
            (
                "Max drawdown",
                f"{-abs(m.max_drawdown):.1%}",
                f"Buy and hold {-abs(mb.max_drawdown):.1%}",
                "down",
            ),
            ("Time in market", f"{res.exposure:.0%}", "Sessions holding shares", "muted"),
            (
                "vs buy and hold",
                f"{diff:+.1f} pts",
                "Return difference",
                "up" if diff > 0 else "down",
            ),
        ]
    )
    for note in res.notes:
        st.caption(f":material/info: {note}")
    st.write("")
    with st.container(border=True):
        st.markdown(f"##### {sym} · portfolio value")
        st.plotly_chart(
            equity_chart(res.simulation.equity_curve, res.benchmark.equity_curve, SWING),
            width="stretch",
        )
        st.markdown("###### Drawdown")
        st.plotly_chart(
            drawdown_chart(res.simulation.equity_curve),
            width="stretch",
            config={"displayModeBar": False},
        )

    trades = pd.DataFrame(
        [
            {
                "Entry": str(t.entry_date),
                "Exit": str(t.exit_date) if t.exit_date is not None else "Open",
                "Shares": t.shares,
                "Entry (PKR)": t.entry_price,
                "Stop (PKR)": t.stop_price,
                "Avg exit (PKR)": t.exit_price,
                "Sessions": t.sessions_held,
                "P&L (PKR)": t.pnl,
                "R": t.r_multiple,
                "Exit reason": t.exit_reason or "Still open",
            }
            for t in res.trades
        ]
    )
    with st.expander(f"Trades ({len(trades)})"):
        if trades.empty:
            st.caption("No setups met every rule in this period.")
        else:
            st.dataframe(
                trades,
                hide_index=True,
                width="stretch",
                column_config={
                    "Entry (PKR)": st.column_config.NumberColumn(format="%.2f"),
                    "Stop (PKR)": st.column_config.NumberColumn(format="%.2f"),
                    "Avg exit (PKR)": st.column_config.NumberColumn(format="%.2f"),
                    "P&L (PKR)": st.column_config.NumberColumn(format="localized"),
                    "R": st.column_config.NumberColumn(format="%+.2f"),
                },
            )
            st.download_button(
                "Export trades",
                trades.to_csv(index=False).encode("utf-8"),
                file_name=f"{sym}_swing_trades.csv",
                mime="text/csv",
                icon=":material/download:",
            )

    with st.expander(":material/table_chart: Same rules on every stock"):
        rows = _universe_table(cfg, cash, rf)
        if rows:
            pct = st.column_config.NumberColumn(format="%+.1f%%")
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "Win rate": st.column_config.NumberColumn(format="%.0f%%"),
                    "Avg R": st.column_config.NumberColumn(format="%+.2f"),
                    "Return": pct,
                    "Max drawdown": pct,
                    "Buy and hold": pct,
                    "B&H drawdown": pct,
                },
            )
        else:
            st.caption("No stock has enough price history yet.")


# ---- Backtest ----
with tab_bt:
    with st.container(border=True):
        c_sym, c_strat, c_cash, c_rf = st.columns([1, 2.6, 1.1, 1.1], vertical_alignment="bottom")
        syms = state.symbols()
        sym = c_sym.selectbox(
            "Stock",
            syms,
            index=syms.index(state.selected_symbol()) if state.selected_symbol() in syms else 0,
        )
        strategy = (
            c_strat.segmented_control(
                "Strategy",
                [SWING, "SMA crossover 10/30", "1-day momentum", "Buy and hold"],
                default=SWING,
                key="bt_strategy",
            )
            or SWING
        )
        cash = c_cash.number_input(
            "Starting cash (PKR)", min_value=50_000, value=1_000_000, step=50_000
        )
        rf = c_rf.slider(
            "Risk-free rate",
            0.0,
            0.30,
            0.12,
            0.01,
            format="%.2f",
            help="Annual, used for the Sharpe ratio",
        )

    swing_cfg = SwingStrategyConfig()
    if strategy == SWING:
        with st.expander(":material/tune: Swing rules", expanded=False):
            st.caption(
                "Buy pullbacks in an uptrend: close above SMA 200, EMA 20 above EMA 50 and "
                "ADX above the minimum. The low touches EMA 20 or the middle Bollinger band "
                "while RSI dips into the zone, then the stock closes back up with MACD rising "
                "and volume above normal. Orders fill at the next open. Stop at ATR x the "
                "multiple, sell half at the profit target, trail the rest with Supertrend."
            )
            r1, r2, r3, r4 = st.columns(4)
            adx_min = r1.slider("ADX minimum", 10, 35, 20)
            rsi_zone = r2.slider("RSI pullback zone", 25, 65, (40, 50))
            rel_vol = r3.slider("Relative volume", 0.5, 2.5, 1.2, 0.1)
            risk_pct = r4.slider("Risk per trade", 0.5, 3.0, 1.0, 0.25, format="%.2f%%")
            r5, r6, r7, r8 = st.columns(4)
            atr_mult = r5.slider("Stop (x ATR)", 1.0, 3.0, 1.5, 0.25)
            target_r = r6.slider("Take half at (R)", 1.0, 4.0, 2.0, 0.5)
            trail = r7.selectbox("Trail the rest with", ["Supertrend", "EMA 20"])
            max_hold = r8.number_input(
                "Time stop (sessions)",
                min_value=0,
                max_value=60,
                value=0,
                help="Sell after this many sessions. 0 turns it off.",
            )
            use_mkt = st.toggle(
                "KSE-100 above SMA 50",
                value=True,
                help="Skipped automatically until there is enough KSE-100 history.",
            )
            swing_cfg = SwingStrategyConfig(
                adx_min=float(adx_min),
                rsi_low=float(rsi_zone[0]),
                rsi_high=float(rsi_zone[1]),
                min_rel_volume=float(rel_vol),
                risk_per_trade=float(risk_pct) / 100.0,
                atr_stop_mult=float(atr_mult),
                take_profit_r=float(target_r),
                trail="supertrend" if trail == "Supertrend" else "ema20",
                use_market_filter=bool(use_mkt),
                max_hold_sessions=int(max_hold),
            )

    df = state.prices(sym)
    if strategy == SWING:
        if len(df) < swing_cfg.trend_sma + 30:
            st.warning(
                f"The swing strategy needs at least {swing_cfg.trend_sma + 30} sessions of "
                f"prices for {sym} (have {len(df)})."
            )
        else:
            render_swing(sym, df, swing_cfg, float(cash), float(rf))
    elif len(df) < 30:
        st.warning(f"Need at least 30 sessions of prices for {sym} (have {len(df)}).")
    else:
        if strategy.startswith("SMA"):
            signals = np.where(
                df["close"].rolling(10).mean() > df["close"].rolling(30).mean(), 1.0, 0.0
            )
        elif strategy.startswith("1-day"):
            signals = np.where(df["close"].pct_change().fillna(0.0) > 0.0, 1.0, 0.0)
        else:
            signals = np.ones(len(df), dtype=float)

        def run(sig: np.ndarray, name: str):  # type: ignore[no-untyped-def]
            return BacktestSimulator(
                initial_cash=float(cash), cost_model=TransactionCostModel()
            ).simulate(df=df, signals=sig, symbol=sym, strategy_name=name)

        result = run(signals, strategy)
        bench = run(np.ones(len(df), dtype=float), "Buy and hold")
        m = calculate_backtest_metrics(result=result, risk_free_rate=float(rf))
        mb = calculate_backtest_metrics(result=bench, risk_free_rate=float(rf))
        diff = (m.total_return - mb.total_return) * 100
        kpi_row(
            [
                (
                    "Total return",
                    f"{m.total_return:+.1%}",
                    "After costs",
                    "up" if m.total_return > 0 else "down",
                ),
                ("CAGR", f"{m.cagr:+.1%}", "Per year", "muted"),
                ("Sharpe ratio", f"{m.sharpe_ratio:.2f}", f"Risk-free {rf:.0%}", "muted"),
                ("Max drawdown", f"{-abs(m.max_drawdown):.1%}", "Worst fall from a peak", "down"),
                ("Win rate", f"{m.win_rate:.0%}", f"{m.total_trades} trades", "muted"),
                (
                    "vs buy and hold",
                    f"{diff:+.1f} pts",
                    "Return difference",
                    "up" if diff > 0 else "down",
                ),
            ]
        )
        st.write("")
        with st.container(border=True):
            st.markdown(f"##### {sym} · portfolio value")
            st.plotly_chart(
                equity_chart(result.equity_curve, bench.equity_curve, strategy), width="stretch"
            )
            st.markdown("###### Drawdown")
            st.plotly_chart(
                drawdown_chart(result.equity_curve),
                width="stretch",
                config={"displayModeBar": False},
            )
        trades = pd.DataFrame(
            [
                {
                    "Date": str(t.trade_date),
                    "Side": str(t.side.value if hasattr(t.side, "value") else t.side),
                    "Shares": t.shares,
                    "Price (PKR)": t.price,
                    "Costs (PKR)": t.cost,
                    "Net value (PKR)": t.net_value,
                }
                for t in result.trades
            ]
        )
        with st.expander(f"Trades ({len(trades)})"):
            if trades.empty:
                st.caption("No trades in this period.")
            else:
                st.dataframe(
                    trades,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Price (PKR)": st.column_config.NumberColumn(format="%.2f"),
                        "Costs (PKR)": st.column_config.NumberColumn(format="%.2f"),
                        "Net value (PKR)": st.column_config.NumberColumn(format="localized"),
                    },
                )
                st.download_button(
                    "Export trades",
                    trades.to_csv(index=False).encode("utf-8"),
                    file_name=f"{sym}_trades.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )

disclaimer()
