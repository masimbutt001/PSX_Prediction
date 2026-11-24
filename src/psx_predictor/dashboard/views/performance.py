"""Performance: live track record and calibration of logged predictions, plus backtests."""

import numpy as np
import pandas as pd
import streamlit as st

from psx_predictor.backtesting.metrics import calculate_backtest_metrics
from psx_predictor.backtesting.simulator import BacktestSimulator, TransactionCostModel
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
                ["SMA crossover 10/30", "1-day momentum", "Buy and hold"],
                default="SMA crossover 10/30",
                key="bt_strategy",
            )
            or "SMA crossover 10/30"
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

    df = state.prices(sym)
    if len(df) < 30:
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
