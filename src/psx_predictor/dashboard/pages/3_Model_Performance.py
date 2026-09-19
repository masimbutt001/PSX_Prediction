"""Page 3: Model Performance — Interactive backtest simulations and equity curve evaluation."""

import numpy as np
import pandas as pd
import streamlit as st

from psx_predictor.backtesting.metrics import calculate_backtest_metrics
from psx_predictor.backtesting.simulator import BacktestSimulator, TransactionCostModel
from psx_predictor.config.loader import load_config
from psx_predictor.dashboard.components.charts import (
    create_drawdown_chart,
    create_equity_curve_chart,
)
from psx_predictor.dashboard.components.metrics_table import (
    render_metrics_summary,
    render_trades_table,
)
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import get_storage_paths

st.set_page_config(page_title="Model Performance — PSX Predictor", page_icon="🧪", layout="wide")

st.title("🧪 Model Performance & Strategy Backtesting")
st.markdown(
    """
    Simulate quantitative trading strategies on historical PSX price series with
    institutional transaction costs (PSX broker commissions, SECP turnover tax, and slippage).
    """
)

cfg = load_config()
paths = get_storage_paths(cfg.settings.data_dir)
symbols = [s.symbol for s in cfg.stocks]

# Configuration controls
c_sym, c_strat, c_cash, c_rf = st.columns(4)
with c_sym:
    selected_sym = st.selectbox("Select Asset", symbols, index=0)
with c_strat:
    strategy_choice = st.selectbox(
        "Trading Strategy",
        ["SMA Crossover (10/30)", "1-Day Momentum", "Buy & Hold Benchmark"],
    )
with c_cash:
    initial_cash = st.number_input(
        "Initial Cash (PKR)", min_value=50_000, value=1_000_000, step=50_000
    )
with c_rf:
    rf_rate = st.slider(
        "Risk-Free Rate (Annual)", min_value=0.0, max_value=0.30, value=0.15, step=0.01
    )

price_file = paths["processed_prices"] / f"{selected_sym}.parquet"

if not price_file.exists():
    st.error(
        f"Price data for '{selected_sym}' not found. "
        "Run `psx bootstrap` or `psx update` first."
    )
else:
    df = read_parquet(price_file)
    if len(df) < 30:
        st.error(f"Insufficient session history ({len(df)} sessions) for backtesting.")
    else:
        # Generate signals based on selected strategy
        if "SMA Crossover" in strategy_choice:
            sma_10 = df["close"].rolling(10).mean()
            sma_30 = df["close"].rolling(30).mean()
            signals = np.where(sma_10 > sma_30, 1.0, 0.0)
        elif "Momentum" in strategy_choice:
            ret_1d = df["close"].pct_change().fillna(0.0)
            signals = np.where(ret_1d > 0.0, 1.0, 0.0)
        else:  # Buy & Hold
            signals = np.ones(len(df), dtype=float)

        # 1. Run strategy simulation
        simulator = BacktestSimulator(
            initial_cash=float(initial_cash),
            cost_model=TransactionCostModel(),
        )
        sim_res = simulator.simulate(
            df=df,
            signals=signals,
            symbol=selected_sym,
            strategy_name=strategy_choice,
        )

        # 2. Run Buy & Hold Benchmark for comparison
        b_signals = np.ones(len(df), dtype=float)
        b_sim = BacktestSimulator(
            initial_cash=float(initial_cash),
            cost_model=TransactionCostModel(),
        )
        b_res = b_sim.simulate(
            df=df,
            signals=b_signals,
            symbol=selected_sym,
            strategy_name="Buy & Hold Benchmark",
        )

        metrics = calculate_backtest_metrics(
            result=sim_res,
            risk_free_rate=float(rf_rate),
        )

        st.subheader("Performance Metrics Summary")
        render_metrics_summary(metrics)

        st.divider()

        # Cumulative Equity Progression Chart
        fig_equity = create_equity_curve_chart(
            equity_df=sim_res.equity_curve,
            symbol=selected_sym,
            benchmark_df=b_res.equity_curve,
        )
        st.plotly_chart(fig_equity, use_container_width=True)

        # Drawdown Profile Chart
        eq_curve = sim_res.equity_curve
        d_col = "trade_date" if "trade_date" in eq_curve.columns else "date"
        fig_drawdown = create_drawdown_chart(eq_curve["drawdown"], eq_curve[d_col])
        st.plotly_chart(fig_drawdown, use_container_width=True)

        # Trades Table
        st.divider()
        trades_records = [
            {
                "Execution Date": str(t.trade_date),
                "Side": str(t.side.value if hasattr(t.side, "value") else t.side),
                "Shares": t.shares,
                "Price (PKR)": f"{t.price:.2f}",
                "Friction (PKR)": f"{t.cost:.2f}",
                "Net Value (PKR)": f"{t.net_value:.2f}",
            }
            for t in sim_res.trades
        ]
        render_trades_table(pd.DataFrame(trades_records))
