"""FastAPI endpoints for backtesting simulations and equity curve evaluation."""

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from psx_predictor.api.schemas import BacktestSummaryResponse, EquityPoint
from psx_predictor.backtesting.metrics import calculate_backtest_metrics
from psx_predictor.backtesting.simulator import BacktestSimulator, TransactionCostModel
from psx_predictor.config.loader import load_config
from psx_predictor.features.technical import compute_moving_averages
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import get_storage_paths

router = APIRouter(prefix="/backtests", tags=["backtests"])


@router.get("/{symbol}", response_model=BacktestSummaryResponse)
def get_backtest_results(
    symbol: str,
    request: Request,
    strategy: str = Query(
        default="sma_crossover",
        description="Backtest strategy: 'sma_crossover', 'buy_and_hold', 'momentum'",
    ),
    initial_cash: float = Query(default=1_000_000.0, ge=10_000.0, description="Initial PKR cash"),
    rf_rate: float = Query(default=0.15, ge=0.0, le=0.5, description="Annual risk-free rate"),
) -> BacktestSummaryResponse:
    """Simulate trading strategy over historical price series with PSX transaction costs."""
    clean_sym = symbol.strip().upper()
    cfg = load_config()
    data_dir = getattr(request.app.state, "data_dir", None) or cfg.settings.data_dir
    paths = get_storage_paths(data_dir)
    price_file = paths["processed_prices"] / f"{clean_sym}.parquet"

    if not price_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Processed price data for '{clean_sym}' not found. Run bootstrap first.",
        )

    df = read_parquet(price_file)
    if len(df) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient history ({len(df)} sessions) for backtest on '{clean_sym}'.",
        )

    strat_name = strategy.lower().strip()

    # Generate signals based on selected strategy
    if strat_name in ("sma_crossover", "sma"):
        df_ma = compute_moving_averages(df)
        signals = np.where(df_ma["price_to_sma_20"] > 1.0, 1.0, 0.0)
        display_strat = "SMA(20) Crossover"
    elif strat_name in ("buy_and_hold", "hold"):
        signals = np.ones(len(df), dtype=float)
        display_strat = "Buy and Hold Benchmark"
    elif strat_name in ("momentum", "roc"):
        ret_1d = df["close"].pct_change().fillna(0.0)
        signals = np.where(ret_1d > 0.0, 1.0, 0.0)
        display_strat = "1-Day Momentum"
    else:
        valid_strats = "'sma_crossover', 'buy_and_hold', or 'momentum'"
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{strategy}'. Choose {valid_strats}.",
        )

    simulator = BacktestSimulator(
        initial_cash=initial_cash,
        cost_model=TransactionCostModel(),
    )

    sim_res = simulator.simulate(
        df=df,
        signals=signals,
        symbol=clean_sym,
        strategy_name=display_strat,
    )

    metrics = calculate_backtest_metrics(
        result=sim_res,
        risk_free_rate=rf_rate,
    )

    # Format equity curve points (sample up to 200 points if series is long)
    eq_df = sim_res.equity_curve
    step = max(1, len(eq_df) // 200)
    sampled_eq = eq_df.iloc[::step].copy()

    # Ensure last row is included
    if sampled_eq.index[-1] != eq_df.index[-1]:
        sampled_eq = pd.concat([sampled_eq, eq_df.iloc[[-1]]])

    curve_points: list[EquityPoint] = []
    for _, row in sampled_eq.iterrows():
        curve_points.append(
            EquityPoint(
                date=str(row["trade_date"]),
                portfolio_value=round(float(row["total_equity"]), 2),
                cash=round(float(row["cash"]), 2),
                drawdown=round(float(row.get("drawdown", 0.0)), 4),
            )
        )

    d_col = "trade_date" if "trade_date" in df.columns else "date"
    start_d = str(df[d_col].iloc[0])
    end_d = str(df[d_col].iloc[-1])

    return BacktestSummaryResponse(
        symbol=clean_sym,
        strategy_name=display_strat,
        start_date=start_d,
        end_date=end_d,
        total_return=round(metrics.total_return, 4),
        cagr=round(metrics.cagr, 4),
        sharpe_ratio=round(metrics.sharpe_ratio, 4),
        max_drawdown=round(metrics.max_drawdown, 4),
        win_rate=round(metrics.win_rate, 4),
        total_trades=metrics.total_trades,
        equity_curve=curve_points,
    )
