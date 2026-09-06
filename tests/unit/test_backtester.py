"""Unit tests for Phase 08: Walk-forward backtesting, trading simulation, and metrics."""

import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from psx_predictor.backtesting.metrics import (
    calculate_backtest_metrics,
)
from psx_predictor.backtesting.runner import BacktestRunner
from psx_predictor.backtesting.simulator import (
    BacktestSimulator,
    OrderSide,
    TransactionCostModel,
)
from psx_predictor.backtesting.walk_forward import WalkForwardPartitioner
from psx_predictor.cli.main import app
from psx_predictor.storage.paths import get_storage_paths


def test_walk_forward_splits_no_overlap() -> None:
    """Validate that walk-forward test indices strictly follow train indices plus embargo."""
    dates = [datetime.date(2022, 1, 1) + datetime.timedelta(days=i) for i in range(700)]
    df = pd.DataFrame({"trade_date": dates, "val": range(700)})

    partitioner = WalkForwardPartitioner(
        train_window=300,
        test_window=50,
        step_size=50,
        embargo=5,
        mode="expanding",
    )
    folds = partitioner.split(df)

    assert len(folds) >= 5
    for fold in folds:
        assert len(fold.train_indices) >= 300
        assert len(fold.test_indices) <= 50
        # Strict separation: test_start >= train_end + embargo
        max_train_idx = fold.train_indices[-1]
        min_test_idx = fold.test_indices[0]
        assert min_test_idx >= max_train_idx + 1 + 5
        # Date ordering
        assert fold.test_start_date > fold.train_end_date


def test_walk_forward_rolling_mode() -> None:
    """Validate rolling window mode maintains fixed training window."""
    dates = [datetime.date(2022, 1, 1) + datetime.timedelta(days=i) for i in range(500)]
    df = pd.DataFrame({"trade_date": dates, "val": range(500)})

    partitioner = WalkForwardPartitioner(
        train_window=200,
        test_window=40,
        step_size=40,
        embargo=2,
        mode="rolling",
    )
    folds = partitioner.split(df)

    for fold in folds:
        assert len(fold.train_indices) == 200
        assert fold.test_indices[0] == fold.train_indices[-1] + 1 + 2


def test_upper_lock_prevents_buy_execution() -> None:
    """Verify that is_upper_lock == True rejects buy entries."""
    df = pd.DataFrame(
        {
            "trade_date": [datetime.date(2023, 1, 1), datetime.date(2023, 1, 2)],
            "close": [100.0, 107.5],
            "is_upper_lock": [True, False],
            "is_lower_lock": [False, False],
        }
    )
    signals = pd.Series([1.0, 1.0])

    simulator = BacktestSimulator(initial_cash=100_000.0)
    result = simulator.simulate(df, signals)

    # First session was locked -> buy rejected
    assert len(result.rejections) == 1
    assert result.rejections[0]["reason"] == "UPPER_CIRCUIT_BREAKER_LOCK"
    # Second session was not locked -> buy executed
    assert len(result.trades) == 1
    assert result.trades[0].side == OrderSide.BUY


def test_lower_lock_prevents_sell_execution() -> None:
    """Verify that is_lower_lock == True rejects sell exits."""
    df = pd.DataFrame(
        {
            "trade_date": [
                datetime.date(2023, 1, 1),
                datetime.date(2023, 1, 2),
                datetime.date(2023, 1, 3),
            ],
            "close": [100.0, 92.5, 95.0],
            "is_upper_lock": [False, False, False],
            "is_lower_lock": [False, True, False],
        }
    )
    # Signal: buy at t=0, sell at t=1 (locked), sell at t=2
    signals = pd.Series([1.0, 0.0, 0.0])

    simulator = BacktestSimulator(initial_cash=100_000.0)
    result = simulator.simulate(df, signals)

    # Buy at t=0
    assert result.trades[0].side == OrderSide.BUY
    # At t=1 sell was attempted but lower lock active -> rejected!
    assert any(r["reason"] == "LOWER_CIRCUIT_BREAKER_LOCK" for r in result.rejections)
    # At t=2 sell was executed
    assert result.trades[1].side == OrderSide.SELL


def test_transaction_cost_deduction() -> None:
    """Verify precise deduction of friction on round-trip trades."""
    cost_model = TransactionCostModel(
        brokerage_pct=0.0015,
        regulatory_pct=0.00015,
        slippage_pct=0.0010,
    )
    assert np.isclose(cost_model.total_one_way_pct, 0.00265)

    df = pd.DataFrame(
        {
            "trade_date": [datetime.date(2023, 1, 1), datetime.date(2023, 1, 2)],
            "close": [100.0, 100.0],
            "is_upper_lock": [False, False],
            "is_lower_lock": [False, False],
        }
    )
    signals = pd.Series([1.0, 0.0])  # Buy then immediately sell at same price

    initial_cash = 100_000.0
    simulator = BacktestSimulator(initial_cash=initial_cash, cost_model=cost_model)
    result = simulator.simulate(df, signals)

    # Price was unchanged, so final equity must be strictly less than initial cash by friction
    assert result.final_equity < initial_cash
    total_cost = sum(t.cost for t in result.trades)
    assert total_cost > 0
    # Difference must equal sum of trade costs within float precision
    assert np.isclose(initial_cash - result.final_equity, total_cost, atol=1e-2)


def test_financial_metrics_calculation() -> None:
    """Validate Sharpe, Sortino, CAGR, and Drawdown calculations."""
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(250)]
    # Flat upward trending equity curve
    equity_values = np.linspace(1_000_000.0, 1_250_000.0, 250)

    from psx_predictor.backtesting.simulator import SimulationResult

    equity_df = pd.DataFrame(
        {
            "trade_date": dates,
            "total_equity": equity_values,
            "daily_return": pd.Series(equity_values).pct_change().fillna(0.0),
            "cum_return": (equity_values / 1_000_000.0) - 1.0,
            "peak_equity": equity_values,
            "drawdown": np.zeros(250),
        }
    )

    dummy_result = SimulationResult(
        symbol="TEST",
        strategy_name="TEST_STRAT",
        initial_cash=1_000_000.0,
        final_equity=1_250_000.0,
        total_return=0.25,
        equity_curve=equity_df,
    )

    metrics = calculate_backtest_metrics(dummy_result, risk_free_rate=0.15)
    assert np.isclose(metrics.total_return, 0.25)
    assert np.isclose(metrics.cagr, 0.25, atol=0.01)
    assert metrics.max_drawdown == 0.0
    assert metrics.sharpe_ratio > 0.0


def test_backtest_runner_integration(tmp_path: Path) -> None:
    """Verify BacktestRunner generates valid simulation and metrics on feature parquet."""
    storage_paths = get_storage_paths(tmp_path)
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(120)]

    df = pd.DataFrame(
        {
            "symbol": ["OGDC"] * 120,
            "trade_date": dates,
            "close": np.linspace(100, 150, 120),
            "sma_20": np.linspace(102, 148, 120),
            "sma_50": np.linspace(98, 140, 120),
            "log_ret_1d": [0.005] * 120,
            "is_upper_lock": [False] * 120,
            "is_lower_lock": [False] * 120,
        }
    )

    feat_dir = storage_paths["features_technical"]
    feat_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(feat_dir / "OGDC_tech_features.parquet")

    runner = BacktestRunner(storage_paths=storage_paths)
    result, metrics = runner.run_strategy("OGDC", strategy="sma_crossover")

    assert result.final_equity > 0
    assert len(result.equity_curve) == 120
    assert metrics.total_return is not None


def test_cli_backtest_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI command psx backtest outputs table."""
    cli_runner = CliRunner()
    storage_paths = get_storage_paths(tmp_path)
    dates = [datetime.date(2023, 1, 1) + datetime.timedelta(days=i) for i in range(120)]

    df = pd.DataFrame(
        {
            "symbol": ["OGDC"] * 120,
            "trade_date": dates,
            "close": np.linspace(100, 150, 120),
            "sma_20": np.linspace(102, 148, 120),
            "sma_50": np.linspace(98, 140, 120),
            "log_ret_1d": [0.005] * 120,
            "is_upper_lock": [False] * 120,
            "is_lower_lock": [False] * 120,
        }
    )

    feat_dir = storage_paths["features_technical"]
    feat_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(feat_dir / "OGDC_tech_features.parquet")

    from psx_predictor.backtesting import runner

    monkeypatch.setattr(
        runner,
        "load_config",
        lambda *args, **kwargs: MagicMock(
            settings=MagicMock(data_dir=tmp_path),
        ),
    )

    args = ["backtest", "--symbol", "OGDC", "--strategy", "sma_crossover"]
    res = cli_runner.invoke(app, args, env={"COLUMNS": "160"})

    assert res.exit_code == 0
    assert "Initiating backtest for OGDC" in res.stdout
    assert "PSX Strategy Backtest Report: OGDC (SMA_CROSSOVER)" in res.stdout
    assert "Final Portfolio Equity" in res.stdout
