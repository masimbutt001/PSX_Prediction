"""PSX Predictor backtesting package: walk-forward partitioner, simulator, and metrics."""

from psx_predictor.backtesting.metrics import (
    BacktestMetrics,
    calculate_backtest_metrics,
    display_backtest_report,
)
from psx_predictor.backtesting.runner import BacktestRunner
from psx_predictor.backtesting.simulator import (
    BacktestSimulator,
    OrderSide,
    PositionState,
    SimulationResult,
    TradeRecord,
    TransactionCostModel,
)
from psx_predictor.backtesting.walk_forward import WalkForwardFold, WalkForwardPartitioner

__all__ = [
    "WalkForwardPartitioner",
    "WalkForwardFold",
    "BacktestSimulator",
    "SimulationResult",
    "TradeRecord",
    "OrderSide",
    "PositionState",
    "TransactionCostModel",
    "BacktestMetrics",
    "calculate_backtest_metrics",
    "display_backtest_report",
    "BacktestRunner",
]
