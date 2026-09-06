"""Financial performance metrics and Rich report generation for backtesting simulations."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from psx_predictor.backtesting.simulator import SimulationResult


@dataclass(frozen=True)
class BacktestMetrics:
    """Comprehensive portfolio performance and risk metrics."""

    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_friction_paid: float
    rejections_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to serializable dictionary."""
        return {
            "total_return": round(self.total_return, 4),
            "cagr": round(self.cagr, 4),
            "annualized_volatility": round(self.annualized_volatility, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "total_friction_paid": round(self.total_friction_paid, 2),
            "rejections_count": self.rejections_count,
        }


def calculate_backtest_metrics(
    result: SimulationResult,
    risk_free_rate: float = 0.15,  # 15.0% annual SBP policy rate
    sessions_per_year: int = 250,
) -> BacktestMetrics:
    """Compute financial, risk-adjusted, and execution metrics from simulation output."""
    equity_df = result.equity_curve
    n_sessions = len(equity_df)

    if n_sessions == 0:
        return BacktestMetrics(
            total_return=0.0,
            cagr=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration_days=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            total_friction_paid=0.0,
            rejections_count=0,
        )

    # 1. Total Return & CAGR
    tot_ret = result.total_return
    years = n_sessions / sessions_per_year
    if years > 0 and result.final_equity > 0:
        cagr = (result.final_equity / result.initial_cash) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    # 2. Daily returns & volatility
    daily_rets = equity_df["daily_return"].to_numpy()
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / sessions_per_year) - 1.0

    std_daily = np.std(daily_rets)
    ann_vol = float(std_daily * np.sqrt(sessions_per_year)) if std_daily > 0 else 0.0

    excess_daily = daily_rets - daily_rf
    mean_excess = np.mean(excess_daily)

    # 3. Sharpe Ratio
    if std_daily > 1e-9:
        sharpe = float(np.sqrt(sessions_per_year) * (mean_excess / std_daily))
    else:
        sharpe = 0.0

    # 4. Sortino Ratio (Downside deviation)
    downside_excess = np.minimum(excess_daily, 0.0)
    downside_std = np.sqrt(np.mean(downside_excess**2))
    if downside_std > 1e-9:
        sortino = float(np.sqrt(sessions_per_year) * (mean_excess / downside_std))
    else:
        sortino = 0.0

    # 5. Drawdown & Drawdown Duration
    drawdown_series = equity_df["drawdown"]
    max_dd = float(drawdown_series.min()) if len(drawdown_series) > 0 else 0.0

    # Calculate max duration in days of drawdown
    mdd_duration = 0
    current_duration = 0
    for dd in drawdown_series:
        if dd < 0:
            current_duration += 1
            if current_duration > mdd_duration:
                mdd_duration = current_duration
        else:
            current_duration = 0

    # 6. Trade performance (pairing BUY and SELL executions)
    trades = result.trades
    total_friction = sum(t.cost for t in trades)

    round_trip_pnl: list[float] = []
    current_entry_cost = 0.0

    for t in trades:
        if t.side.value == "BUY":
            current_entry_cost = t.net_value
        elif t.side.value == "SELL" and current_entry_cost > 0:
            net_proceeds = t.net_value
            pnl = net_proceeds - current_entry_cost
            round_trip_pnl.append(pnl)
            current_entry_cost = 0.0

    wins = [p for p in round_trip_pnl if p > 0]
    losses = [p for p in round_trip_pnl if p <= 0]
    n_trades = len(round_trip_pnl)
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = (n_wins / n_trades) if n_trades > 0 else 0.0

    sum_wins = sum(wins)
    sum_losses = abs(sum(losses))
    if sum_losses > 0:
        profit_factor = sum_wins / sum_losses
    elif sum_wins > 0:
        profit_factor = 99.0
    else:
        profit_factor = 0.0

    return BacktestMetrics(
        total_return=tot_ret,
        cagr=cagr,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration_days=mdd_duration,
        total_trades=n_trades,
        winning_trades=n_wins,
        losing_trades=n_losses,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_friction_paid=total_friction,
        rejections_count=len(result.rejections),
    )


def display_backtest_report(
    metrics: BacktestMetrics,
    result: SimulationResult,
    console: Optional[Console] = None,
) -> None:
    """Print an aesthetic Rich summary report for backtesting simulation."""
    c = console or Console()

    header_table = Table(
        box=box.ROUNDED,
        title=f"PSX Strategy Backtest Report: {result.symbol} ({result.strategy_name})",
        header_style="bold cyan",
        show_lines=True,
    )
    header_table.add_column("Portfolio / Metric", style="bold white")
    header_table.add_column("Value", style="bold green", justify="right")
    header_table.add_column("Benchmark / Context", style="dim", justify="left")

    header_table.add_row(
        "Initial Capital",
        f"PKR {result.initial_cash:,.2f}",
        "Starting cash allocation",
    )
    header_table.add_row(
        "Final Portfolio Equity",
        f"PKR {result.final_equity:,.2f}",
        "Ending mark-to-market value",
    )
    ret_color = "green" if metrics.total_return >= 0 else "red"
    header_table.add_row(
        "Cumulative Return",
        f"[{ret_color}]{metrics.total_return:+.2%}[/{ret_color}]",
        "Total unannualized return",
    )
    header_table.add_row(
        "CAGR (Annualized)",
        f"[{ret_color}]{metrics.cagr:+.2%}[/{ret_color}]",
        "Compound annual growth rate (250 sessions/yr)",
    )
    header_table.add_row(
        "Annualized Volatility",
        f"{metrics.annualized_volatility:.2%}",
        "Standard deviation of daily portfolio returns",
    )
    sharpe_color = "green" if metrics.sharpe_ratio > 0 else "red"
    header_table.add_row(
        "Sharpe Ratio (Rf=15%)",
        f"[{sharpe_color}]{metrics.sharpe_ratio:.2f}[/{sharpe_color}]",
        "Excess return per unit of volatility (SBP 15% Rf)",
    )
    header_table.add_row(
        "Sortino Ratio",
        f"[{sharpe_color}]{metrics.sortino_ratio:.2f}[/{sharpe_color}]",
        "Excess return per unit of downside risk",
    )
    header_table.add_row(
        "Maximum Drawdown (MDD)",
        f"[red]{metrics.max_drawdown:.2%}[/red]",
        f"Peak-to-trough decline (longest duration: {metrics.max_drawdown_duration_days} days)",
    )
    header_table.add_row(
        "Round-trip Trades",
        str(metrics.total_trades),
        f"Wins: {metrics.winning_trades} | Losses: {metrics.losing_trades}",
    )
    header_table.add_row(
        "Win Rate",
        f"{metrics.win_rate:.1%}",
        f"Profit Factor: {metrics.profit_factor:.2f}",
    )
    header_table.add_row(
        "Total Friction Paid",
        f"PKR {metrics.total_friction_paid:,.2f}",
        "Brokerage (15 bps) + Regulatory (1.5 bps) + Slippage (10 bps)",
    )
    header_table.add_row(
        "Circuit Lock Rejections",
        str(metrics.rejections_count),
        "Upper lock buy rejections & Lower lock sell rejections",
    )

    c.print()
    c.print(header_table)
    c.print(
        Panel(
            "[dim]Execution Rules: Orders executed at market close. "
            "Upper circuit lock (+7.5%) strictly prevents buy entries. "
            "Lower circuit lock (-7.5%) strictly prevents sell exits.[/dim]",
            border_style="dim",
        )
    )
