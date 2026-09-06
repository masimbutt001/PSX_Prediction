"""Market simulation engine with realistic PSX transaction costs and circuit locks."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd


class OrderSide(str, Enum):
    """Trading order direction."""

    BUY = "BUY"
    SELL = "SELL"


class PositionState(str, Enum):
    """Portfolio market exposure state."""

    FLAT = "FLAT"
    LONG = "LONG"


@dataclass(frozen=True)
class TransactionCostModel:
    """PSX market transaction cost structure.

    Default parameters (per side):
        - Brokerage commission: 15 bps (0.15%)
        - SECP / CDC / NCCPL fees: 1.5 bps (0.015%)
        - Slippage: 10 bps (0.10%)
        - Total one-way friction: 26.5 bps (0.265%)
    """

    brokerage_pct: float = 0.0015
    regulatory_pct: float = 0.00015
    slippage_pct: float = 0.0010

    @property
    def total_one_way_pct(self) -> float:
        """Total friction applied per transaction side."""
        return self.brokerage_pct + self.regulatory_pct + self.slippage_pct

    def compute_cost(self, trade_value: float) -> float:
        """Compute absolute transaction friction for a trade value."""
        return float(trade_value * self.total_one_way_pct)


@dataclass
class TradeRecord:
    """Individual trade execution record."""

    trade_date: Any
    side: OrderSide
    shares: int
    price: float
    gross_value: float
    cost: float
    net_value: float
    cash_after: float
    note: str = ""


@dataclass
class SimulationResult:
    """Complete results from backtest simulation run."""

    symbol: str
    strategy_name: str
    initial_cash: float
    final_equity: float
    total_return: float
    equity_curve: pd.DataFrame
    trades: list[TradeRecord] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)


class BacktestSimulator:
    """Simulates spot market execution with PSX microstructure constraints.

    Rules:
        1. Long-only spot equity trading.
        2. Upper Lock Rejection: Buy orders are rejected when is_upper_lock == True.
        3. Lower Lock Rejection: Sell orders are rejected when is_lower_lock == True.
        4. Frictions are deducted from cash on buy, and deducted from proceeds on sell.
    """

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        cost_model: Optional[TransactionCostModel] = None,
    ) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive.")
        self.initial_cash = initial_cash
        self.cost_model = cost_model or TransactionCostModel()

    def simulate(
        self,
        df: pd.DataFrame,
        signals: np.ndarray | pd.Series,
        symbol: str = "ASSET",
        strategy_name: str = "Strategy",
    ) -> SimulationResult:
        """Run step-by-step portfolio simulation along price series and signals.

        Args:
            df: Historical DataFrame with ['trade_date', 'close'] and optional
                ['is_upper_lock', 'is_lower_lock'].
            signals: Array of binary signals (1.0 = Target LONG, 0.0 = Target FLAT).
            symbol: Ticker symbol.
            strategy_name: Name of strategy producing signals.

        Returns:
            SimulationResult containing equity curve, executed trades, and rejected orders.
        """
        signals_arr = np.asarray(signals)
        if len(signals_arr) != len(df):
            raise ValueError(
                f"Signals length ({len(signals_arr)}) does not match DataFrame ({len(df)})."
            )

        cash = self.initial_cash
        shares = 0
        position = PositionState.FLAT
        trades: list[TradeRecord] = []
        rejections: list[dict[str, Any]] = []

        equity_history: list[dict[str, Any]] = []

        for i, row in df.reset_index(drop=True).iterrows():
            date = row.get("trade_date", i)
            price = float(row["close"])
            upper_lock = bool(row.get("is_upper_lock", False))
            lower_lock = bool(row.get("is_lower_lock", False))
            target_signal = float(signals_arr[i])

            # Process execution signals
            if target_signal > 0.5 and position == PositionState.FLAT:
                # Attempt to enter LONG
                if upper_lock:
                    rejections.append(
                        {
                            "date": date,
                            "side": OrderSide.BUY,
                            "price": price,
                            "reason": "UPPER_CIRCUIT_BREAKER_LOCK",
                        }
                    )
                else:
                    # Allocate available cash (leaving room for friction)
                    eff_cost_mult = 1.0 + self.cost_model.total_one_way_pct
                    affordable_shares = int(cash // (price * eff_cost_mult))

                    # PSX standard lot size: allow integer shares >= 1
                    if affordable_shares > 0:
                        gross_val = affordable_shares * price
                        friction = self.cost_model.compute_cost(gross_val)
                        total_outflow = gross_val + friction

                        if total_outflow <= cash:
                            cash -= total_outflow
                            shares = affordable_shares
                            position = PositionState.LONG

                            trades.append(
                                TradeRecord(
                                    trade_date=date,
                                    side=OrderSide.BUY,
                                    shares=shares,
                                    price=price,
                                    gross_value=gross_val,
                                    cost=friction,
                                    net_value=total_outflow,
                                    cash_after=cash,
                                    note="Entry LONG",
                                )
                            )

            elif target_signal <= 0.5 and position == PositionState.LONG:
                # Attempt to exit to FLAT
                if lower_lock:
                    rejections.append(
                        {
                            "date": date,
                            "side": OrderSide.SELL,
                            "price": price,
                            "reason": "LOWER_CIRCUIT_BREAKER_LOCK",
                        }
                    )
                else:
                    gross_val = shares * price
                    friction = self.cost_model.compute_cost(gross_val)
                    net_inflow = gross_val - friction

                    cash += net_inflow
                    trades.append(
                        TradeRecord(
                            trade_date=date,
                            side=OrderSide.SELL,
                            shares=shares,
                            price=price,
                            gross_value=gross_val,
                            cost=friction,
                            net_value=net_inflow,
                            cash_after=cash,
                            note="Exit to FLAT",
                        )
                    )
                    shares = 0
                    position = PositionState.FLAT

            # End of day mark-to-market
            holdings_value = shares * price
            total_equity = cash + holdings_value

            equity_history.append(
                {
                    "trade_date": date,
                    "close": price,
                    "signal": target_signal,
                    "position": position.value,
                    "shares": shares,
                    "cash": cash,
                    "holdings_value": holdings_value,
                    "total_equity": total_equity,
                }
            )

        equity_df = pd.DataFrame(equity_history)
        if len(equity_df) > 0:
            equity_df["daily_return"] = equity_df["total_equity"].pct_change().fillna(0.0)
            equity_df["cum_return"] = (equity_df["total_equity"] / self.initial_cash) - 1.0
            equity_df["peak_equity"] = equity_df["total_equity"].cummax()
            equity_df["drawdown"] = (
                equity_df["total_equity"] - equity_df["peak_equity"]
            ) / equity_df["peak_equity"]

        final_eq = (
            float(equity_df["total_equity"].iloc[-1]) if len(equity_df) > 0 else self.initial_cash
        )
        tot_ret = (final_eq / self.initial_cash) - 1.0

        return SimulationResult(
            symbol=symbol,
            strategy_name=strategy_name,
            initial_cash=self.initial_cash,
            final_equity=final_eq,
            total_return=tot_ret,
            equity_curve=equity_df,
            trades=trades,
            rejections=rejections,
        )
