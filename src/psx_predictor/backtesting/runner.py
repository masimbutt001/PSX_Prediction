"""Backtesting execution runner orchestrating strategy signals and walk-forward folds."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from psx_predictor.backtesting.metrics import (
    BacktestMetrics,
    calculate_backtest_metrics,
)
from psx_predictor.backtesting.simulator import (
    BacktestSimulator,
    SimulationResult,
    TransactionCostModel,
)
from psx_predictor.backtesting.walk_forward import WalkForwardPartitioner
from psx_predictor.config.loader import load_config
from psx_predictor.models.linear import LogisticRegressionBaseline
from psx_predictor.models.split import DEFAULT_TECHNICAL_FEATURES
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import ensure_directories


class BacktestRunner:
    """Executes trading strategies and walk-forward simulations against PSX market data."""

    def __init__(self, storage_paths: Optional[dict[str, Path]] = None) -> None:
        if storage_paths is None:
            config = load_config()
            self.storage_paths = ensure_directories(config.settings.data_dir)
        else:
            self.storage_paths = storage_paths

    def load_dataset(self, symbol: str) -> pd.DataFrame:
        """Load enriched technical feature dataset for the symbol."""
        feat_path = self.storage_paths["features_technical"] / f"{symbol}_tech_features.parquet"
        if not feat_path.exists():
            raise FileNotFoundError(
                f"Feature dataset not found for '{symbol}' at: {feat_path}. "
                f"Please run 'psx features build --with-targets --symbols {symbol}' first."
            )
        return read_parquet(feat_path)

    def run_strategy(
        self,
        symbol: str,
        strategy: str = "sma_crossover",
        initial_cash: float = 1_000_000.0,
        risk_free_rate: float = 0.15,
        use_walk_forward: bool = False,
    ) -> tuple[SimulationResult, BacktestMetrics]:
        """Execute strategy simulation and compute comprehensive metrics.

        Args:
            symbol: Ticker symbol (e.g. 'OGDC').
            strategy: Strategy name ('sma_crossover', 'naive_persistence', 'logistic').
            initial_cash: Starting portfolio capital in PKR.
            risk_free_rate: Annual risk-free rate (e.g. 0.15 for 15% SBP policy rate).
            use_walk_forward: Whether to use walk-forward expanding folds for ML models.

        Returns:
            Tuple of (SimulationResult, BacktestMetrics).
        """
        logger.info(f"[{symbol}] Loading market and feature data for backtest...")
        df = self.load_dataset(symbol)

        clean_strat = strategy.lower().strip()
        signals = np.zeros(len(df), dtype=float)

        if clean_strat == "sma_crossover":
            logger.info(f"[{symbol}] Generating signals for SMA Crossover (20/50)...")
            if "sma_20" in df.columns and "sma_50" in df.columns:
                signals = np.where(df["sma_20"] > df["sma_50"], 1.0, 0.0)
            else:
                raise ValueError("Required SMA columns (sma_20, sma_50) missing from dataset.")

        elif clean_strat == "naive_persistence":
            logger.info(f"[{symbol}] Generating signals for Naive Persistence...")
            if "log_ret_1d" in df.columns:
                signals = np.where(df["log_ret_1d"] > 0, 1.0, 0.0)
            else:
                signals = np.where(df["close"].pct_change().fillna(0) > 0, 1.0, 0.0)

        elif clean_strat == "logistic":
            logger.info(f"[{symbol}] Generating walk-forward signals using Logistic Regression...")
            target_col = "target_next_day_dir"
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' missing from dataset.")

            feature_cols = [c for c in DEFAULT_TECHNICAL_FEATURES if c in df.columns]
            clean_df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

            partitioner = WalkForwardPartitioner(
                train_window=400,
                test_window=60,
                step_size=60,
                embargo=5,
                mode="expanding",
            )
            folds = partitioner.split(clean_df)
            logger.info(f"[{symbol}] Executing across {len(folds)} walk-forward folds...")

            # Collect out-of-sample predictions
            oos_signals = np.zeros(len(clean_df), dtype=float)
            for fold in folds:
                X_tr = clean_df.iloc[fold.train_indices][feature_cols]
                y_tr = clean_df.iloc[fold.train_indices][target_col]
                X_te = clean_df.iloc[fold.test_indices][feature_cols]

                clf = LogisticRegressionBaseline()
                clf.fit(X_tr, y_tr)
                preds = clf.predict(X_te)
                oos_signals[fold.test_indices] = preds

            # Use the clean_df evaluation slice
            eval_start = folds[0].test_indices[0]
            df = clean_df.iloc[eval_start:].reset_index(drop=True)
            signals = oos_signals[eval_start:]

        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Choose 'sma_crossover', 'naive_persistence', or 'logistic'."
            )

        logger.info(
            f"[{symbol}] Running execution simulator (Initial Cash: PKR {initial_cash:,.2f})..."
        )
        simulator = BacktestSimulator(
            initial_cash=initial_cash,
            cost_model=TransactionCostModel(),
        )

        sim_result = simulator.simulate(
            df=df,
            signals=signals,
            symbol=symbol,
            strategy_name=strategy.upper(),
        )

        metrics = calculate_backtest_metrics(
            result=sim_result,
            risk_free_rate=risk_free_rate,
        )

        return sim_result, metrics
