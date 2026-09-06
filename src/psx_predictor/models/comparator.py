"""Multi-model benchmarking comparator evaluating ML and financial simulation metrics."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from psx_predictor.backtesting.metrics import calculate_backtest_metrics
from psx_predictor.backtesting.simulator import BacktestSimulator, TransactionCostModel
from psx_predictor.config.loader import load_config
from psx_predictor.models.base import BaseModel
from psx_predictor.models.baselines import (
    MajorityClassifier,
    NaivePersistenceClassifier,
    SMACrossoverClassifier,
)
from psx_predictor.models.evaluation import evaluate_classifier
from psx_predictor.models.linear import LogisticRegressionBaseline
from psx_predictor.models.split import chronological_train_test_split
from psx_predictor.models.trees import RandomForestBaseline, XGBoostBaseline
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import ensure_directories


@dataclass(frozen=True)
class ModelBenchmarkSummary:
    """Consolidated performance profile pairing ML and financial simulation metrics."""

    model_name: str
    target_name: str
    n_train: int
    n_test: int
    accuracy: float
    f1_macro: float
    roc_auc: float | None
    brier_score: float | None
    total_return: float
    cagr: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "target_name": self.target_name,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "accuracy": round(self.accuracy, 4),
            "f1_macro": round(self.f1_macro, 4),
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "brier_score": round(self.brier_score, 4) if self.brier_score is not None else None,
            "total_return": round(self.total_return, 4),
            "cagr": round(self.cagr, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "total_trades": self.total_trades,
        }


class ModelComparator:
    """Benchmarks all candidate models against identical chronological out-of-sample data."""

    def __init__(self, storage_paths: Optional[dict[str, Path]] = None) -> None:
        if storage_paths is None:
            config = load_config()
            self.storage_paths = ensure_directories(config.settings.data_dir)
        else:
            self.storage_paths = storage_paths

    def compare_universe(
        self,
        symbol: str,
        target_col: str = "target_next_day_dir",
        train_ratio: float = 0.8,
        initial_cash: float = 1_000_000.0,
        risk_free_rate: float = 0.15,
    ) -> list[ModelBenchmarkSummary]:
        """Train and benchmark all models on identical test partition."""
        feat_path = self.storage_paths["features_technical"] / f"{symbol}_tech_features.parquet"
        if not feat_path.exists():
            raise FileNotFoundError(f"Feature dataset not found for '{symbol}' at: {feat_path}")

        df = read_parquet(feat_path)
        logger.info(f"[{symbol}] Splitting {len(df)} records chronologically...")
        split = chronological_train_test_split(
            df=df,
            target_col=target_col,
            train_ratio=train_ratio,
        )

        candidate_models: list[BaseModel] = [
            MajorityClassifier(),
            NaivePersistenceClassifier(),
            SMACrossoverClassifier(),
            LogisticRegressionBaseline(),
            RandomForestBaseline(),
            XGBoostBaseline(),
        ]

        # Extract test price dataframe for simulation
        # Need 'close' and circuit lock columns for the test window
        clean_df = df.dropna(subset=split.feature_names + [target_col]).reset_index(drop=True)
        split_idx = len(split.X_train)
        test_df = clean_df.iloc[split_idx:].reset_index(drop=True)

        simulator = BacktestSimulator(
            initial_cash=initial_cash,
            cost_model=TransactionCostModel(),
        )

        summaries: list[ModelBenchmarkSummary] = []

        for model in candidate_models:
            logger.info(f"[{symbol}] Fitting {model.model_name}...")
            model.fit(split.X_train, split.y_train)

            # Predictions and probabilities
            y_pred = model.predict(split.X_test)
            y_prob = model.predict_proba(split.X_test)

            # 1. Classification metrics
            clf_metrics = evaluate_classifier(
                y_true=split.y_test.to_numpy(),
                y_pred=y_pred,
                y_prob=y_prob,
                model_name=model.model_name,
                target_name=target_col,
                n_train=len(split.X_train),
                classes=model.classes_,
            )

            # 2. Trading simulation metrics
            # Use binary signal: 1.0 if predicted UP else 0.0
            binary_signals = np.where(y_pred > 0, 1.0, 0.0)
            sim_res = simulator.simulate(
                df=test_df,
                signals=binary_signals,
                symbol=symbol,
                strategy_name=model.model_name,
            )
            trade_metrics = calculate_backtest_metrics(
                result=sim_res,
                risk_free_rate=risk_free_rate,
            )

            summary = ModelBenchmarkSummary(
                model_name=model.model_name,
                target_name=target_col,
                n_train=len(split.X_train),
                n_test=len(split.X_test),
                accuracy=clf_metrics.accuracy,
                f1_macro=clf_metrics.f1_macro,
                roc_auc=clf_metrics.roc_auc,
                brier_score=clf_metrics.brier_score,
                total_return=trade_metrics.total_return,
                cagr=trade_metrics.cagr,
                sharpe_ratio=trade_metrics.sharpe_ratio,
                max_drawdown=trade_metrics.max_drawdown,
                win_rate=trade_metrics.win_rate,
                total_trades=trade_metrics.total_trades,
            )
            summaries.append(summary)

        return summaries


def display_comparison_table(
    summaries: list[ModelBenchmarkSummary],
    symbol: str = "",
    console: Optional[Console] = None,
) -> None:
    """Render a comprehensive Rich comparison table."""
    c = console or Console()

    target_name = summaries[0].target_name if summaries else "target"
    n_tr = summaries[0].n_train if summaries else 0
    n_te = summaries[0].n_test if summaries else 0

    title = (
        f"Comprehensive Model Benchmark Comparison: {symbol} "
        f"({target_name} | Train: {n_tr} | Test: {n_te})"
        if symbol
        else "Model Benchmark Comparison"
    )

    table = Table(
        title=title,
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Model", style="bold white", justify="left", min_width=18)
    table.add_column("Acc", style="bold green", justify="right", no_wrap=True)
    table.add_column("F1", style="bold yellow", justify="right", no_wrap=True)
    table.add_column("AUC", style="bold magenta", justify="right", no_wrap=True)
    table.add_column("Return", justify="right", no_wrap=True)
    table.add_column("CAGR", justify="right", no_wrap=True)
    table.add_column("Sharpe", justify="right", no_wrap=True)
    table.add_column("Max DD", style="red", justify="right", no_wrap=True)
    table.add_column("Win%", justify="right", no_wrap=True)

    for s in summaries:
        roc_str = f"{s.roc_auc:.4f}" if s.roc_auc is not None else "N/A"
        ret_color = "green" if s.total_return >= 0 else "red"
        ret_str = f"[{ret_color}]{s.total_return:+.1%}[/{ret_color}]"
        cagr_str = f"[{ret_color}]{s.cagr:+.1%}[/{ret_color}]"
        sharpe_color = "green" if s.sharpe_ratio > 0 else "red"
        sharpe_str = f"[{sharpe_color}]{s.sharpe_ratio:.2f}[/{sharpe_color}]"

        table.add_row(
            s.model_name,
            f"{s.accuracy:.1%}",
            f"{s.f1_macro:.3f}",
            roc_str,
            ret_str,
            cagr_str,
            sharpe_str,
            f"{s.max_drawdown:.1%}",
            f"{s.win_rate:.0%}",
        )

    c.print()
    c.print(table)
    c.print(
        Panel(
            "[dim]Evaluation: All models trained on identical historical sessions and "
            "simulated on identical out-of-sample test window with full PSX frictions "
            "(26.5 bps per side) and circuit breaker fill constraints.[/dim]",
            border_style="dim",
        )
    )
