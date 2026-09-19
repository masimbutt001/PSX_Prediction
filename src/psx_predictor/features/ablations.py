"""Ablation study runner evaluating empirical contributions of news and macro modalities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table

from psx_predictor.backtesting.metrics import calculate_backtest_metrics
from psx_predictor.backtesting.simulator import BacktestSimulator, TransactionCostModel
from psx_predictor.config.loader import load_config
from psx_predictor.features.merger import (
    MACRO_FEATURE_COLUMNS,
    NEWS_FEATURE_COLUMNS,
    MultiModalFeatureMerger,
)
from psx_predictor.models.base import BaseModel
from psx_predictor.models.evaluation import evaluate_classifier
from psx_predictor.models.linear import LogisticRegressionBaseline
from psx_predictor.models.split import DEFAULT_TECHNICAL_FEATURES
from psx_predictor.models.trees import RandomForestBaseline, XGBoostBaseline
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import ensure_directories


@dataclass(frozen=True)
class AblationConfig:
    """Specification of an ablation feature group."""

    config_id: str
    name: str
    description: str
    feature_columns: list[str]


@dataclass(frozen=True)
class AblationExperimentResult:
    """Consolidated metrics from evaluating an ablation configuration."""

    config_id: str
    config_name: str
    model_name: str
    target_name: str
    feature_count: int
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
    delta_accuracy: float
    delta_sharpe: float
    delta_total_return: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "config_name": self.config_name,
            "model_name": self.model_name,
            "target_name": self.target_name,
            "feature_count": self.feature_count,
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
            "delta_accuracy": round(self.delta_accuracy, 4),
            "delta_sharpe": round(self.delta_sharpe, 4),
            "delta_total_return": round(self.delta_total_return, 4),
        }


class AblationStudyRunner:
    """Orchestrates empirical ablation experiments across multi-modal feature groups."""

    def __init__(self, storage_paths: Optional[dict[str, Path]] = None) -> None:
        """Initialize ablation runner with resolved analytical storage paths."""
        if storage_paths is None:
            config = load_config()
            self.storage_paths = ensure_directories(config.settings.data_dir)
        else:
            self.storage_paths = storage_paths

        self.merger = MultiModalFeatureMerger(self.storage_paths)

    def get_standard_ablation_configs(
        self, available_columns: list[str]
    ) -> list[AblationConfig]:
        """Generate the 5 standardized ablation configurations filtered by available columns."""
        avail_set = set(available_columns)

        # 1. Price Only: technical indicators
        tech_cols = [c for c in DEFAULT_TECHNICAL_FEATURES if c in avail_set]

        # 2. Price + Relative Market (KSE-100)
        market_cols = [
            c
            for c in ["kse100_index", "kse100_return_1d", "rel_kse100_ret_1d"]
            if c in avail_set
        ]
        price_market_cols = sorted(list(set(tech_cols + market_cols)))

        # 3. Price + News Sentiment & Disclosures
        news_cols = [c for c in NEWS_FEATURE_COLUMNS if c in avail_set]
        price_news_cols = sorted(list(set(tech_cols + news_cols)))

        # 4. Price + Macroeconomic series & spreads
        macro_cols = [
            c
            for c in MACRO_FEATURE_COLUMNS
            + ["policy_rate_spread", "real_interest_rate", "rel_kse100_ret_1d"]
            if c in avail_set
        ]
        price_macro_cols = sorted(list(set(tech_cols + macro_cols)))

        # 5. Full Multi-Modal (Price + Relative Market + News + Macro)
        all_cols = sorted(list(set(tech_cols + market_cols + news_cols + macro_cols)))

        return [
            AblationConfig(
                config_id="EXP_1",
                name="Price Only",
                description="Vectorized OHLCV technical indicators alone",
                feature_columns=tech_cols,
            ),
            AblationConfig(
                config_id="EXP_2",
                name="Price + Relative Market",
                description="Technical features + KSE-100 index return & relative alpha",
                feature_columns=price_market_cols,
            ),
            AblationConfig(
                config_id="EXP_3",
                name="Price + News",
                description="Technical features + News sentiment polarity & announcement flags",
                feature_columns=price_news_cols,
            ),
            AblationConfig(
                config_id="EXP_4",
                name="Price + Macro",
                description="Technical features + SBP policy rates, KIBOR, CPI, FX, Oil, & spreads",
                feature_columns=price_macro_cols,
            ),
            AblationConfig(
                config_id="EXP_5",
                name="Price + News + Macro (Full)",
                description="Full multi-modal feature set combining all layers",
                feature_columns=all_cols,
            ),
        ]

    def _instantiate_model(self, model_name: str) -> BaseModel:
        """Instantiate predictive model baseline for ablation testing."""
        name = model_name.lower().strip()
        if name in ("logistic", "logreg", "lr"):
            return LogisticRegressionBaseline()
        elif name in ("random_forest", "rf", "forest"):
            return RandomForestBaseline(n_estimators=100, random_state=42)
        elif name in ("xgboost", "xgb"):
            return XGBoostBaseline(n_estimators=100, random_state=42)
        else:
            raise ValueError(
                f"Unsupported model_name '{model_name}'. Choose 'logistic', 'rf', or 'xgb'."
            )

    def run_ablation_study(
        self,
        symbol: str,
        model_name: str = "logistic",
        target_col: str = "target_next_day_dir",
        train_ratio: float = 0.8,
        risk_free_rate: float = 0.15,
        initial_cash: float = 1_000_000.0,
    ) -> list[AblationExperimentResult]:
        """Execute 5-configuration empirical ablation study on identical chronological folds.

        Args:
            symbol: Ticker symbol (e.g. 'OGDC').
            model_name: Baseline learner ('logistic', 'rf', or 'xgb').
            target_col: Supervised classification target column.
            train_ratio: Chronological split ratio.
            risk_free_rate: Annualized risk-free rate for Sharpe calculation.
            initial_cash: Starting portfolio cash.

        Returns:
            List of AblationExperimentResult comparing each configuration.
        """
        clean_sym = symbol.strip().upper()
        logger.info(
            f"[{clean_sym}] Starting ablation study with {model_name} on {target_col}..."
        )

        # 1. Load or build combined multi-modal dataset
        combined_path = (
            self.storage_paths["features_combined"] / f"{clean_sym}_combined.parquet"
        )
        if not combined_path.exists():
            logger.info(f"[{clean_sym}] Combined features not found. Merging modalities...")
            df = self.merger.merge_symbol(clean_sym, save=True)
        else:
            df = read_parquet(combined_path)

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in combined features.")

        # 2. Get standardized configurations, filtering out columns with all/mostly NaNs
        valid_cols = [c for c in df.columns if df[c].notna().sum() >= 20]
        configs = self.get_standard_ablation_configs(valid_cols)

        # 3. Clean dataset to ensure identical sample size across all configurations
        active_cols = sorted(list({c for cfg in configs for c in cfg.feature_columns}))
        clean_df = df.dropna(subset=[target_col] + active_cols).reset_index(drop=True)

        if "trade_date" in clean_df.columns:
            clean_df = clean_df.sort_values("trade_date").reset_index(drop=True)
        elif "session_date" in clean_df.columns:
            clean_df = clean_df.sort_values("session_date").reset_index(drop=True)

        total_sessions = len(clean_df)
        split_idx = int(total_sessions * train_ratio)
        if split_idx < 10 or (total_sessions - split_idx) < 10:
            raise ValueError(
                f"Insufficient sessions ({total_sessions}) for chronological train/test split."
            )

        test_df = clean_df.iloc[split_idx:].reset_index(drop=True)
        y_train = clean_df[target_col].iloc[:split_idx].to_numpy()
        y_test = clean_df[target_col].iloc[split_idx:].to_numpy()

        simulator = BacktestSimulator(
            initial_cash=initial_cash,
            cost_model=TransactionCostModel(),
        )

        results: list[AblationExperimentResult] = []
        baseline_acc: Optional[float] = None
        baseline_sharpe: Optional[float] = None
        baseline_return: Optional[float] = None

        for cfg in configs:
            logger.info(
                f"[{clean_sym}] Running {cfg.name} with {len(cfg.feature_columns)} features..."
            )
            X_train = clean_df[cfg.feature_columns].iloc[:split_idx]
            X_test = clean_df[cfg.feature_columns].iloc[split_idx:]

            model = self._instantiate_model(model_name)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)

            # ML evaluation
            clf_metrics = evaluate_classifier(
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                model_name=f"{model_name}_{cfg.config_id}",
                target_name=target_col,
                n_train=len(X_train),
                classes=model.classes_,
            )

            # Backtest simulation
            binary_signals = np.where(y_pred > 0, 1.0, 0.0)
            sim_res = simulator.simulate(
                df=test_df,
                signals=binary_signals,
                symbol=clean_sym,
                strategy_name=cfg.name,
            )
            trade_metrics = calculate_backtest_metrics(
                result=sim_res,
                risk_free_rate=risk_free_rate,
            )

            # Deltas relative to Price Only baseline
            if (
                baseline_acc is None
                or baseline_sharpe is None
                or baseline_return is None
            ):
                baseline_acc = clf_metrics.accuracy
                baseline_sharpe = trade_metrics.sharpe_ratio
                baseline_return = trade_metrics.total_return
                delta_acc = 0.0
                delta_sharpe = 0.0
                delta_ret = 0.0
            else:
                delta_acc = clf_metrics.accuracy - baseline_acc
                delta_sharpe = trade_metrics.sharpe_ratio - baseline_sharpe
                delta_ret = trade_metrics.total_return - baseline_return

            res = AblationExperimentResult(
                config_id=cfg.config_id,
                config_name=cfg.name,
                model_name=model_name,
                target_name=target_col,
                feature_count=len(cfg.feature_columns),
                n_train=len(X_train),
                n_test=len(X_test),
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
                delta_accuracy=delta_acc,
                delta_sharpe=delta_sharpe,
                delta_total_return=delta_ret,
            )
            results.append(res)

        return results

    def display_ablation_table(
        self,
        results: list[AblationExperimentResult],
        symbol: str = "",
        console: Optional[Console] = None,
    ) -> None:
        """Display an empirical ablation comparison table with rich formatting."""
        if not results:
            return

        out = console or Console()
        m_name = results[0].model_name.upper()
        t_name = results[0].target_name

        table = Table(
            title=f"Empirical Multi-Modal Ablation Study: {symbol} ({m_name} on {t_name})",
            box=box.ROUNDED,
            header_style="bold cyan",
        )

        table.add_column("Configuration", style="bold", min_width=20)
        table.add_column("Feats", justify="right")
        table.add_column("Acc", justify="right")
        table.add_column("+/-Acc", justify="right")
        table.add_column("F1", justify="right")
        table.add_column("AUC", justify="right")
        table.add_column("Return", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("+/-Sh", justify="right")
        table.add_column("Win%", justify="right")

        for r in results:
            acc_str = f"{r.accuracy * 100:.1f}%"
            d_acc = r.delta_accuracy * 100
            if r.config_id == "EXP_1":
                d_acc_str = "[dim]--[/dim]"
                d_sharpe_str = "[dim]--[/dim]"
            else:
                d_acc_color = "green" if d_acc > 0 else ("red" if d_acc < 0 else "dim")
                d_acc_str = f"[{d_acc_color}]{d_acc:+.1f}%[/{d_acc_color}]"

                d_sh = r.delta_sharpe
                d_sh_color = "green" if d_sh > 0 else ("red" if d_sh < 0 else "dim")
                d_sharpe_str = f"[{d_sh_color}]{d_sh:+.2f}[/{d_sh_color}]"

            f1_str = f"{r.f1_macro:.3f}"
            auc_str = f"{r.roc_auc:.3f}" if r.roc_auc is not None else "--"
            ret_str = f"{r.total_return * 100:.1f}%"
            ret_color = "green" if r.total_return >= 0 else "red"
            ret_styled = f"[{ret_color}]{ret_str}[/{ret_color}]"

            sharpe_str = f"{r.sharpe_ratio:.2f}"
            win_str = f"{r.win_rate * 100:.1f}%"

            table.add_row(
                r.config_name,
                str(r.feature_count),
                acc_str,
                d_acc_str,
                f1_str,
                auc_str,
                ret_styled,
                sharpe_str,
                d_sharpe_str,
                win_str,
            )

        out.print(table)
