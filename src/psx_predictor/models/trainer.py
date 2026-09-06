"""Model training and evaluation orchestrator across benchmark models."""

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from psx_predictor.config.loader import load_config
from psx_predictor.models.base import BaseModel, ModelEvaluationResult
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


class ModelTrainer:
    """Orchestrates model training, out-of-sample evaluation, and checkpoint persistence."""

    def __init__(self, storage_paths: Optional[dict[str, Path]] = None) -> None:
        if storage_paths is None:
            config = load_config()
            self.storage_paths = ensure_directories(config.settings.data_dir)
        else:
            self.storage_paths = storage_paths

    def load_feature_dataset(self, symbol: str) -> pd.DataFrame:
        """Load technical feature dataset with targets for the given symbol."""
        feat_path = self.storage_paths["features_technical"] / f"{symbol}_tech_features.parquet"
        if not feat_path.exists():
            raise FileNotFoundError(
                f"Features dataset not found for '{symbol}' at: {feat_path}. "
                f"Please run 'psx features build --with-targets --symbols {symbol}' first."
            )
        return read_parquet(feat_path)

    def train_and_evaluate(
        self,
        symbol: str,
        model_type: str = "all",
        target_col: str = "target_next_day_dir",
        train_ratio: float = 0.8,
        save_models: bool = False,
    ) -> list[ModelEvaluationResult]:
        """Train and evaluate baseline and linear models on chronological split.

        Args:
            symbol: Ticker symbol (e.g. 'OGDC').
            model_type: 'baselines', 'logistic', 'random_forest', 'xgboost', 'trees', or 'all'.
            target_col: Prediction target column.
            train_ratio: Out-of-sample split ratio (default: 0.8).
            save_models: Whether to serialize fitted models to disk.

        Returns:
            List of ModelEvaluationResult for each evaluated model.
        """
        logger.info(f"[{symbol}] Loading features and prediction targets...")
        df = self.load_feature_dataset(symbol)

        logger.info(
            f"[{symbol}] Splitting {len(df)} sessions chronologically "
            f"(train_ratio={train_ratio})..."
        )
        split = chronological_train_test_split(
            df=df,
            target_col=target_col,
            train_ratio=train_ratio,
        )

        logger.info(
            f"[{symbol}] Train set: {len(split.X_train)} sessions "
            f"({split.train_dates.iloc[0]} -> {split.train_dates.iloc[-1]}), "
            f"Test set: {len(split.X_test)} sessions "
            f"({split.test_dates.iloc[0]} -> {split.test_dates.iloc[-1]})"
        )

        # Select models to evaluate
        m_type = model_type.lower().strip()
        models_to_run: list[BaseModel] = []
        if m_type in ("baselines", "all"):
            models_to_run.extend(
                [
                    MajorityClassifier(),
                    NaivePersistenceClassifier(),
                    SMACrossoverClassifier(),
                ]
            )

        if m_type in ("logistic", "all"):
            models_to_run.append(LogisticRegressionBaseline())

        if m_type in ("random_forest", "rf", "trees", "all"):
            models_to_run.append(RandomForestBaseline())

        if m_type in ("xgboost", "xgb", "trees", "all"):
            models_to_run.append(XGBoostBaseline())

        if not models_to_run:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                "Choose baselines, logistic, random_forest, xgboost, trees, or all."
            )

        results: list[ModelEvaluationResult] = []

        for model in models_to_run:
            logger.info(f"[{symbol}] Training {model.model_name}...")
            model.fit(split.X_train, split.y_train)

            # Predict on test set
            y_pred = model.predict(split.X_test)
            y_prob = model.predict_proba(split.X_test)

            eval_res = evaluate_classifier(
                y_true=split.y_test.to_numpy(),
                y_pred=y_pred,
                y_prob=y_prob,
                model_name=model.model_name,
                target_name=target_col,
                n_train=len(split.X_train),
                classes=model.classes_,
            )
            results.append(eval_res)

            if save_models:
                slug = (
                    model.model_name.lower()
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("/", "_")
                )
                model_path = self.storage_paths["models"] / symbol / f"{slug}.joblib"
                model.save(model_path)
                logger.info(f"[{symbol}] Saved model checkpoint to: {model_path}")

        return results
