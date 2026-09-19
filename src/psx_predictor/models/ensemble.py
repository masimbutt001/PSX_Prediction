"""Multi-modal stacking ensemble fusing price, news sentiment, and macro sub-models."""

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression

from psx_predictor.features.merger import (
    INTERACTION_FEATURE_COLUMNS,
    MACRO_FEATURE_COLUMNS,
    NEWS_FEATURE_COLUMNS,
)
from psx_predictor.models.base import BaseModel
from psx_predictor.models.linear import LogisticRegressionBaseline
from psx_predictor.models.split import DEFAULT_TECHNICAL_FEATURES
from psx_predictor.models.trees import RandomForestBaseline, XGBoostBaseline
from psx_predictor.models.weighting import OutOfFoldWeightOptimizer


class MultiModalStackingEnsemble(BaseModel):
    """Stacking meta-learner combining specialized Price, News, and Macro sub-models.

    Architecture:
      - Price Specialist: Tree-based model (XGBoost or Random Forest) on technical indicators.
      - News Specialist: Calibrated model on news sentiment polarity and corporate disclosures.
      - Macro Specialist: Regularized model on SBP rates, KIBOR, CPI, FX, Oil, and spreads.
      - Stacking Meta-Learner: Fitted strictly on out-of-fold (OOF) cross-validation predictions.
    """

    def __init__(
        self,
        model_name: str = "MultiModalStackingEnsemble",
        price_model_type: str = "xgboost",
        n_splits: int = 4,
        meta_learner_type: str = "logistic",  # 'logistic' or 'weighted_average'
    ) -> None:
        """Initialize ensemble with specified sub-model architectures.

        Args:
            model_name: Model identifier string.
            price_model_type: 'xgboost' or 'random_forest'.
            n_splits: Number of out-of-fold cross-validation folds.
            meta_learner_type: 'logistic' (logistic stacking) or 'weighted_average'.
        """
        super().__init__(model_name=model_name)
        self.price_model_type = price_model_type.lower()
        self.n_splits = n_splits
        self.meta_learner_type = meta_learner_type.lower()

        # Initialize sub-models
        if self.price_model_type in ("xgb", "xgboost"):
            self.price_model: BaseModel = XGBoostBaseline(n_estimators=100, random_state=42)
        else:
            self.price_model = RandomForestBaseline(n_estimators=100, random_state=42)

        self.news_model: BaseModel = LogisticRegressionBaseline(C=1.0)
        self.macro_model: BaseModel = LogisticRegressionBaseline(C=1.0)

        # Meta-learner
        self.meta_model = LogisticRegression(C=1.0, max_iter=1000)
        self.optimizer = OutOfFoldWeightOptimizer(n_splits=n_splits)

        # Learned weights and column mappings
        self.modality_weights_: dict[str, float] = {}
        self.tech_cols_: list[str] = []
        self.news_cols_: list[str] = []
        self.macro_cols_: list[str] = []
        self.sub_model_names_: list[str] = ["price", "news", "macro"]

    def _partition_features(self, X: pd.DataFrame) -> None:
        """Identify available columns for each information modality."""
        avail_cols = set(X.columns)

        self.tech_cols_ = [
            c for c in DEFAULT_TECHNICAL_FEATURES if c in avail_cols and X[c].notna().sum() >= 20
        ]
        self.news_cols_ = [c for c in NEWS_FEATURE_COLUMNS if c in avail_cols]
        self.macro_cols_ = [
            c
            for c in MACRO_FEATURE_COLUMNS + INTERACTION_FEATURE_COLUMNS
            if c in avail_cols and X[c].notna().sum() >= 20
        ]

        # Fallbacks if columns are missing or dataset has generic names
        if not self.tech_cols_:
            self.tech_cols_ = [
                c for c in X.columns if c not in self.news_cols_ and c not in self.macro_cols_
            ]

        if not self.news_cols_:
            self.news_cols_ = self.tech_cols_

        if not self.macro_cols_:
            self.macro_cols_ = self.tech_cols_

        logger.info(
            f"Partitioned features: {len(self.tech_cols_)} Tech, "
            f"{len(self.news_cols_)} News, {len(self.macro_cols_)} Macro."
        )

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "BaseModel":
        """Fit stacking meta-learner using out-of-fold validation on base estimators.

        Args:
            X: Training features (DataFrame preferred with modality columns).
            y: Target binary direction vector.

        Returns:
            Fitted MultiModalStackingEnsemble instance.
        """
        if isinstance(X, np.ndarray):
            feature_names = [f"feat_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()

        y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.array(y)
        self.classes_ = np.unique(y_arr)
        self.feature_names_ = X_df.columns.tolist()

        # 1. Partition modality feature subsets
        self._partition_features(X_df)

        base_models_spec: dict[str, tuple[BaseModel, list[str]]] = {
            "price": (self.price_model, self.tech_cols_),
            "news": (self.news_model, self.news_cols_),
            "macro": (self.macro_model, self.macro_cols_),
        }

        # 2. Compute out-of-fold predictions to train meta-learner
        logger.info("Step 1/3: Generating out-of-fold cross-validation predictions...")
        Z_oof, y_oof, sub_names = self.optimizer.compute_oof_predictions(
            base_models=base_models_spec,
            X=X_df,
            y=y_arr,
        )
        self.sub_model_names_ = sub_names

        # 3. Optimize modality weights
        logger.info("Step 2/3: Optimizing stacking modality weights on OOF predictions...")
        optimal_weights = self.optimizer.optimize_weights(Z_oof=Z_oof, y_oof=y_oof)
        self.modality_weights_ = {
            name: round(float(w), 4) for name, w in zip(sub_names, optimal_weights, strict=False)
        }
        logger.success(f"Optimized Modality Weights: {self.modality_weights_}")

        # Fit logistic meta-learner on OOF predictions
        self.meta_model.fit(Z_oof, y_oof)

        # 4. Refit base estimators on full training dataset
        logger.info("Step 3/3: Refitting base specialist models on full training dataset...")
        self.price_model.fit(X_df[self.tech_cols_], y_arr)
        self.news_model.fit(X_df[self.news_cols_], y_arr)
        self.macro_model.fit(X_df[self.macro_cols_], y_arr)

        self.is_fitted = True
        return self

    def _get_base_probabilities(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute base specialist model probabilities on input feature matrix."""
        if not self.is_fitted:
            raise ValueError(f"Model '{self.model_name}' must be fitted before inference.")

        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names_)
        else:
            X_df = X

        # Predict probabilities from each specialist
        p_price_all = self.price_model.predict_proba(X_df[self.tech_cols_])
        p_news_all = self.news_model.predict_proba(X_df[self.news_cols_])
        p_macro_all = self.macro_model.predict_proba(X_df[self.macro_cols_])

        p_price = p_price_all[:, 1] if p_price_all.ndim == 2 else p_price_all.ravel()
        p_news = p_news_all[:, 1] if p_news_all.ndim == 2 else p_news_all.ravel()
        p_macro = p_macro_all[:, 1] if p_macro_all.ndim == 2 else p_macro_all.ravel()

        return np.column_stack([p_price, p_news, p_macro])

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probability distribution of shape (n_samples, 2).

        Args:
            X: Input feature matrix.

        Returns:
            Array of shape (n_samples, 2) containing [P(y=0), P(y=1)].
        """
        Z = self._get_base_probabilities(X)

        if self.meta_learner_type == "weighted_average":
            weights = np.array(
                [self.modality_weights_.get(name, 1.0 / 3.0) for name in self.sub_model_names_]
            )
            weights = weights / np.sum(weights)
            p_pos = np.dot(Z, weights)
            p_pos = np.clip(p_pos, 1e-6, 1.0 - 1e-6)
            return np.column_stack([1.0 - p_pos, p_pos])
        else:
            # Logistic stacking meta-learner
            probs = self.meta_model.predict_proba(Z)
            return probs

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict discrete binary class labels (0 or 1).

        Args:
            X: Input feature matrix.

        Returns:
            Predicted class array of shape (n_samples,).
        """
        probs = self.predict_proba(X)
        if probs.shape[1] >= 2:
            return np.where(probs[:, 1] >= 0.5, 1, 0)
        return (probs.ravel() >= 0.5).astype(int)

    def get_modality_contributions(self) -> dict[str, Any]:
        """Return human-readable metadata regarding learned ensemble weights."""
        return {
            "model_name": self.model_name,
            "meta_learner_type": self.meta_learner_type,
            "price_model": self.price_model.model_name,
            "news_model": self.news_model.model_name,
            "macro_model": self.macro_model.model_name,
            "weights": self.modality_weights_,
            "feature_counts": {
                "technical": len(self.tech_cols_),
                "news": len(self.news_cols_),
                "macro": len(self.macro_cols_),
            },
        }
