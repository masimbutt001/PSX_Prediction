"""Tree-based machine learning models: Random Forest and XGBoost with early stopping."""

from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from psx_predictor.models.base import BaseModel


class RandomForestBaseline(BaseModel):
    """Random Forest classifier with controlled tree depth and leaf regularization.

    Regularization parameters to prevent overfitting on noisy financial data:
        - max_depth=5
        - min_samples_leaf=20
        - n_estimators=100
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        super().__init__(model_name="Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.clf_: Optional[RandomForestClassifier] = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "RandomForestBaseline":
        """Fit Random Forest on training partition."""
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_mat = X.to_numpy()
        else:
            self.feature_names_ = [f"feat_{i}" for i in range(X.shape[1])]
            X_mat = np.asarray(X)

        self.clf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.clf_.fit(X_mat, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.clf_ is None:
            raise ValueError("Model is not fitted yet.")
        X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.clf_.predict(X_mat)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.clf_ is None:
            raise ValueError("Model is not fitted yet.")
        X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.clf_.predict_proba(X_mat)

    def get_feature_importances(self) -> dict[str, float]:
        """Extract Gini feature importances sorted descending."""
        if not self.is_fitted or self.clf_ is None:
            raise ValueError("Model is not fitted yet.")
        importances = self.clf_.feature_importances_
        imp_dict = {name: float(importances[i]) for i, name in enumerate(self.feature_names_)}
        return dict(sorted(imp_dict.items(), key=lambda item: item[1], reverse=True))


class XGBoostBaseline(BaseModel):
    """XGBoost classifier with chronological internal validation and early stopping.

    Guarantees:
        1. Internal validation set for early stopping is strictly posterior to training data.
        2. Regularized parameters: learning_rate=0.03, max_depth=4, subsample=0.8,
           colsample_bytree=0.8.
        3. Supports binary and multiclass target formulations automatically.
    """

    def __init__(
        self,
        learning_rate: float = 0.03,
        max_depth: int = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_estimators: int = 200,
        early_stopping_rounds: int = 20,
        val_ratio: float = 0.10,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        super().__init__(model_name="XGBoost")
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.clf_: Optional[xgb.XGBClassifier] = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "XGBoostBaseline":
        """Fit XGBoost model with chronological early stopping."""
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_mat = X.to_numpy()
        else:
            self.feature_names_ = [f"feat_{i}" for i in range(X.shape[1])]
            X_mat = np.asarray(X)

        # Label encoding for XGBoost if classes are negative (e.g. -1, 0, 1)
        # Map classes to 0, 1, ...
        self._class_to_code = {c: i for i, c in enumerate(self.classes_)}
        self._code_to_class = {i: c for i, c in enumerate(self.classes_)}
        y_encoded = np.array([self._class_to_code[val] for val in y_arr])

        # Chronological validation split for early stopping
        n_samples = len(X_mat)
        val_size = int(np.floor(n_samples * self.val_ratio))

        if val_size >= 10 and self.early_stopping_rounds > 0:
            split_idx = n_samples - val_size
            X_tr, y_tr = X_mat[:split_idx], y_encoded[:split_idx]
            X_val, y_val = X_mat[split_idx:], y_encoded[split_idx:]
            eval_set = [(X_val, y_val)]
            es_rounds = self.early_stopping_rounds
        else:
            X_tr, y_tr = X_mat, y_encoded
            eval_set = None
            es_rounds = None

        is_binary = len(self.classes_) == 2
        objective = "binary:logistic" if is_binary else "multi:softprob"
        eval_metric = "logloss" if is_binary else "mlogloss"

        xgb_params: dict[str, Any] = {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "n_estimators": self.n_estimators,
            "objective": objective,
            "eval_metric": eval_metric,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }
        if es_rounds is not None:
            xgb_params["early_stopping_rounds"] = es_rounds

        self.clf_ = xgb.XGBClassifier(**xgb_params)

        if eval_set is not None:
            self.clf_.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
        else:
            self.clf_.fit(X_tr, y_tr, verbose=False)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.clf_ is None:
            raise ValueError("Model is not fitted yet.")
        X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        preds_encoded = self.clf_.predict(X_mat)
        # Decode back to original classes
        return np.array([self._code_to_class[code] for code in preds_encoded], dtype=float)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.clf_ is None:
            raise ValueError("Model is not fitted yet.")
        X_mat = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self.clf_.predict_proba(X_mat)

    def get_feature_importances(self) -> dict[str, float]:
        """Extract XGBoost feature importances sorted descending."""
        if not self.is_fitted or self.clf_ is None:
            raise ValueError("Model is not fitted yet.")
        importances = self.clf_.feature_importances_
        imp_dict = {name: float(importances[i]) for i, name in enumerate(self.feature_names_)}
        return dict(sorted(imp_dict.items(), key=lambda item: item[1], reverse=True))
