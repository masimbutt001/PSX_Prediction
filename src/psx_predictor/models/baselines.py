"""Heuristic baseline benchmark models for market prediction."""

import numpy as np
import pandas as pd

from psx_predictor.models.base import BaseModel


class MajorityClassifier(BaseModel):
    """Benchmark classifier that always predicts the most frequent class in training data."""

    def __init__(self) -> None:
        super().__init__(model_name="Majority Class Baseline")
        self.majority_class_: float = 0.0
        self.class_priors_: np.ndarray = np.array([])

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "MajorityClassifier":
        y_arr = np.asarray(y)
        self.classes_, counts = np.unique(y_arr, return_counts=True)
        max_idx = int(np.argmax(counts))
        self.majority_class_ = float(self.classes_[max_idx])
        self.class_priors_ = counts / len(y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        n_samples = len(X)
        return np.full(n_samples, self.majority_class_, dtype=float)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        n_samples = len(X)
        return np.tile(self.class_priors_, (n_samples, 1))


class NaivePersistenceClassifier(BaseModel):
    """Benchmark classifier predicting price moves in the same direction as recent return."""

    def __init__(self, return_col: str = "log_ret_1d") -> None:
        super().__init__(model_name="Naive Persistence Baseline")
        self.return_col = return_col

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> "NaivePersistenceClassifier":
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")

        # Extract 1-day return feature
        if isinstance(X, pd.DataFrame) and self.return_col in X.columns:
            ret = X[self.return_col].to_numpy()
        elif isinstance(X, pd.DataFrame) and len(X.columns) > 0:
            ret = X.iloc[:, 0].to_numpy()
        elif isinstance(X, np.ndarray):
            ret = X[:, 0]
        else:
            raise ValueError("Unsupported input format for NaivePersistenceClassifier.")

        preds = np.zeros(len(ret), dtype=float)
        # Binary target: 1 if return > 0 else 0
        if set(self.classes_) == {0.0, 1.0} or len(self.classes_) == 2:
            preds[ret > 0] = 1.0
            preds[ret <= 0] = 0.0
        # 3-class target: 1 if > 0.0075, -1 if < -0.0075, else 0
        elif set(self.classes_) == {-1.0, 0.0, 1.0} or len(self.classes_) == 3:
            preds[ret > 0.0075] = 1.0
            preds[ret < -0.0075] = -1.0
            preds[(ret >= -0.0075) & (ret <= 0.0075)] = 0.0
        else:
            preds[ret > 0] = 1.0

        return preds

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        n_samples = len(preds)
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes), dtype=float)

        for i, c in enumerate(self.classes_):
            proba[preds == c, i] = 1.0

        # Smooth degenerate probabilities slightly for calibration metrics
        eps = 1e-4
        proba = np.clip(proba, eps, 1.0 - eps)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba


class SMACrossoverClassifier(BaseModel):
    """Trend-following baseline predicting Up (1.0) when SMA(20) > SMA(50), else Down (0.0)."""

    def __init__(self, fast_col: str = "sma_20", slow_col: str = "sma_50") -> None:
        super().__init__(model_name="SMA Crossover Baseline (20/50)")
        self.fast_col = fast_col
        self.slow_col = slow_col

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
    ) -> "SMACrossoverClassifier":
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")

        if isinstance(X, pd.DataFrame):
            if self.fast_col in X.columns and self.slow_col in X.columns:
                fast = X[self.fast_col].to_numpy()
                slow = X[self.slow_col].to_numpy()
            elif "price_to_sma_50" in X.columns:
                # Alternative proxy: price above SMA 50
                diff = X["price_to_sma_50"].to_numpy()
                return np.where(diff > 0, 1.0, 0.0)
            else:
                fast = X.iloc[:, 0].to_numpy()
                slow = X.iloc[:, 1].to_numpy() if X.shape[1] > 1 else np.zeros_like(fast)
        elif isinstance(X, np.ndarray):
            fast = X[:, 0]
            slow = X[:, 1] if X.shape[1] > 1 else np.zeros_like(fast)
        else:
            raise ValueError("Unsupported input format for SMACrossoverClassifier.")

        preds = np.where(fast > slow, 1.0, 0.0)
        return preds.astype(float)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        n_samples = len(preds)
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes), dtype=float)

        for i, c in enumerate(self.classes_):
            proba[preds == c, i] = 1.0

        eps = 1e-4
        proba = np.clip(proba, eps, 1.0 - eps)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba
