"""Base model interface and evaluation data structures for PSX prediction models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Standardized performance metrics container for classification models."""

    model_name: str
    target_name: str
    n_train: int
    n_test: int
    accuracy: float
    precision: float
    recall: float
    f1_macro: float
    roc_auc: float | None = None
    brier_score: float | None = None
    confusion_matrix: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation metrics to dictionary."""
        return {
            "model_name": self.model_name,
            "target_name": self.target_name,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_macro": round(self.f1_macro, 4),
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "brier_score": round(self.brier_score, 4) if self.brier_score is not None else None,
            "confusion_matrix": self.confusion_matrix,
        }


class BaseModel(ABC):
    """Abstract base class for all PSX predictive models."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.is_fitted: bool = False
        self.classes_: np.ndarray = np.array([])
        self.feature_names_: list[str] = []

    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "BaseModel":
        """Fit model on training feature matrix and target vector."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict discrete class labels for input feature matrix."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probability distribution of shape (n_samples, n_classes)."""
        pass

    def save(self, path: Path | str) -> None:
        """Persist fitted model pipeline to disk."""
        if not self.is_fitted:
            raise ValueError(f"Cannot save unfitted model: {self.model_name}")
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target_path)

    @classmethod
    def load(cls, path: Path | str) -> "BaseModel":
        """Load persisted model pipeline from disk."""
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found: {target_path}")
        model = joblib.load(target_path)
        if not isinstance(model, BaseModel):
            raise TypeError(f"Loaded object is not a BaseModel instance: {type(model)}")
        return model
