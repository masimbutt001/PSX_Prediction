"""L2 Regularized Logistic Regression baseline classifier with leakage-free scaling."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from psx_predictor.models.base import BaseModel


class LogisticRegressionBaseline(BaseModel):
    """L2-regularized Logistic Regression model with leak-free out-of-sample standardization.

    Guarantees:
        1. StandardScaler is fit EXCLUSIVELY on the training split during fit().
        2. Test/inference features are transformed using only the train-derived mean and variance.
        3. Feature names are preserved to inspect learned linear weights/coefficients.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        super().__init__(model_name="Logistic Regression (L2)")
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler_: StandardScaler | None = None
        self.clf_: LogisticRegression | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "LogisticRegressionBaseline":
        """Fit scaler and logistic regression on training partition."""
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_mat = X.to_numpy()
        else:
            self.feature_names_ = [f"feat_{i}" for i in range(X.shape[1])]
            X_mat = np.asarray(X)

        # Fit scaler ONLY on training data
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_mat)

        # Fit regularized linear classifier (L2 is default when solver='lbfgs')
        self.clf_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.clf_.fit(X_scaled, y_arr)
        self.is_fitted = True
        return self

    def _prepare_X(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.scaler_ is None or self.clf_ is None:
            raise ValueError("Model is not fitted yet.")

        if isinstance(X, pd.DataFrame):
            X_mat = X.to_numpy()
        else:
            X_mat = np.asarray(X)

        # Transform using strictly training parameters
        return self.scaler_.transform(X_mat)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict discrete class labels."""
        X_scaled = self._prepare_X(X)
        assert self.clf_ is not None
        return self.clf_.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probability distribution."""
        X_scaled = self._prepare_X(X)
        assert self.clf_ is not None
        return self.clf_.predict_proba(X_scaled)

    def get_feature_importances(self) -> dict[str, float]:
        """Return learned model coefficients mapped to feature names."""
        if not self.is_fitted or self.clf_ is None:
            raise ValueError("Model is not fitted yet.")

        coefs = self.clf_.coef_
        # For binary classification, coef_ has shape (1, n_features)
        weights = coefs[0] if coefs.shape[0] == 1 else np.mean(np.abs(coefs), axis=0)

        importance_dict = {name: float(weights[i]) for i, name in enumerate(self.feature_names_)}
        # Sort descending by absolute weight magnitude
        return dict(sorted(importance_dict.items(), key=lambda item: abs(item[1]), reverse=True))
