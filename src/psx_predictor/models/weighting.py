"""Out-of-fold walk-forward validation and ensemble weight optimization."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from psx_predictor.models.base import BaseModel


@dataclass(frozen=True)
class TimeSeriesFold:
    """Chronological train and validation indices without lookahead leakage."""

    train_indices: np.ndarray
    val_indices: np.ndarray
    fold_idx: int


def generate_expanding_folds(
    n_samples: int,
    n_splits: int = 4,
    min_train_ratio: float = 0.4,
) -> list[TimeSeriesFold]:
    """Generate expanding chronological folds for walk-forward validation.

    Args:
        n_samples: Total number of sequential observations.
        n_splits: Number of walk-forward validation splits (default: 4).
        min_train_ratio: Minimum fraction of samples reserved for initial fold.

    Returns:
        List of TimeSeriesFold instances with non-overlapping sequential val splits.
    """
    if n_samples < 20:
        raise ValueError(f"Insufficient samples ({n_samples}) to construct time series folds.")

    min_train = max(10, int(n_samples * min_train_ratio))
    val_pool = n_samples - min_train

    if val_pool < n_splits:
        n_splits = max(1, val_pool)

    val_size = val_pool // n_splits
    folds: list[TimeSeriesFold] = []

    for i in range(n_splits):
        val_start = min_train + i * val_size
        val_end = min_train + (i + 1) * val_size if i < n_splits - 1 else n_samples

        train_idx = np.arange(0, val_start)
        val_idx = np.arange(val_start, val_end)

        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append(
                TimeSeriesFold(
                    train_indices=train_idx,
                    val_indices=val_idx,
                    fold_idx=i,
                )
            )

    return folds


class OutOfFoldWeightOptimizer:
    """Computes out-of-fold predictions and optimizes stacking ensemble weights."""

    def __init__(self, n_splits: int = 4) -> None:
        """Initialize optimizer with target number of time series splits."""
        self.n_splits = n_splits

    def compute_oof_predictions(
        self,
        base_models: dict[str, tuple[BaseModel, list[str]]],
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Generate out-of-fold probability predictions across base estimators.

        Args:
            base_models: Dictionary mapping name -> (model_instance, feature_columns).
            X: Full training feature DataFrame.
            y: Binary or multi-class target labels array.

        Returns:
            Tuple of:
              - Z_oof: Matrix of shape (N_val, num_models) containing positive class probabilities.
              - y_oof: Vector of shape (N_val,) of corresponding targets.
              - model_names: List of model names corresponding to columns of Z_oof.
        """
        n_samples = len(X)
        folds = generate_expanding_folds(n_samples=n_samples, n_splits=self.n_splits)

        if not folds:
            raise ValueError(f"Could not generate valid folds for {n_samples} samples.")

        model_names = list(base_models.keys())
        oof_preds: dict[str, list[float]] = {name: [] for name in model_names}
        oof_targets: list[int] = []

        logger.info(
            f"Computing OOF predictions across {len(model_names)} base models and "
            f"{len(folds)} chronological folds..."
        )

        for fold in folds:
            X_tr = X.iloc[fold.train_indices]
            y_tr = y[fold.train_indices]
            X_val = X.iloc[fold.val_indices]
            y_val = y[fold.val_indices]

            oof_targets.extend(y_val.tolist())

            for name, (model, cols) in base_models.items():
                active_cols = [c for c in cols if c in X_tr.columns]
                # Fit base model on fold training slice
                model.fit(X_tr[active_cols], y_tr)
                # Predict probabilities on fold validation slice
                prob_val = model.predict_proba(X_val[active_cols])

                # Extract positive class probability
                if prob_val.ndim == 2 and prob_val.shape[1] >= 2:
                    p_pos = prob_val[:, 1]
                else:
                    p_pos = prob_val.ravel()

                oof_preds[name].extend(p_pos.tolist())

        Z_oof = np.column_stack([np.array(oof_preds[name]) for name in model_names])
        y_oof = np.array(oof_targets)

        logger.info(
            f"Generated OOF prediction matrix of shape {Z_oof.shape} "
            f"across {len(y_oof)} validation points."
        )
        return Z_oof, y_oof, model_names

    def optimize_weights(
        self,
        Z_oof: np.ndarray,
        y_oof: np.ndarray,
        initial_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Find optimal non-negative ensemble weights minimizing cross-entropy log loss.

        Args:
            Z_oof: Out-of-fold probability predictions of shape (N, M).
            y_oof: Target array of shape (N,).
            initial_weights: Optional starting weights. Defaults to uniform.

        Returns:
            Normalized weight vector of length M summing to 1.0.
        """
        n_models = Z_oof.shape[1]
        if initial_weights is None:
            w0 = np.full(n_models, 1.0 / n_models)
        else:
            w0 = np.array(initial_weights, dtype=float)

        def loss_func(weights: np.ndarray) -> float:
            # Clip weights to sum to 1
            w = np.maximum(0.0, weights)
            s = np.sum(w)
            if s > 0:
                w = w / s
            else:
                w = np.full(n_models, 1.0 / n_models)

            # Combined probability
            p_combined = np.clip(np.dot(Z_oof, w), 1e-6, 1.0 - 1e-6)
            try:
                loss_val = float(log_loss(y_oof, p_combined))
                return loss_val
            except Exception:
                return 1e6

        bounds = [(0.0, 1.0) for _ in range(n_models)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        opt = minimize(
            loss_func,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-6},
        )

        if opt.success and np.sum(opt.x) > 0:
            final_weights = np.maximum(0.0, opt.x)
            final_weights = final_weights / np.sum(final_weights)
        else:
            logger.warning("Optimization failed or non-convergent; using equal weights.")
            final_weights = np.full(n_models, 1.0 / n_models)

        return final_weights
