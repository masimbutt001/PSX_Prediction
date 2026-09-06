"""Walk-forward expanding and rolling cross-validation partitioner with embargoing."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    """Represents a single chronological train/test fold."""

    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start_date: Any
    train_end_date: Any
    test_start_date: Any
    test_end_date: Any
    n_train: int
    n_test: int


class WalkForwardPartitioner:
    """Chronological walk-forward cross-validation splitter.

    Guarantees:
        1. Preserves temporal ordering across all folds.
        2. Test window strictly follows training window + embargo period.
        3. Supports both 'expanding' (anchored start) and 'rolling' (fixed window) modes.
        4. Embargo prevents information leakage across multi-day target horizons.
    """

    def __init__(
        self,
        train_window: int = 500,
        test_window: int = 60,
        step_size: int = 60,
        embargo: int = 5,
        mode: Literal["expanding", "rolling"] = "expanding",
        min_test_samples: int = 15,
    ) -> None:
        """Initialize partitioner parameters.

        Args:
            train_window: Number of sessions in the initial training partition.
            test_window: Number of sessions evaluated in each test partition.
            step_size: Step forward in sessions between consecutive folds.
            embargo: Buffer sessions between train end and test start (purges lookahead).
            mode: 'expanding' (grows train window) or 'rolling' (fixed-size train window).
            min_test_samples: Minimum test samples required to form a valid fold.
        """
        if train_window <= 0:
            raise ValueError("train_window must be positive.")
        if test_window <= 0:
            raise ValueError("test_window must be positive.")
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        if embargo < 0:
            raise ValueError("embargo must be non-negative.")
        if mode not in ("expanding", "rolling"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'expanding' or 'rolling'.")

        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.embargo = embargo
        self.mode = mode
        self.min_test_samples = min_test_samples

    def split(
        self,
        df: pd.DataFrame,
        date_col: str = "trade_date",
    ) -> list[WalkForwardFold]:
        """Generate chronological folds from DataFrame.

        Args:
            df: Input dataset.
            date_col: Name of date column (used for metadata).

        Returns:
            List of WalkForwardFold instances.
        """
        n_samples = len(df)
        total_required = self.train_window + self.embargo + self.min_test_samples
        if n_samples < total_required:
            raise ValueError(
                f"Dataset length ({n_samples}) is insufficient for configured partitioner "
                f"(requires at least {total_required} sessions)."
            )

        has_dates = date_col in df.columns
        dates = df[date_col].to_numpy() if has_dates else np.arange(n_samples)

        folds: list[WalkForwardFold] = []
        fold_idx = 0
        current_step = 0

        while True:
            if self.mode == "expanding":
                train_start_idx = 0
                train_end_idx = self.train_window + current_step * self.step_size
            else:  # rolling
                train_start_idx = current_step * self.step_size
                train_end_idx = train_start_idx + self.train_window

            test_start_idx = train_end_idx + self.embargo
            test_end_idx = min(test_start_idx + self.test_window, n_samples)

            if test_start_idx >= n_samples:
                break

            n_test = test_end_idx - test_start_idx
            if n_test < self.min_test_samples:
                break

            train_indices = np.arange(train_start_idx, train_end_idx)
            test_indices = np.arange(test_start_idx, test_end_idx)

            fold = WalkForwardFold(
                fold_idx=fold_idx,
                train_indices=train_indices,
                test_indices=test_indices,
                train_start_date=dates[train_start_idx],
                train_end_date=dates[train_end_idx - 1],
                test_start_date=dates[test_start_idx],
                test_end_date=dates[test_end_idx - 1],
                n_train=len(train_indices),
                n_test=n_test,
            )
            folds.append(fold)

            fold_idx += 1
            current_step += 1

            if test_end_idx >= n_samples:
                break

        return folds
