"""Model evaluation metrics and performance reporting tables for PSX predictions."""

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from psx_predictor.models.base import ModelEvaluationResult


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    model_name: str = "Model",
    target_name: str = "target",
    n_train: int = 0,
    classes: np.ndarray | None = None,
) -> ModelEvaluationResult:
    """Calculate comprehensive classification metrics.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted class labels.
        y_prob: Predicted probability matrix of shape (n_samples, n_classes).
        model_name: Name of the model.
        target_name: Name of the target variable.
        n_train: Number of training samples.
        classes: Array of class labels.

    Returns:
        ModelEvaluationResult dataclass with calculated metrics.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()

    roc_auc: float | None = None
    brier: float | None = None

    if y_prob is not None:
        unique_classes = np.unique(y_true_arr)
        # Binary classification
        if len(unique_classes) == 2 and y_prob.shape[1] >= 2:
            try:
                # Find positive class index (class 1.0 or highest class)
                pos_idx = 1 if y_prob.shape[1] == 2 else int(np.argmax(classes == 1.0))
                prob_pos = y_prob[:, pos_idx]
                roc_auc = float(roc_auc_score(y_true_arr, prob_pos))
                brier = float(brier_score_loss(y_true_arr, prob_pos))
            except Exception:
                roc_auc = None
                brier = None
        # Multiclass classification (e.g. 3-class)
        elif len(unique_classes) > 2 and y_prob.shape[1] == len(unique_classes):
            try:
                roc_auc = float(roc_auc_score(y_true_arr, y_prob, multi_class="ovr"))
            except Exception:
                roc_auc = None

    return ModelEvaluationResult(
        model_name=model_name,
        target_name=target_name,
        n_train=n_train,
        n_test=len(y_true_arr),
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1_macro=f1,
        roc_auc=roc_auc,
        brier_score=brier,
        confusion_matrix=cm,
    )


def display_evaluation_table(
    results: list[ModelEvaluationResult],
    symbol: str = "",
    console: Console | None = None,
) -> None:
    """Print a clean Rich table comparing model performance metrics."""
    c = console or Console()
    target_str = results[0].target_name.replace("target_", "") if results else "target"
    n_tr = results[0].n_train if results else 0
    n_te = results[0].n_test if results else 0
    title = (
        f"Model Evaluation Benchmark: {symbol} "
        f"[dim]({target_str} | Train: {n_tr} | Test: {n_te})[/dim]"
        if symbol
        else "Model Evaluation Benchmark"
    )

    table = Table(
        title=title,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Model", style="bold white", justify="left", min_width=26, no_wrap=True)
    table.add_column("Accuracy", style="bold green", justify="right", no_wrap=True)
    table.add_column("Precision", justify="right", no_wrap=True)
    table.add_column("Recall", justify="right", no_wrap=True)
    table.add_column("F1-Macro", style="bold yellow", justify="right", no_wrap=True)
    table.add_column("ROC-AUC", style="bold magenta", justify="right", no_wrap=True)
    table.add_column("Brier", justify="right", no_wrap=True)

    for r in results:
        roc_str = f"{r.roc_auc:.4f}" if r.roc_auc is not None else "N/A"
        brier_str = f"{r.brier_score:.4f}" if r.brier_score is not None else "N/A"

        table.add_row(
            r.model_name,
            f"{r.accuracy:.2%}",
            f"{r.precision:.4f}",
            f"{r.recall:.4f}",
            f"{r.f1_macro:.4f}",
            roc_str,
            brier_str,
        )

    c.print()
    c.print(table)
    c.print(
        Panel(
            "[dim]Notes: Evaluation performed strictly on out-of-sample test split. "
            "ROC-AUC > 0.50 denotes signal over random guessing. "
            "Lower Brier score indicates superior probability calibration.[/dim]",
            border_style="dim",
        )
    )
