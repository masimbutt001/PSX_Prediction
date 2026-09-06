"""PSX Predictor models package: baselines, linear, trees, and benchmark comparators."""

from psx_predictor.models.base import BaseModel, ModelEvaluationResult
from psx_predictor.models.baselines import (
    MajorityClassifier,
    NaivePersistenceClassifier,
    SMACrossoverClassifier,
)
from psx_predictor.models.comparator import (
    ModelBenchmarkSummary,
    ModelComparator,
    display_comparison_table,
)
from psx_predictor.models.evaluation import display_evaluation_table, evaluate_classifier
from psx_predictor.models.linear import LogisticRegressionBaseline
from psx_predictor.models.split import (
    DEFAULT_TECHNICAL_FEATURES,
    ChronologicalSplit,
    chronological_train_test_split,
)
from psx_predictor.models.trainer import ModelTrainer
from psx_predictor.models.trees import RandomForestBaseline, XGBoostBaseline

__all__ = [
    "BaseModel",
    "ModelEvaluationResult",
    "MajorityClassifier",
    "NaivePersistenceClassifier",
    "SMACrossoverClassifier",
    "LogisticRegressionBaseline",
    "RandomForestBaseline",
    "XGBoostBaseline",
    "ChronologicalSplit",
    "chronological_train_test_split",
    "DEFAULT_TECHNICAL_FEATURES",
    "evaluate_classifier",
    "display_evaluation_table",
    "ModelTrainer",
    "ModelComparator",
    "ModelBenchmarkSummary",
    "display_comparison_table",
]
