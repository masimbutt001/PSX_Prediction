"""PSX Predictor models package: baselines, linear classifiers, evaluation, and splitting."""

from psx_predictor.models.base import BaseModel, ModelEvaluationResult
from psx_predictor.models.baselines import (
    MajorityClassifier,
    NaivePersistenceClassifier,
    SMACrossoverClassifier,
)
from psx_predictor.models.evaluation import display_evaluation_table, evaluate_classifier
from psx_predictor.models.linear import LogisticRegressionBaseline
from psx_predictor.models.split import (
    DEFAULT_TECHNICAL_FEATURES,
    ChronologicalSplit,
    chronological_train_test_split,
)
from psx_predictor.models.trainer import ModelTrainer

__all__ = [
    "BaseModel",
    "ModelEvaluationResult",
    "MajorityClassifier",
    "NaivePersistenceClassifier",
    "SMACrossoverClassifier",
    "LogisticRegressionBaseline",
    "ChronologicalSplit",
    "chronological_train_test_split",
    "DEFAULT_TECHNICAL_FEATURES",
    "evaluate_classifier",
    "display_evaluation_table",
    "ModelTrainer",
]
