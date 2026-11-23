"""Live prediction generator forecasting upcoming sessions and logging to registry."""

import datetime
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from psx_predictor.config.loader import load_config
from psx_predictor.models.base import BaseModel
from psx_predictor.models.linear import LogisticRegressionBaseline
from psx_predictor.models.split import DEFAULT_TECHNICAL_FEATURES
from psx_predictor.models.trees import RandomForestBaseline, XGBoostBaseline
from psx_predictor.news.calendar import PSXMarketCalendar
from psx_predictor.predictions.registry import PredictionRecord, PredictionRegistry
from psx_predictor.research.features import (
    RESEARCH_FEATURE_COLUMNS,
    build_research_features,
    research_context_for_date,
)
from psx_predictor.research.store import ResearchStore
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import ensure_directories


class LivePredictor:
    """Generates out-of-sample forward forecasts for the upcoming trading session."""

    def __init__(self, storage_paths: Optional[dict[str, Path]] = None) -> None:
        if storage_paths is None:
            config = load_config()
            self.storage_paths = ensure_directories(config.settings.data_dir)
        else:
            self.storage_paths = storage_paths

        self.registry = PredictionRegistry(storage_paths=self.storage_paths)
        self.calendar = PSXMarketCalendar()

    def _next_business_day(self, current_date: datetime.date) -> datetime.date:
        """Next PSX trading session after the given date, skipping weekends and PSX holidays."""
        return self.calendar.next_trading_day(current_date)

    def generate_prediction(
        self,
        symbol: str,
        model_name: str = "xgboost",
        target_col: str = "target_next_day_dir",
        log_to_registry: bool = True,
        use_research: bool = True,
    ) -> PredictionRecord:
        """Generate prediction for the next trading session and register it.

        Args:
            symbol: Ticker symbol (e.g. 'OGDC').
            model_name: 'xgboost', 'random_forest', or 'logistic'.
            target_col: Prediction target column name.
            log_to_registry: Whether to persist the forecast in predictions.parquet.
            use_research: Add point-in-time research features from data/research/ when
                research files exist for the symbol, and attach the research snapshot.

        Returns:
            Constructed and logged PredictionRecord.
        """
        clean_sym = symbol.upper().strip()
        feat_path = self.storage_paths["features_technical"] / f"{clean_sym}_tech_features.parquet"
        if not feat_path.exists():
            raise FileNotFoundError(
                f"Features dataset not found for '{clean_sym}' at: {feat_path}. "
                f"Please run 'psx features build --with-targets --symbols {clean_sym}' first."
            )

        df = read_parquet(feat_path).reset_index(drop=True)
        feature_cols = [c for c in DEFAULT_TECHNICAL_FEATURES if c in df.columns]

        reports = (
            ResearchStore(storage_paths=self.storage_paths).load_reports(clean_sym)
            if use_research
            else []
        )
        if reports:
            research_df = build_research_features(df["trade_date"], reports)
            for col in RESEARCH_FEATURE_COLUMNS:
                df[col] = research_df[col].to_numpy()
            feature_cols = feature_cols + RESEARCH_FEATURE_COLUMNS

        # The last row represents contemporaneous session T (where target T+1 is NaN)
        last_row = df.iloc[-1]
        last_date = last_row["trade_date"]
        if isinstance(last_date, str):
            last_dt = datetime.date.fromisoformat(last_date)
        elif isinstance(last_date, pd.Timestamp):
            last_dt = last_date.date()
        else:
            last_dt = last_date

        target_date = self._next_business_day(last_dt)

        # Training data: all historical sessions where target is NOT NaN
        train_df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        # Inference vector: session T features
        X_infer = pd.DataFrame([last_row[feature_cols]])

        # Model instantiation
        model_lower = model_name.lower().strip()
        model: BaseModel
        if model_lower in ("xgboost", "xgb"):
            model = XGBoostBaseline()
            resolved_name = "XGBoost"
        elif model_lower in ("random_forest", "rf"):
            model = RandomForestBaseline()
            resolved_name = "Random Forest"
        elif model_lower in ("logistic", "linear"):
            model = LogisticRegressionBaseline()
            resolved_name = "Logistic Regression (L2)"
        else:
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                "Choose 'xgboost', 'random_forest', or 'logistic'."
            )

        logger.info(
            f"[{clean_sym}] Fitting {resolved_name} on {len(X_train)} historical sessions..."
        )
        model.fit(X_train, y_train)

        # Inference
        proba = model.predict_proba(X_infer)
        up_prob = float(proba[0, 1]) if proba.shape[1] >= 2 else float(proba[0, 0])

        # Directional signal logic with confidence bounds
        if up_prob >= 0.52:
            signal = "BUY"
        elif up_prob <= 0.48:
            signal = "SELL"
        else:
            signal = "HOLD"

        confidence = float(abs(up_prob - 0.5) * 2.0)

        # Top feature drivers extraction
        drivers: dict[str, float] = {}
        if hasattr(model, "get_feature_importances"):
            try:
                imp = model.get_feature_importances()
                drivers = {k: round(v, 4) for k, v in list(imp.items())[:5]}
            except Exception:
                drivers = {}

        research_context: dict[str, object] = {}
        if reports:
            research_context = research_context_for_date(reports, last_dt)
            outlook = research_context.get("outlook_short_term")
            if isinstance(outlook, (int, float)) and signal != "HOLD" and outlook != 0:
                research_context["agrees_with_model"] = (outlook > 0) == (signal == "BUY")

        record = PredictionRecord(
            symbol=clean_sym,
            target_date=str(target_date),
            model_name=resolved_name,
            model_version="1.0.0",
            up_probability=round(up_prob, 4),
            expected_return=None,
            signal=signal,
            confidence=round(confidence, 4),
            feature_version="1.0.0",
            drivers_json=json.dumps(drivers),
            research_json=json.dumps(research_context),
        )

        if log_to_registry:
            self.registry.log_prediction(record)
            logger.info(
                f"[{clean_sym}] Logged prediction {record.prediction_id} "
                f"to registry for {record.target_date}"
            )

        return record
