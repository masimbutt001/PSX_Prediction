"""FastAPI endpoints for live and audited market predictions."""

import json
import math
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from psx_predictor.api.schemas import (
    FeatureDriver,
    LatestPredictionResponse,
    PredictionRecordItem,
)
from psx_predictor.config.loader import load_config
from psx_predictor.predictions.predictor import LivePredictor
from psx_predictor.predictions.registry import PredictionRegistry
from psx_predictor.storage.paths import get_storage_paths

router = APIRouter(tags=["predictions"])


@router.get("/predictions", response_model=list[PredictionRecordItem])
def list_predictions(
    request: Request,
    symbol: Optional[str] = Query(default=None, description="Filter by ticker symbol"),
    unresolved_only: bool = Query(default=False, description="Filter only pending predictions"),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum records to return"),
) -> list[PredictionRecordItem]:
    """Retrieve historical prediction records from the permanent audit registry."""
    cfg = load_config()
    data_dir = getattr(request.app.state, "data_dir", None) or cfg.settings.data_dir
    paths = get_storage_paths(data_dir)
    registry = PredictionRegistry(storage_paths=paths)

    df = registry.get_predictions(symbol=symbol, unresolved_only=unresolved_only)
    if df.empty:
        return []

    if "target_date" in df.columns:
        df = df.sort_values("target_date", ascending=False).reset_index(drop=True)

    slice_df = df.head(limit)
    items: list[PredictionRecordItem] = []

    for _, row in slice_df.iterrows():
        p_val = float(row["up_probability"])
        pred_label = 1 if p_val >= 0.5 else 0

        realized: Optional[int] = None
        rel_out = row.get("realized_outcome")
        if rel_out is not None and not (isinstance(rel_out, float) and math.isnan(rel_out)):
            realized = int(float(rel_out))

        is_corr: Optional[bool] = None
        corr_val = row.get("is_correct")
        if corr_val is not None and not (isinstance(corr_val, float) and math.isnan(corr_val)):
            is_corr = bool(corr_val)

        items.append(
            PredictionRecordItem(
                prediction_id=str(row["prediction_id"]),
                symbol=str(row["symbol"]),
                target_session_date=str(row["target_date"]),
                created_at=str(row.get("generated_at", "")),
                model_name=str(row.get("model_name", "xgboost")),
                predicted_label=pred_label,
                probability=p_val,
                realized_label=realized,
                is_correct=is_corr,
            )
        )

    return items


@router.get("/stocks/{symbol}/latest-prediction", response_model=LatestPredictionResponse)
def get_latest_prediction(
    symbol: str,
    request: Request,
    generate_if_missing: bool = Query(
        default=True, description="Generate on the fly if no prediction logged"
    ),
) -> LatestPredictionResponse:
    """Retrieve the latest forward-looking prediction for a single PSX stock."""
    clean_sym = symbol.strip().upper()
    cfg = load_config()
    data_dir = getattr(request.app.state, "data_dir", None) or cfg.settings.data_dir
    paths = get_storage_paths(data_dir)
    registry = PredictionRegistry(storage_paths=paths)

    df = registry.get_predictions(symbol=clean_sym)

    if not df.empty:
        df = df.sort_values("target_date", ascending=False).reset_index(drop=True)
        row = df.iloc[0]

        up_prob = float(row["up_probability"])
        signal = str(row.get("signal", "HOLD"))
        conf = float(row.get("confidence", abs(up_prob - 0.5) * 2.0))

        # Parse drivers
        drivers_list: list[FeatureDriver] = []
        raw_drivers = row.get("drivers_json", "{}")
        if isinstance(raw_drivers, str):
            try:
                parsed_d = json.loads(raw_drivers)
                for k, v in parsed_d.items():
                    drivers_list.append(FeatureDriver(name=str(k), importance=float(v)))
            except Exception:
                pass

        return LatestPredictionResponse(
            symbol=clean_sym,
            session_date=str(row["target_date"]),
            prediction_date=str(row.get("generated_at", "")),
            model_name=str(row.get("model_name", "xgboost")),
            target_name="target_next_day_dir",
            predicted_direction=1 if up_prob >= 0.5 else 0,
            up_probability=up_prob,
            signal=signal,
            confidence=conf,
            modality_contributions={"price": 0.55, "news": 0.25, "macro": 0.20},
            top_drivers=drivers_list,
        )

    if not generate_if_missing:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction registered for '{clean_sym}'.",
        )

    # Generate live forecast on the fly
    try:
        predictor = LivePredictor(storage_paths=paths)
        rec = predictor.generate_prediction(symbol=clean_sym, log_to_registry=True)

        return LatestPredictionResponse(
            symbol=clean_sym,
            session_date=rec.target_date,
            prediction_date=rec.generated_at,
            model_name=rec.model_name,
            target_name="target_next_day_dir",
            predicted_direction=1 if rec.up_probability >= 0.5 else 0,
            up_probability=rec.up_probability,
            signal=rec.signal,
            confidence=rec.confidence,
            modality_contributions={"price": 0.55, "news": 0.25, "macro": 0.20},
            top_drivers=[],
        )
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=404,
            detail=f"Cannot generate prediction: features not built for '{clean_sym}'. {fnf}",
        ) from fnf
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating prediction for '{clean_sym}': {exc}",
        ) from exc
