"""API route routers package."""

from psx_predictor.api.routes.backtests import router as backtests_router
from psx_predictor.api.routes.predictions import router as predictions_router
from psx_predictor.api.routes.stocks import router as stocks_router

__all__ = ["stocks_router", "predictions_router", "backtests_router"]
