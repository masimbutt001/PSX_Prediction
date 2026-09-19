"""FastAPI application factory for the PSX Prediction Platform REST backend."""

import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from psx_predictor import __version__
from psx_predictor.api.routes.backtests import router as backtests_router
from psx_predictor.api.routes.predictions import router as predictions_router
from psx_predictor.api.routes.stocks import router as stocks_router
from psx_predictor.api.schemas import HealthResponse
from psx_predictor.config.loader import load_config
from psx_predictor.storage.paths import ensure_directories, get_storage_paths


def create_app(data_dir: Optional[Path] = None) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Args:
        data_dir: Optional custom analytical storage directory.

    Returns:
        Configured FastAPI application instance.
    """
    cfg = load_config()
    target_dir = data_dir or cfg.settings.data_dir
    ensure_directories(target_dir)

    app = FastAPI(
        title="PSX Predictor API",
        version=__version__,
        description=(
            "High-performance REST API for Pakistan Stock Exchange (PSX) market predictions, "
            "technical features, news sentiment, macro drivers, and backtesting simulations."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.data_dir = target_dir

    # Enable Cross-Origin Resource Sharing (CORS) for web dashboard consumers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount component routers
    app.include_router(stocks_router)
    app.include_router(predictions_router)
    app.include_router(backtests_router)

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    def health_check() -> HealthResponse:
        """Return platform diagnostic health and local store status."""
        paths = get_storage_paths(target_dir)
        db_connected = paths["root"].exists()
        stock_count = len(cfg.stocks)

        return HealthResponse(
            status="ok",
            version=__version__,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            database_connected=db_connected,
            total_configured_stocks=stock_count,
        )

    @app.get("/", tags=["root"])
    def root_index() -> dict[str, str]:
        """Root welcome endpoint providing documentation links."""
        return {
            "message": "Welcome to the PSX Predictor Platform API",
            "version": __version__,
            "documentation": "/docs",
            "alternative_documentation": "/redoc",
            "health": "/health",
        }

    return app
