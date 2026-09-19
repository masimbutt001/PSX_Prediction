"""FastAPI endpoints for stock universe inspection and historical price series."""

from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from psx_predictor.api.schemas import StockHistoryPoint, StockHistoryResponse, StockSummary
from psx_predictor.config.loader import load_config
from psx_predictor.storage.parquet_io import read_parquet
from psx_predictor.storage.paths import get_storage_paths

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("", response_model=list[StockSummary])
def list_stocks(
    request: Request,
    sector: Optional[str] = Query(default=None, description="Filter by industry sector"),
    enabled_only: bool = Query(default=False, description="Filter only enabled stocks"),
) -> list[StockSummary]:
    """Retrieve list of configured PSX stocks and local storage status."""
    cfg = load_config()
    data_dir = getattr(request.app.state, "data_dir", None) or cfg.settings.data_dir
    paths = get_storage_paths(data_dir)
    summaries: list[StockSummary] = []

    for stock_cfg in cfg.stocks:
        if enabled_only and not stock_cfg.enabled:
            continue
        if sector and stock_cfg.sector.lower() != sector.lower():
            continue

        sym = stock_cfg.symbol.upper()
        price_file = paths["processed_prices"] / f"{sym}.parquet"
        has_data = price_file.exists()
        total_sessions: Optional[int] = None
        latest_date: Optional[str] = None

        if has_data:
            try:
                df = read_parquet(price_file)
                total_sessions = len(df)
                if not df.empty:
                    d_col = "trade_date" if "trade_date" in df.columns else "date"
                    latest_date = str(df[d_col].iloc[-1])
            except Exception:
                has_data = False

        summaries.append(
            StockSummary(
                symbol=sym,
                name=stock_cfg.name,
                sector=stock_cfg.sector,
                is_enabled=stock_cfg.enabled,
                has_data=has_data,
                total_sessions=total_sessions,
                latest_date=latest_date,
            )
        )

    return summaries


@router.get("/{symbol}", response_model=StockSummary)
def get_stock(symbol: str, request: Request) -> StockSummary:
    """Retrieve metadata and local storage status for a single PSX stock."""
    clean_sym = symbol.strip().upper()
    cfg = load_config()
    stock_cfg = cfg.get_stock(clean_sym)

    if not stock_cfg:
        raise HTTPException(
            status_code=404, detail=f"Stock '{clean_sym}' is not configured in platform universe."
        )

    data_dir = getattr(request.app.state, "data_dir", None) or cfg.settings.data_dir
    paths = get_storage_paths(data_dir)
    price_file = paths["processed_prices"] / f"{clean_sym}.parquet"
    has_data = price_file.exists()
    total_sessions: Optional[int] = None
    latest_date: Optional[str] = None

    if has_data:
        try:
            df = read_parquet(price_file)
            total_sessions = len(df)
            if not df.empty:
                d_col = "trade_date" if "trade_date" in df.columns else "date"
                latest_date = str(df[d_col].iloc[-1])
        except Exception:
            has_data = False

    return StockSummary(
        symbol=clean_sym,
        name=stock_cfg.name,
        sector=stock_cfg.sector,
        is_enabled=stock_cfg.enabled,
        has_data=has_data,
        total_sessions=total_sessions,
        latest_date=latest_date,
    )


@router.get("/{symbol}/history", response_model=StockHistoryResponse)
def get_stock_history(
    symbol: str,
    request: Request,
    days: int = Query(default=100, ge=1, le=1500, description="Number of recent sessions"),
) -> StockHistoryResponse:
    """Retrieve historical daily OHLCV price series for a single PSX stock."""
    clean_sym = symbol.strip().upper()
    cfg = load_config()
    data_dir = getattr(request.app.state, "data_dir", None) or cfg.settings.data_dir
    paths = get_storage_paths(data_dir)
    price_file = paths["processed_prices"] / f"{clean_sym}.parquet"

    if not price_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Historical price data for '{clean_sym}' not found. Run bootstrap first.",
        )

    df = read_parquet(price_file)
    if df.empty:
        return StockHistoryResponse(symbol=clean_sym, total_sessions=0, history=[])

    slice_df = df.tail(days).copy()
    d_col = "trade_date" if "trade_date" in slice_df.columns else "date"

    points: list[StockHistoryPoint] = []
    for _, row in slice_df.iterrows():
        d_val = str(row[d_col])
        adj_val = row.get("adjusted_close")
        adj_c = float(adj_val) if adj_val is not None and pd.notna(adj_val) else None
        points.append(
            StockHistoryPoint(
                date=d_val,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
                adjusted_close=adj_c,
            )
        )

    return StockHistoryResponse(
        symbol=clean_sym,
        total_sessions=len(points),
        history=points,
    )
