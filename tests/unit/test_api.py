"""Unit tests for Phase 17: FastAPI REST Backend and API CLI."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from psx_predictor.api.app import create_app
from psx_predictor.cli.main import app as cli_app
from psx_predictor.predictions.registry import PredictionRecord, PredictionRegistry
from psx_predictor.storage.parquet_io import write_parquet_atomic
from psx_predictor.storage.paths import ensure_directories, get_storage_paths


@pytest.fixture
def api_test_env(tmp_path: Path) -> dict[str, object]:
    """Provide isolated environment with synthetic stock and prediction data."""
    paths = get_storage_paths(tmp_path)
    ensure_directories(tmp_path)

    # 1. Create synthetic processed price data for OGDC (60 sessions)
    dates = pd.date_range(start="2024-01-01", periods=60, freq="B")
    np.random.seed(42)
    closes = 100.0 + np.cumsum(np.random.randn(60) * 1.5)

    price_df = pd.DataFrame(
        {
            "trade_date": [d.strftime("%Y-%m-%d") for d in dates],
            "open": closes - 0.5,
            "high": closes + 1.5,
            "low": closes - 1.5,
            "close": closes,
            "volume": np.random.randint(500_000, 2_000_000, 60),
            "adjusted_close": closes,
        }
    )
    ogdc_file = paths["processed_prices"] / "OGDC.parquet"
    write_parquet_atomic(price_df, ogdc_file)

    # 2. Register a mock prediction
    registry = PredictionRegistry(storage_paths=paths)
    rec = PredictionRecord(
        prediction_id="pred-test-001",
        symbol="OGDC",
        target_date="2024-03-26",
        generated_at="2024-03-25T16:00:00+05:00",
        model_name="ensemble_stacked",
        target_name="target_next_day_dir",
        up_probability=0.68,
        signal="BUY",
        confidence=0.36,
        drivers_json='{"rsi_14": 0.42, "sentiment_compound": 0.35}',
        realized_outcome=1.0,
        is_correct=True,
    )
    registry.log_prediction(rec)

    # 3. Create FastAPI test client
    fastapi_app = create_app(data_dir=tmp_path)
    client = TestClient(fastapi_app)

    return {
        "client": client,
        "paths": paths,
        "tmp_path": tmp_path,
    }


def test_health_endpoint(api_test_env: dict[str, object]) -> None:
    """Test GET /health returns diagnostic status and configured stock count."""
    client = api_test_env["client"]
    assert isinstance(client, TestClient)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["database_connected"] is True
    assert data["total_configured_stocks"] >= 1


def test_root_endpoint(api_test_env: dict[str, object]) -> None:
    """Test GET / welcome endpoint returns documentation paths."""
    client = api_test_env["client"]
    assert isinstance(client, TestClient)

    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "documentation" in data
    assert "message" in data


def test_stocks_list_endpoint(api_test_env: dict[str, object]) -> None:
    """Test GET /stocks returns universe summaries with storage status."""
    client = api_test_env["client"]
    assert isinstance(client, TestClient)

    response = client.get("/stocks")
    assert response.status_code == 200
    stocks = response.json()
    assert isinstance(stocks, list)
    assert len(stocks) >= 1

    ogdc_item = next((s for s in stocks if s["symbol"] == "OGDC"), None)
    assert ogdc_item is not None
    assert ogdc_item["has_data"] is True
    assert ogdc_item["total_sessions"] == 60


def test_stocks_filter_by_sector(api_test_env: dict[str, object]) -> None:
    """Test GET /stocks filtering by sector."""
    client = api_test_env["client"]
    assert isinstance(client, TestClient)

    response = client.get("/stocks?sector=Oil%20%26%20Gas%20Exploration")
    assert response.status_code == 200
    stocks = response.json()
    assert all("Oil" in s["sector"] for s in stocks)


def test_stock_detail_endpoint(api_test_env: dict[str, object]) -> None:
    """Test GET /stocks/{symbol} returns individual stock details."""
    client = api_test_env["client"]
    assert isinstance(client, TestClient)

    response = client.get("/stocks/OGDC")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "OGDC"
    assert "Oil" in data["name"]
    assert data["has_data"] is True
    assert data["total_sessions"] == 60


def test_stock_detail_not_found(api_test_env: dict[str, object]) -> None:
    """Test GET /stocks/{symbol} returns 404 for unconfigured stock."""
    client = api_test_env["client"]
    assert isinstance(client, TestClient)

    response = client.get("/stocks/NONEXISTENT")
    assert response.status_code == 404
    assert "not configured" in response.json()["detail"]


def test_stock_history_endpoint(api_test_env: dict[str, object]) -> None:
    """Test GET /stocks/{symbol}/history returns OHLCV series."""
    client = api_test_env["client"]
    assert isinstance(client, TestClient)

    response = client.get("/stocks/OGDC/history?days=15")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "OGDC"
    assert len(data["history"]) == 15
    first_pt = data["history"][0]
    assert "close" in first_pt
    assert "volume" in first_pt
    assert "trade_date" not in first_pt  # Aliased to date
    assert "date" in first_pt


def test_stock_history_not_found(api_test_env: dict[str, object]) -> None:
    """Test GET /stocks/{symbol}/history returns 404 if no price file exists."""
    client = api_test_env["client"]
    assert isinstance(client, TestClient)

    response = client.get("/stocks/HUBC/history")
    assert response.status_code == 404


def test_predictions_list_endpoint(api_test_env: dict[str, object]) -> None:
    """Test GET /predictions returns audit registry records."""
    client = api_test_env["client"]
    assert isinstance(client, TestClient)

    response = client.get("/predictions")
    assert response.status_code == 200
    preds = response.json()
    assert isinstance(preds, list)
    assert len(preds) >= 1
    assert preds[0]["symbol"] == "OGDC"
    assert preds[0]["probability"] == 0.68
    assert preds[0]["is_correct"] is True


def test_latest_prediction_endpoint(api_test_env: dict[str, object]) -> None:
    """Test GET /stocks/{symbol}/latest-prediction returns latest logged record."""
    client = api_test_env["client"]
    assert isinstance(client, TestClient)

    response = client.get("/stocks/OGDC/latest-prediction?generate_if_missing=false")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "OGDC"
    assert data["up_probability"] == 0.68
    assert data["signal"] == "BUY"
    assert len(data["top_drivers"]) >= 1


def test_latest_prediction_not_found(api_test_env: dict[str, object]) -> None:
    """Test GET /stocks/{symbol}/latest-prediction returns 404 when missing and no gen."""
    client = api_test_env["client"]
    assert isinstance(client, TestClient)

    response = client.get("/stocks/HUBC/latest-prediction?generate_if_missing=false")
    assert response.status_code == 404


def test_backtest_endpoint(api_test_env: dict[str, object]) -> None:
    """Test GET /backtests/{symbol} executes simulation and returns equity curve."""
    client = api_test_env["client"]
    assert isinstance(client, TestClient)

    response = client.get("/backtests/OGDC?strategy=buy_and_hold&initial_cash=500000")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "OGDC"
    assert "total_return" in data
    assert "sharpe_ratio" in data
    assert "max_drawdown" in data
    assert "equity_curve" in data
    assert len(data["equity_curve"]) == 60


def test_backtest_unknown_strategy(api_test_env: dict[str, object]) -> None:
    """Test GET /backtests/{symbol} returns 400 for unsupported strategy."""
    client = api_test_env["client"]
    assert isinstance(client, TestClient)

    response = client.get("/backtests/OGDC?strategy=magical_ai_alpha")
    assert response.status_code == 400
    assert "Unknown strategy" in response.json()["detail"]


def test_backtest_missing_symbol(api_test_env: dict[str, object]) -> None:
    """Test GET /backtests/{symbol} returns 404 if symbol has no price history."""
    client = api_test_env["client"]
    assert isinstance(client, TestClient)

    response = client.get("/backtests/MISSING_SYM")
    assert response.status_code == 404


def test_cli_api_start_dry_run() -> None:
    """Test CLI command `psx api start --dry-run` prints config and exits 0."""
    runner = CliRunner()
    result = runner.invoke(
        cli_app, ["api", "start", "--host", "127.0.0.1", "--port", "9999", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "Initializing PSX Predictor REST API" in result.output
    assert "9999" in result.output
