# Phase 17 — FastAPI REST Backend

## 1. Objective
Implement a high-performance, typed REST API using FastAPI. Expose endpoints for configured stock universes, historical price data, feature sets, model predictions, explainability drivers, and backtest results.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/api/
├── __init__.py
├── app.py                             # FastAPI application factory
├── routes/
│   ├── __init__.py
│   ├── stocks.py                      # /stocks and /stocks/{symbol} endpoints
│   ├── predictions.py                 # /predictions and latest signals
│   └── backtests.py                   # /backtests results
└── schemas.py                         # Pydantic response and request models

tests/unit/
└── test_api.py                        # FastAPI TestClient endpoint tests
```

---

## 3. Detailed Specifications

### 3.1 Endpoints
- `GET /health` $\to$ Service health status.
- `GET /stocks` $\to$ List of configured PSX stocks and sectors.
- `GET /stocks/{symbol}/history?days=100` $\to$ OHLCV + adjusted prices.
- `GET /stocks/{symbol}/latest-prediction` $\to$ Latest forecast, probability, signal, drivers.
- `GET /predictions?symbol=OGDC&limit=30` $\to$ Historical forecasts and realized outcomes.
- `GET /backtests/{symbol}` $\to$ Full walk-forward metrics, equity curve, drawdowns.

---

## 4. Testing Plan
- `test_health_endpoint()`: Asserts HTTP 200 and status `ok`.
- `test_stocks_list_endpoint()`: Asserts returns JSON list matching `config/stocks.yaml`.
- `test_latest_prediction_endpoint()`: Verifies schema includes `symbol`, `up_probability`, `signal`, `key_drivers`.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_api.py -v
psx api start --port 8000 --dry-run
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] All REST endpoints tested using `TestClient`.
- [ ] Pydantic responses validated with interactive Swagger UI at `/docs`.
- [ ] Phase 17 completion report documented.
