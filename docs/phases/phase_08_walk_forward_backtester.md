# Phase 08 — Walk-Forward Backtester & Trading Simulation

## 1. Objective
Build the platform's core evaluation engine using walk-forward (rolling and expanding window) cross-validation. Incorporate realistic PSX transaction costs, brokerage commissions, and circuit breaker lock-limit execution constraints.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/backtesting/
├── __init__.py
├── walk_forward.py                    # Expanding and rolling window partitioner
├── simulator.py                       # Order execution and portfolio accounting
└── metrics.py                         # Sharpe, Sortino, Drawdown, CAGR computations

tests/unit/
└── test_backtester.py                 # Simulation accuracy, lock rules, and friction tests
```

---

## 3. Detailed Specifications

### 3.1 Walk-Forward Partitioner (`walk_forward.py`)
- Configurable training window (e.g., 36 months) and step size (e.g., 3 months).
- Automatically splits dataset into $K$ chronological folds with zero overlap between training targets and test features (purging and embargoing).

### 3.2 Simulation Engine (`simulator.py`)
- **Friction Modeling:**
  - Brokerage: 0.15% per side
  - SECP/CDC/NCCPL charges: 0.015% per side
  - Slippage: 0.10% per side
- **Circuit Breaker Fill Rejection:**
  - If `is_upper_lock == True` at session close, **buy order is rejected** (zero fill).
  - If `is_lower_lock == True` at session close, **sell order is rejected** (zero fill).
- Portfolio tracks: Cash balance, equity value, trade log, daily mark-to-market.

### 3.3 Financial Metrics (`metrics.py`)
- Total Return, CAGR
- Annualized Sharpe Ratio (using SBP Policy Rate or KIBOR as risk-free rate)
- Maximum Drawdown (MDD) & Drawdown Duration
- Win Rate, Profit Factor, Turnover

---

## 4. Testing Plan
- `test_walk_forward_splits_no_overlap()`: Validates that test dates are strictly greater than train dates across all folds.
- `test_upper_lock_prevents_buy_execution()`: Asserts buy signal on a +7.5% locked session generates no trade fill.
- `test_transaction_cost_deduction()`: Validates that round-trip trade deducts exactly specified basis points.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_backtester.py -v
psx backtest --symbol OGDC --strategy sma_crossover
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] Walk-forward simulation engine operational.
- [ ] PSX friction and circuit locks enforced.
- [ ] Full metric summary report generated.
- [ ] Phase 8 completion report documented.
