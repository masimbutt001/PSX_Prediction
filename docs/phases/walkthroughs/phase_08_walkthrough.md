# Walkthrough — Phase 08: Walk-Forward Backtester & Trading Simulation

**Status:** Completed & Validated  
**Date:** September 2026  
**Specification:** [phase_08_walk_forward_backtester.md](../phase_08_walk_forward_backtester.md)

---

## 1. Objective Accomplished
Built the platform's financial evaluation and trading simulation engine. Features walk-forward expanding and rolling cross-validation with purging and embargo buffers, realistic PSX transaction cost modeling (brokerage, regulatory fees, slippage), and circuit breaker lock execution constraints (+7.5% upper lock buy rejection, -7.5% lower lock sell rejection). Computes industry-standard financial metrics calibrated to the State Bank of Pakistan (SBP) risk-free rate ($R_f = 15.0\%$).

---

## 2. Deliverables & Files Created / Modified

| File | Purpose |
| :--- | :--- |
| [`src/psx_predictor/backtesting/walk_forward.py`](../../../src/psx_predictor/backtesting/walk_forward.py) | `WalkForwardPartitioner` generating chronological folds with zero lookahead, configurable `train_window`, `test_window`, `step_size`, `mode` (`expanding` or `rolling`), and `embargo` periods preventing multi-horizon forward leakage. |
| [`src/psx_predictor/backtesting/simulator.py`](../../../src/psx_predictor/backtesting/simulator.py) | `BacktestSimulator`, `TransactionCostModel`, `TradeRecord`, and `SimulationResult`. Implements PSX microstructure rules: 26.5 bps per-side friction (53 bps round-trip) and circuit breaker order rejection logic. |
| [`src/psx_predictor/backtesting/metrics.py`](../../../src/psx_predictor/backtesting/metrics.py) | `BacktestMetrics` calculation (Total Return, CAGR, Sharpe Ratio, Sortino Ratio, Maximum Drawdown, MDD duration, Win Rate, Profit Factor, Total Friction) and Rich report formatter (`display_backtest_report`). |
| [`src/psx_predictor/backtesting/runner.py`](../../../src/psx_predictor/backtesting/runner.py) | `BacktestRunner` orchestrating dataset loading, strategy signal generation (`sma_crossover`, `naive_persistence`, `logistic`), walk-forward cross-validation, simulation execution, and metrics generation. |
| [`src/psx_predictor/backtesting/__init__.py`](../../../src/psx_predictor/backtesting/__init__.py) | Package initialization exporting the backtesting API. |
| [`src/psx_predictor/cli/main.py`](../../../src/psx_predictor/cli/main.py) | Registered `psx backtest` command (`--symbol`, `--strategy`, `--initial-cash`, `--rf-rate`). |
| [`tests/unit/test_backtester.py`](../../../tests/unit/test_backtester.py) | 8 unit tests covering fold monotonicity and embargoing, rolling mode window sizes, upper-lock buy rejection, lower-lock sell rejection, exact friction deduction, financial metric equations, runner integration, and CLI execution. |

---

## 3. Financial & Microstructure Specifications

### 3.1 PSX Transaction Cost Structure
Modeled by `TransactionCostModel`:
- **Brokerage Commission:** 15 bps (0.15%) per side
- **Regulatory Fees (SECP, CDC, NCCPL):** 1.5 bps (0.015%) per side
- **Market Slippage:** 10 bps (0.10%) per side
- **Total Friction:** 26.5 bps per side (53.0 bps round-trip)

### 3.2 Circuit Breaker Lock Rules
Modeled by `BacktestSimulator`:
- **Upper Circuit Lock (`is_upper_lock == True`):** Stock locked at daily +7.5% ceiling with pending buy orders. In simulated trading, incoming BUY orders are strictly **rejected** (zero fill).
- **Lower Circuit Lock (`is_lower_lock == True`):** Stock locked at daily -7.5% floor with pending sell orders. In simulated trading, incoming SELL orders are strictly **rejected** (zero fill).

---

## 4. Verification & Test Results

### 4.1 Code Quality & Static Typing
```powershell
uv run ruff format --check .
# Output: 51 files already formatted

uv run ruff check .
# Output: All checks passed!

uv run mypy src
# Output: Success: no issues found in 37 source files
```

### 4.2 Automated Unit Test Execution
```powershell
uv run pytest -v
```
**Test Results:**
```text
============================= test session starts =============================
platform win32 -- Python 3.12.11, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Development\PSX_Prediction
configfile: pyproject.toml
testpaths: tests
plugins: anyio-4.15.1, respx-0.23.1
collected 80 items / 2 deselected / 78 selected

tests\unit\test_backtester.py ........                                   [ 10%]
tests\unit\test_baselines.py ........                                    [ 20%]
tests\unit\test_collectors.py ............                               [ 35%]
tests\unit\test_config.py ..........                                     [ 48%]
tests\unit\test_features.py ......                                       [ 56%]
tests\unit\test_pipeline.py ..........                                   [ 69%]
tests\unit\test_storage.py ............                                  [ 84%]
tests\unit\test_targets.py ......                                        [ 92%]
tests\unit\test_updater.py ......                                        [100%]

====================== 78 passed, 2 deselected in 3.08s =======================
```

### 4.3 Live PSX Market Simulation Results

#### Strategy 1: SMA Crossover (20/50) on OGDC
```powershell
uv run psx backtest --symbol OGDC --strategy sma_crossover
```
```text
              PSX Strategy Backtest Report: OGDC (SMA_CROSSOVER)               
+-----------------------------------------------------------------------------+
| Portfolio / Metric      |            Value | Benchmark / Context            |
|-------------------------+------------------+--------------------------------|
| Initial Capital         | PKR 1,000,000.00 | Starting cash allocation       |
| Final Portfolio Equity  | PKR 2,157,575.27 | Ending mark-to-market value    |
| Cumulative Return       |         +115.76% | Total unannualized return      |
| CAGR (Annualized)       |          +16.77% | Compound annual growth rate    |
| Annualized Volatility   |           28.60% | Standard deviation of daily    |
| Sharpe Ratio (Rf=15%)   |             0.20 | Excess return per unit vol     |
| Sortino Ratio           |             0.30 | Excess return per downside vol |
| Maximum Drawdown (MDD)  |          -30.62% | Peak-to-trough decline         |
| Round-trip Trades       |               15 | Wins: 6 | Losses: 9            |
| Win Rate                |            40.0% | Profit Factor: 2.95            |
| Total Friction Paid     |   PKR 117,392.62 | Brokerage + Regulatory + Slip  |
+-----------------------------------------------------------------------------+
```

#### Strategy 2: 11-Fold Walk-Forward Out-of-Sample Logistic Regression on OGDC
```powershell
uv run psx backtest --symbol OGDC --strategy logistic
```
```text
                 PSX Strategy Backtest Report: OGDC (LOGISTIC)                 
+-----------------------------------------------------------------------------+
| Portfolio / Metric      |            Value | Benchmark / Context            |
|-------------------------+------------------+--------------------------------|
| Initial Capital         | PKR 1,000,000.00 | Starting cash allocation       |
| Final Portfolio Equity  | PKR 1,237,507.49 | Ending mark-to-market value    |
| Cumulative Return       |          +23.75% | Total unannualized return      |
| CAGR (Annualized)       |           +8.75% | Compound annual growth rate    |
| Annualized Volatility   |           25.91% | Standard deviation of daily    |
| Sharpe Ratio (Rf=15%)   |            -0.09 | Excess return per unit vol     |
| Sortino Ratio           |            -0.14 | Excess return per downside vol |
| Maximum Drawdown (MDD)  |          -18.64% | Peak-to-trough decline         |
| Round-trip Trades       |               20 | Wins: 15 | Losses: 5           |
| Win Rate                |            75.0% | Profit Factor: 1.86            |
| Total Friction Paid     |   PKR 130,628.97 | Brokerage + Regulatory + Slip  |
+-----------------------------------------------------------------------------+
```

---

## 5. Phase 08 Sign-off
Phase 08 is completely verified with 78 unit tests, validated on real market data, and ready for Phase 09 (Tree-Based Ensembles: XGBoost & LightGBM).
