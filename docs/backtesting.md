# Backtesting Methodology & Trading Simulation — PSX Platform

## 1. Simulation Principles

The backtesting engine simulates realistic market execution for model-generated signals. Academic backtests often overstate returns by assuming costless, frictionless execution and instant liquidity. Our backtesting engine addresses this by directly modeling PSX-specific constraints.

---

## 2. PSX Market Microstructure Constraints

### 2.1 Circuit Breakers & Lock Limits
The Pakistan Stock Exchange enforces a daily price fluctuation limit (typically $\pm 7.5\%$ or 1 PKR, whichever is higher, from the previous closing price).

- **Upper Lock (Limit Up):**
  If a stock reaches $+7.5\%$, sellers disappear, and buy orders queue up with zero or negligible fill rates.
  *Rule:* The backtester **strictly prohibits executing buy orders** at the close of an upper lock session.
  
- **Lower Lock (Limit Down):**
  If a stock reaches $-7.5\%$, buyers disappear.
  *Rule:* The backtester **strictly prohibits executing sell orders** at the close of a lower lock session.

### 2.2 Realistic Transaction Friction & Taxes
Every simulation deducts realistic Pakistani market trading costs:

| Cost Item | Rate / Basis | Notes |
| :--- | :--- | :--- |
| **Brokerage Commission** | 0.15% per side | Typical institutional / retail rate (or 3–5 paisas/share) |
| **SECP Transaction Levy** | 0.003% per side | Regulatory fee |
| **CDC Charges** | 0.005% per side | Central Depository Company clearing |
| **NCCPL Charges** | 0.005% per side | National Clearing Company settlement |
| **Capital Gains Tax (CGT)** | Configurable (15% standard)| Applied to realized net gains based on holding period |
| **Estimated Slippage** | 0.10% | Execution slippage from signal price to fill price |

*Total round-trip friction assumed by default:* $\approx 0.50\%$ of traded capital.

---

## 3. Walk-Forward Simulation Architecture

The backtester integrates directly with the walk-forward evaluation splits:

1. **Step-by-step rolling evaluation:**
   For each test window (e.g. Q1 2024), the model is trained strictly on data prior to Q1 2024.
2. **Signal Generation:**
   On each session close $t$, the model generates a probability $P(Up)$ and signal $S_t \in \{\text{BUY}, \text{SELL}, \text{HOLD}\}$.
3. **Order Execution:**
   Simulated execution occurs at the open of session $t+1$ (or close of session $t$ with slippage), subject to lock-limit restrictions.
4. **Portfolio State:**
   Tracks cash, equity holdings, realized PnL, unrealized PnL, and maximum drawdown day-by-day.

---

## 4. Evaluation Metrics

The backtester computes both statistical accuracy and portfolio performance:

### Quantitative Performance Metrics:
- **Total Return:** $(V_{end} - V_{start}) / V_{start}$
- **Compound Annual Growth Rate (CAGR):** Annualized geometric return.
- **Benchmark Comparison:** Relative performance vs. Buy-and-Hold of the underlying stock and the KSE-100 index.
- **Sharpe Ratio:** Annualized risk-adjusted excess return over risk-free rate (6-month KIBOR).
- **Sortino Ratio:** Focuses solely on downside volatility.
- **Maximum Drawdown (MDD):** Peak-to-trough decline percentage and duration in days.
- **Win Rate:** Percentage of closed trades with positive net return.
- **Profit Factor:** Gross profits divided by gross losses.
- **Average Trade Duration:** Holding period in sessions.
- **Turnover:** Frequency of portfolio rebalancing.
