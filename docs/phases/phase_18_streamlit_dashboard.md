# Phase 18 — Streamlit Research Dashboard

## 1. Objective
Build an interactive, intuitive research dashboard using Streamlit. The dashboard visualizes market overviews, candlestick charts with technical indicators, latest model forecasts with explainable factors, and historical backtest performance curves.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/dashboard/
├── __init__.py
├── app.py                             # Streamlit multi-page application entrypoint
├── components/
│   ├── __init__.py
│   ├── charts.py                      # Candlestick & indicator plots (Plotly)
│   ├── prediction_card.py             # Forecast probability, signal, and SHAP drivers
│   └── metrics_table.py               # Backtest performance & benchmark metrics
└── pages/
    ├── 1_Market_Overview.py           # Universal market heatmap & latest signals
    ├── 2_Stock_Analysis.py            # Deep-dive individual stock profile
    └── 3_Model_Performance.py         # Walk-forward backtest equity curves and drawdowns
```

---

## 3. Detailed Specifications

### 3.1 Views
1. **Market Overview:**
   - Table of configured stocks, latest prices, 1-day change, model signal (`BUY`, `HOLD`, `SELL`), confidence gauge.
2. **Stock Profile:**
   - Interactive Plotly candlestick chart with volume, Bollinger Bands, and SMA overlays.
   - Latest prediction breakdown: Up-probability gauge, expected 5-day return, and bulleted SHAP driver explanations.
   - Recent relevant news and corporate announcements list.
3. **Model Performance:**
   - Cumulative strategy equity curve vs. Buy-and-Hold and KSE-100 index.
   - Underwater drawdown plot.
   - Metrics card: Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor.

---

## 4. Testing Plan
- `test_dashboard_imports_cleanly()`: Verifies Streamlit pages and custom components import without runtime errors.
- `test_chart_generator()`: Asserts Plotly chart object creates successfully with sample OHLCV dataframe.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_dashboard.py -v
psx dashboard --dry-run
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] Multi-page Streamlit application loads without errors.
- [ ] Interactive charts, prediction explainers, and backtest visualizer functional.
- [ ] Phase 18 completion report documented.
