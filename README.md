# Pakistan Stock Exchange (PSX) Prediction Platform

An end-to-end quantitative analytics and predictive modeling system designed specifically for the Pakistan Stock Exchange (PSX), incorporating local market microstructures (circuit breakers, high-dividend corporate actions, Friday session splits), historical price data, macro factors, and news sentiment.

## Architecture Highlights
- **Local-First Analytical Engine:** Apache Parquet storage with DuckDB in-process SQL execution.
- **Zero Look-Ahead Bias:** Strict point-in-time joins and walk-forward chronological backtesting.
- **PSX Microstructure Fidelity:** Explicit accounting for ±7.5% price limit locks and Friday prayer breaks.

## Getting Started

### Prerequisites
- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### Installation
```bash
# 1. Clone repository
git clone https://github.com/masimbutt001/PSX_Prediction.git
cd PSX_Prediction

# 2. Create virtual environment and install package in editable mode with dev dependencies
uv venv --python 3.12 .venv
.venv\Scripts\activate  # On Windows PowerShell
uv pip install -e ".[dev]"
```

### CLI Commands
```bash
# Verify version
psx version

# Inspect configured stock universe and directories
psx config show
```

### Running Tests and Linting
```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
