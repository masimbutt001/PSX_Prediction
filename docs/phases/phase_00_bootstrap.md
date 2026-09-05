# Phase 00 — Repository Bootstrap & Tooling Configuration

## 1. Objective
Establish a clean, modern, fully testable Python project skeleton with standards-compliant packaging (`pyproject.toml`), environment management via `uv`, linting/formatting with `ruff`, unit testing with `pytest`, a type-safe Pydantic configuration loader, and an initial CLI entrypoint.

---

## 2. Deliverables & Files to Create
```text
psx_predictor/
├── pyproject.toml                     # Package metadata, uv dependencies, tools config
├── .gitignore                         # Python, data, caches, models, .env
├── .env.example                       # Documented environment variables
├── README.md                          # Project overview, setup, and usage
├── config/
│   ├── stocks.yaml                    # Initial configurable PSX stock universe
│   └── settings.yaml                  # Application settings (paths, log levels)
├── src/
│   └── psx_predictor/
│       ├── __init__.py                # Package version export
│       ├── config/
│       │   ├── __init__.py
│       │   ├── schema.py              # Pydantic models for stocks & app config
│       │   └── loader.py              # YAML loader with env override support
│       └── cli/
│           ├── __init__.py
│           └── main.py                # Typer CLI application entrypoint
└── tests/
    ├── __init__.py
    ├── conftest.py                    # Shared test fixtures
    └── unit/
        ├── __init__.py
        └── test_config.py             # Config loading & validation tests
```

---

## 3. Detailed Specifications

### 3.1 Packaging & Dependencies (`pyproject.toml`)
- **Python Version:** `>=3.12`
- **Build System:** `hatchling` (or `setuptools`)
- **Core Dependencies:**
  - `pydantic>=2.7.0`
  - `pydantic-settings>=2.2.0`
  - `pyyaml>=6.0.1`
  - `typer>=0.12.0`
  - `rich>=13.7.0`
  - `loguru>=0.7.2`
- **Dev Dependencies:**
  - `pytest>=8.0.0`
  - `ruff>=0.4.0`
  - `mypy>=1.10.0`
- **CLI Script Mapping:** `psx = "psx_predictor.cli.main:app"`
- **Ruff Configuration:** line length 100, rules `E`, `F`, `W`, `I` (isort), `B` (flake8-bugbear).

### 3.2 Configuration System (`src/psx_predictor/config/`)
- `StockConfig`:
  - `symbol: str` (validated uppercase, e.g., `OGDC`)
  - `name: str`
  - `sector: str`
  - `yahoo_ticker: str` (e.g., `OGDC.KA`)
  - `enabled: bool = True`
- `AppConfig`:
  - `data_dir: Path = Path("data")`
  - `log_level: str = "INFO"`
  - `stocks: List[StockConfig]`
- `load_config(config_dir: Optional[Path] = None) -> AppConfig`:
  - Loads `config/settings.yaml` and `config/stocks.yaml`.
  - Supports overriding with environment variables prefixed by `PSX_`.

### 3.3 Initial CLI (`psx config show`)
- Running `psx config show` outputs:
  - Formatted table (using `rich`) of all configured stocks.
  - Active data directories and storage settings.
  - Configuration health status.

---

## 4. Testing Plan
- `test_package_imports()`: Asserts version and core modules import without error.
- `test_default_config_loads()`: Loads actual `config/stocks.yaml` and verifies fields.
- `test_stock_symbol_validation()`: Verifies invalid symbols or missing fields raise Pydantic `ValidationError`.
- `test_get_enabled_stocks()`: Filters enabled vs. disabled stocks correctly.
- `test_cli_config_show()`: Runs CLI runner and asserts output contains configured symbols and exit code 0.

---

## 5. Validation Commands
```bash
# 1. Run automated tests
pytest tests/unit/test_config.py -v

# 2. Run Ruff linting & formatting check
ruff check .
ruff format --check .

# 3. Test CLI command directly
psx config show
```

---

## 6. Phase Completion Exit Criteria
- [ ] `pytest` passes with 100% success.
- [ ] `ruff check .` returns 0 warnings and 0 errors.
- [ ] `psx config show` prints the enabled stocks table clearly.
- [ ] Phase 0 completion report documented.
