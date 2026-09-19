"""Unit tests for multi-modal feature merging, zero look-ahead bias, and empirical ablations."""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from psx_predictor.cli.main import app
from psx_predictor.features.ablations import AblationStudyRunner
from psx_predictor.features.merger import (
    INTERACTION_FEATURE_COLUMNS,
    MACRO_FEATURE_COLUMNS,
    NEWS_FEATURE_COLUMNS,
    MultiModalFeatureMerger,
)
from psx_predictor.models.split import DEFAULT_TECHNICAL_FEATURES
from psx_predictor.storage.parquet_io import read_parquet, write_parquet_atomic
from psx_predictor.storage.paths import ensure_directories


def _create_mock_environment(tmp_path: Path) -> dict[str, Path]:
    """Create isolated test environment with synthetic technical, news, and macro data."""
    paths = ensure_directories(tmp_path)

    # 1. Synthetic technical price data for 350 days (~250 trading sessions)
    dates = [datetime.date(2025, 1, 1) + datetime.timedelta(days=i) for i in range(350)]
    # Keep only weekdays
    trading_dates = [d for d in dates if d.weekday() < 5]
    n_days = len(trading_dates)

    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = base_price * np.exp(np.cumsum(returns))

    tech_rows = []
    for i, d in enumerate(trading_dates):
        c = float(prices[i])
        o = float(c * (1.0 + np.random.normal(0, 0.005)))
        h = max(o, c) * 1.01
        l_val = min(o, c) * 0.99
        v = int(np.random.randint(100_000, 1_000_000))
        tech_rows.append(
            {
                "symbol": "OGDC",
                "trade_date": d,
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l_val, 2),
                "close": round(c, 2),
                "adjusted_close": round(c, 2),
                "volume": v,
                "dividend_amount": 0.0,
                "split_ratio": 1.0,
                "is_upper_lock": False,
                "is_lower_lock": False,
            }
        )
    df_prices = pd.DataFrame(tech_rows)
    write_parquet_atomic(df_prices, paths["processed_prices"] / "OGDC.parquet")

    # 2. Synthetic news features for OGDC
    news_rows = []
    for d in trading_dates:
        news_rows.append(
            {
                "symbol": "OGDC",
                "session_date": d.strftime("%Y-%m-%d"),
                "sentiment_24h": round(float(np.random.uniform(-0.5, 0.8)), 3),
                "sentiment_72h": round(float(np.random.uniform(-0.3, 0.5)), 3),
                "news_count_24h": int(np.random.poisson(2)),
                "news_count_72h": int(np.random.poisson(5)),
                "positive_count_24h": 1,
                "negative_count_24h": 0,
                "has_earnings_announcement_today": 1.0 if d == trading_dates[20] else 0.0,
                "has_dividend_announcement_today": 0.0,
                "has_discovery_announcement_today": 0.0,
            }
        )
    df_news = pd.DataFrame(news_rows)
    write_parquet_atomic(df_news, paths["features_news"] / "OGDC.parquet")

    # 3. Synthetic macro daily features
    macro_rows = []
    for d in trading_dates:
        macro_rows.append(
            {
                "session_date": d.strftime("%Y-%m-%d"),
                "usd_pkr": 278.50,
                "usd_pkr_return_1d": 0.0002,
                "sbp_policy_rate": 19.50,
                "kibor_6m": 20.15,
                "cpi_yoy": 9.60,
                "brent_crude": 78.40,
                "brent_return_1d": -0.005,
                "kse100_index": 75400.0,
                "kse100_return_1d": 0.0012,
            }
        )
    df_macro = pd.DataFrame(macro_rows)
    write_parquet_atomic(df_macro, paths["processed_macro"] / "macro_daily.parquet")

    return paths


def test_asof_join_does_not_peek_future(tmp_path: Path) -> None:
    """Verify that retrospective alignment guarantees future macro releases cannot leak."""
    merger = MultiModalFeatureMerger(storage_paths=ensure_directories(tmp_path))

    # Target stock sessions from Jan 15 to Feb 10
    sessions = pd.date_range("2026-01-15", "2026-02-10", freq="B")
    target_df = pd.DataFrame(
        {
            "session_date": [d.strftime("%Y-%m-%d") for d in sessions],
            "log_ret_1d": [0.01] * len(sessions),
        }
    )

    # Macro series with a January CPI released on Feb 01
    # Before Feb 01, CPI is 20.0 (Dec CPI). On Feb 01, Jan CPI of 15.0 is released.
    macro_df = pd.DataFrame(
        [
            {
                "session_date": "2026-01-01",
                "cpi_yoy": 20.0,
                "usd_pkr": 280.0,
                "usd_pkr_return_1d": 0.0,
                "sbp_policy_rate": 22.0,
                "kibor_6m": 22.5,
                "brent_crude": 75.0,
                "brent_return_1d": 0.0,
                "kse100_index": 65000.0,
                "kse100_return_1d": 0.0,
            },
            {
                "session_date": "2026-02-01",
                "cpi_yoy": 15.0,  # Jan CPI officially released Feb 01
                "usd_pkr": 279.0,
                "usd_pkr_return_1d": -0.003,
                "sbp_policy_rate": 22.0,
                "kibor_6m": 22.4,
                "brent_crude": 76.0,
                "brent_return_1d": 0.01,
                "kse100_index": 66000.0,
                "kse100_return_1d": 0.01,
            },
        ]
    )

    merged = merger._merge_macro_retrospective(target_df, macro_df)

    # Sessions in January MUST have cpi_yoy == 20.0 (NEVER 15.0)
    jan_sessions = merged[merged["session_date"] <= "2026-01-31"]
    assert not jan_sessions.empty
    assert (jan_sessions["cpi_yoy"] == 20.0).all(), "Future Jan CPI leaked into Jan sessions!"

    # Sessions on or after Feb 01 observe the updated 15.0 CPI
    feb_sessions = merged[merged["session_date"] >= "2026-02-01"]
    assert not feb_sessions.empty
    assert (feb_sessions["cpi_yoy"] == 15.0).all()


def test_merger_combines_all_modalities(tmp_path: Path) -> None:
    """Test full multi-modal merge combining technical, news, macro, and interaction layers."""
    paths = _create_mock_environment(tmp_path)
    merger = MultiModalFeatureMerger(storage_paths=paths)

    df_combined = merger.merge_symbol("OGDC", save=True)

    # 1. Output checks
    assert not df_combined.empty
    assert len(df_combined) > 50

    # 2. Modality column checks
    # Technical features
    for col in ["open", "high", "low", "close", "volume", "rsi_14", "macd_line"]:
        assert col in df_combined.columns, f"Missing tech column: {col}"

    # News features
    for col in NEWS_FEATURE_COLUMNS:
        assert col in df_combined.columns, f"Missing news column: {col}"

    # Macro features
    for col in MACRO_FEATURE_COLUMNS:
        assert col in df_combined.columns, f"Missing macro column: {col}"

    # Interaction & Relative features
    for col in INTERACTION_FEATURE_COLUMNS:
        assert col in df_combined.columns, f"Missing interaction column: {col}"

    # Supervised prediction targets
    assert "target_next_day_dir" in df_combined.columns

    # 3. File existence check
    saved_file = paths["features_combined"] / "OGDC_combined.parquet"
    assert saved_file.exists()
    disk_df = read_parquet(saved_file)
    assert len(disk_df) == len(df_combined)


def test_merger_handles_missing_news_gracefully(tmp_path: Path) -> None:
    """Verify merger handles symbols with no news features without crashing or NaNs."""
    paths = _create_mock_environment(tmp_path)

    # Delete news file
    news_file = paths["features_news"] / "OGDC.parquet"
    if news_file.exists():
        news_file.unlink()

    merger = MultiModalFeatureMerger(storage_paths=paths)
    df_combined = merger.merge_symbol("OGDC", save=False)

    assert not df_combined.empty
    for col in NEWS_FEATURE_COLUMNS:
        assert col in df_combined.columns
        assert (df_combined[col] == 0.0).all(), f"Expected {col} to default to 0.0"


def test_ablation_configurations_load(tmp_path: Path) -> None:
    """Verify all 5 ablation configurations load correct feature column sets."""
    paths = _create_mock_environment(tmp_path)
    merger = MultiModalFeatureMerger(storage_paths=paths)
    df = merger.merge_symbol("OGDC", save=False)

    runner = AblationStudyRunner(storage_paths=paths)
    configs = runner.get_standard_ablation_configs(df.columns.tolist())

    assert len(configs) == 5
    ids = [c.config_id for c in configs]
    assert ids == ["EXP_1", "EXP_2", "EXP_3", "EXP_4", "EXP_5"]

    # EXP_1: Price Only
    exp1 = configs[0]
    assert exp1.name == "Price Only"
    for col in exp1.feature_columns:
        assert col in DEFAULT_TECHNICAL_FEATURES

    # EXP_2: Price + Relative Market
    exp2 = configs[1]
    assert "kse100_return_1d" in exp2.feature_columns
    assert len(exp2.feature_columns) > len(exp1.feature_columns)

    # EXP_3: Price + News
    exp3 = configs[2]
    assert "sentiment_24h" in exp3.feature_columns
    assert len(exp3.feature_columns) > len(exp1.feature_columns)

    # EXP_4: Price + Macro
    exp4 = configs[3]
    assert "usd_pkr" in exp4.feature_columns
    assert "sbp_policy_rate" in exp4.feature_columns
    assert len(exp4.feature_columns) > len(exp1.feature_columns)

    # EXP_5: Full Multi-Modal
    exp5 = configs[4]
    assert len(exp5.feature_columns) >= max(len(exp2.feature_columns), len(exp4.feature_columns))


def test_ablation_study_runner_execution(tmp_path: Path) -> None:
    """Verify AblationStudyRunner executes all 5 configs and computes deltas."""
    paths = _create_mock_environment(tmp_path)
    runner = AblationStudyRunner(storage_paths=paths)

    results = runner.run_ablation_study(
        symbol="OGDC",
        model_name="logistic",
        target_col="target_next_day_dir",
        train_ratio=0.75,
    )

    assert len(results) == 5

    # Check identical sample size across all folds
    n_train_baseline = results[0].n_train
    n_test_baseline = results[0].n_test
    for r in results:
        assert r.n_train == n_train_baseline, "Train sample size shifted across configs!"
        assert r.n_test == n_test_baseline, "Test sample size shifted across configs!"
        assert 0.0 <= r.accuracy <= 1.0

    # EXP_1 baseline delta must be 0
    assert results[0].delta_accuracy == 0.0
    assert results[0].delta_sharpe == 0.0
    assert results[0].delta_total_return == 0.0

    # Non-baseline deltas exist as valid floats
    for r in results[1:]:
        assert isinstance(r.delta_accuracy, float)
        assert isinstance(r.delta_sharpe, float)


def test_cli_features_merge_and_ablation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test CLI commands 'psx features merge' and 'psx model ablation' via CliRunner."""
    monkeypatch.setenv("PSX_DATA_DIR", str(tmp_path))
    _create_mock_environment(tmp_path)

    runner = CliRunner()

    # 1. Test features merge
    res_merge = runner.invoke(app, ["features", "merge", "--symbol", "OGDC"])
    assert res_merge.exit_code == 0
    assert "Multi-Modal Feature Merge Summary" in res_merge.stdout
    assert "SUCCESS" in res_merge.stdout

    # 2. Test model ablation
    res_ablation = runner.invoke(
        app, ["model", "ablation", "--symbol", "OGDC", "--model", "logistic"]
    )
    assert res_ablation.exit_code == 0
    assert "Empirical Multi-Modal Ablation Study: OGDC" in res_ablation.stdout
    assert "Price Only" in res_ablation.stdout
    assert "Price + News" in res_ablation.stdout
    assert "Price + Macro" in res_ablation.stdout
