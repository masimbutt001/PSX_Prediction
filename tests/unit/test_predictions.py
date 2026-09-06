"""Unit tests for Phase 10: Prediction Registry, Live Predictor, and Audit Engine."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from psx_predictor.cli.main import app
from psx_predictor.predictions.auditor import PredictionAuditor
from psx_predictor.predictions.predictor import LivePredictor
from psx_predictor.predictions.registry import PredictionRecord, PredictionRegistry
from psx_predictor.storage.parquet_io import write_parquet_atomic


@pytest.fixture
def temp_storage(tmp_path: Path) -> dict[str, Path]:
    """Provide isolated directory paths for storage testing."""
    paths = {
        "raw": tmp_path / "raw",
        "processed_prices": tmp_path / "processed" / "prices",
        "features_technical": tmp_path / "features" / "technical",
        "predictions": tmp_path / "predictions",
        "models": tmp_path / "models",
        "reports": tmp_path / "reports",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


@pytest.fixture
def sample_feature_data() -> pd.DataFrame:
    """Generate 60 sessions of synthetic technical features with targets."""
    dates = pd.date_range(start="2024-01-01", periods=60, freq="B")
    np.random.seed(42)
    closes = 100.0 + np.cumsum(np.random.randn(60) * 1.5)

    data = {
        "trade_date": [d.strftime("%Y-%m-%d") for d in dates],
        "open": closes - np.random.uniform(0, 1, 60),
        "high": closes + np.random.uniform(0.5, 2, 60),
        "low": closes - np.random.uniform(0.5, 2, 60),
        "close": closes,
        "volume": np.random.randint(100_000, 1_000_000, 60),
        # 45 standard features
        "returns_1d": np.random.randn(60) * 0.01,
        "returns_5d": np.random.randn(60) * 0.02,
        "returns_10d": np.random.randn(60) * 0.03,
        "returns_20d": np.random.randn(60) * 0.04,
        "sma_ratio_5": np.random.uniform(0.98, 1.02, 60),
        "sma_ratio_10": np.random.uniform(0.98, 1.02, 60),
        "sma_ratio_20": np.random.uniform(0.98, 1.02, 60),
        "sma_ratio_50": np.random.uniform(0.98, 1.02, 60),
        "ema_ratio_12": np.random.uniform(0.98, 1.02, 60),
        "ema_ratio_26": np.random.uniform(0.98, 1.02, 60),
        "rsi_14": np.random.uniform(30, 70, 60),
        "volatility_20d": np.random.uniform(0.01, 0.03, 60),
        "atr_14": np.random.uniform(1.0, 3.0, 60),
        "volume_ratio_20": np.random.uniform(0.8, 1.5, 60),
        "target_next_day_dir": [1 if i % 2 == 0 else 0 for i in range(60)],
    }
    df = pd.DataFrame(data)
    # The very last session has NaN target in live trading!
    df.loc[59, "target_next_day_dir"] = np.nan
    return df


class TestPredictionRecord:
    def test_record_instantiation_defaults(self) -> None:
        rec = PredictionRecord(
            symbol="OGDC",
            target_date="2024-03-25",
            model_name="XGBoost",
            up_probability=0.62,
            signal="BUY",
            confidence=0.24,
        )
        assert rec.prediction_id is not None
        assert rec.symbol == "OGDC"
        assert rec.up_probability == 0.62
        assert rec.signal == "BUY"
        assert rec.realized_outcome is None
        assert rec.is_correct is None

    def test_record_probability_validation(self) -> None:
        with pytest.raises(ValueError):
            PredictionRecord(
                symbol="OGDC",
                target_date="2024-03-25",
                model_name="XGBoost",
                up_probability=1.5,  # Out of [0, 1]
                signal="BUY",
            )


class TestPredictionRegistry:
    def test_log_and_query_predictions(self, temp_storage: dict[str, Path]) -> None:
        registry = PredictionRegistry(storage_paths=temp_storage)

        rec1 = PredictionRecord(
            symbol="OGDC",
            target_date="2024-03-25",
            model_name="XGBoost",
            up_probability=0.55,
            signal="BUY",
        )
        rec2 = PredictionRecord(
            symbol="PPL",
            target_date="2024-03-25",
            model_name="Random Forest",
            up_probability=0.42,
            signal="SELL",
        )

        registry.log_predictions([rec1, rec2])

        df_all = registry.get_predictions()
        assert len(df_all) == 2
        assert set(df_all["symbol"]) == {"OGDC", "PPL"}

        df_ogdc = registry.get_predictions(symbol="OGDC")
        assert len(df_ogdc) == 1
        assert df_ogdc.iloc[0]["symbol"] == "OGDC"

    def test_deduplication_on_repeated_log(self, temp_storage: dict[str, Path]) -> None:
        registry = PredictionRegistry(storage_paths=temp_storage)
        rec = PredictionRecord(
            symbol="OGDC",
            target_date="2024-03-25",
            model_name="XGBoost",
            up_probability=0.55,
            signal="BUY",
        )
        registry.log_prediction(rec)
        registry.log_prediction(rec)  # Same UUID

        df = registry.get_predictions()
        assert len(df) == 1


class TestLivePredictor:
    def test_generate_prediction_success(
        self, temp_storage: dict[str, Path], sample_feature_data: pd.DataFrame
    ) -> None:
        feat_path = temp_storage["features_technical"] / "OGDC_tech_features.parquet"
        write_parquet_atomic(sample_feature_data, feat_path)

        predictor = LivePredictor(storage_paths=temp_storage)
        rec = predictor.generate_prediction(
            symbol="OGDC", model_name="xgboost", log_to_registry=True
        )

        assert rec.symbol == "OGDC"
        assert rec.model_name == "XGBoost"
        assert 0.0 <= rec.up_probability <= 1.0
        assert rec.signal in ("BUY", "SELL", "HOLD")

        # Verify it was logged into registry
        reg = PredictionRegistry(storage_paths=temp_storage)
        df = reg.get_predictions()
        assert len(df) == 1
        assert df.iloc[0]["prediction_id"] == rec.prediction_id

    def test_missing_features_raises_error(self, temp_storage: dict[str, Path]) -> None:
        predictor = LivePredictor(storage_paths=temp_storage)
        with pytest.raises(FileNotFoundError):
            predictor.generate_prediction(symbol="UNKNOWN")


class TestPredictionAuditor:
    def test_reconcile_outcomes_and_brier_score(self, temp_storage: dict[str, Path]) -> None:
        registry = PredictionRegistry(storage_paths=temp_storage)

        # 1. Create 2 predictions: 2024-01-03 (UP forecast) and 2024-01-04 (DOWN forecast)
        rec1 = PredictionRecord(
            symbol="OGDC",
            target_date="2024-01-03",
            model_name="XGBoost",
            up_probability=0.80,
            signal="BUY",
        )
        rec2 = PredictionRecord(
            symbol="OGDC",
            target_date="2024-01-04",
            model_name="XGBoost",
            up_probability=0.20,
            signal="SELL",
        )
        rec_pending = PredictionRecord(
            symbol="OGDC",
            target_date="2025-01-01",  # Future date not in prices
            model_name="XGBoost",
            up_probability=0.55,
            signal="BUY",
        )
        registry.log_predictions([rec1, rec2, rec_pending])

        # 2. Create realized price dataset
        # 2024-01-02 Close: 100.0
        # 2024-01-03 Close: 102.0 (+2.0% UP -> rec1 BUY is correct!)
        # 2024-01-04 Close: 99.0 (-2.94% DOWN -> rec2 SELL is correct!)
        price_df = pd.DataFrame(
            {
                "trade_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "open": [99.0, 101.0, 100.0],
                "high": [101.0, 103.0, 101.0],
                "low": [98.0, 100.0, 98.0],
                "close": [100.0, 102.0, 99.0],
                "volume": [500_000, 600_000, 450_000],
            }
        )
        proc_path = temp_storage["processed_prices"] / "OGDC.parquet"
        write_parquet_atomic(price_df, proc_path)

        # 3. Run Auditor
        auditor = PredictionAuditor(storage_paths=temp_storage)
        df_audited, summary = auditor.reconcile(symbol="OGDC")

        assert summary.total_records == 3
        assert summary.resolved_records == 2
        assert summary.pending_records == 1
        assert summary.hit_rate == 1.0  # Both BUY and SELL were correct!
        assert summary.brier_score is not None
        # rec1: prob=0.8, outcome=1 -> (0.8-1)^2 = 0.04
        # rec2: prob=0.2, outcome=0 -> (0.2-0)^2 = 0.04
        # Mean Brier = 0.04
        assert pytest.approx(summary.brier_score, rel=1e-3) == 0.04


class TestCLIIntegration:
    def test_predict_and_audit_cli(
        self, temp_storage: dict[str, Path], sample_feature_data: pd.DataFrame
    ) -> None:
        runner = CliRunner()
        # Test predict --help
        res_help = runner.invoke(app, ["predict", "--help"])
        assert res_help.exit_code == 0
        assert "audit" in res_help.stdout
        assert "generate" in res_help.stdout
