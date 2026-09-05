"""Shared pytest fixtures for unit and integration testing."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from psx_predictor.config.schema import AppConfig, SettingsConfig, StockConfig


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide a Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_stock_config() -> StockConfig:
    """Fixture providing a standard valid StockConfig instance."""
    return StockConfig(
        symbol="OGDC",
        name="Oil & Gas Development Company Limited",
        sector="Oil & Gas Exploration",
        yahoo_ticker="OGDC.KA",
        enabled=True,
    )


@pytest.fixture
def sample_settings_config(tmp_path: Path) -> SettingsConfig:
    """Fixture providing a SettingsConfig with isolated temp directories."""
    return SettingsConfig(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data/raw",
        processed_dir=tmp_path / "data/processed",
        features_dir=tmp_path / "data/features",
        predictions_dir=tmp_path / "data/predictions",
        models_dir=tmp_path / "data/models",
        reports_dir=tmp_path / "data/reports",
        log_level="DEBUG",
        timezone="Asia/Karachi",
    )


@pytest.fixture
def sample_app_config(
    sample_settings_config: SettingsConfig, sample_stock_config: StockConfig
) -> AppConfig:
    """Fixture providing an AppConfig with sample settings and stocks."""
    return AppConfig(
        settings=sample_settings_config,
        stocks=[
            sample_stock_config,
            StockConfig(
                symbol="ENGRO",
                name="Engro Corporation Limited",
                sector="Fertilizer / Conglomerate",
                yahoo_ticker="ENGRO.KA",
                enabled=False,
            ),
        ],
    )
