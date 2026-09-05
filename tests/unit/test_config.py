"""Unit tests for configuration loading, validation, schemas, and CLI commands."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import psx_predictor
from psx_predictor.cli.main import app
from psx_predictor.config.loader import load_config
from psx_predictor.config.schema import AppConfig, SettingsConfig, StockConfig


def test_package_imports() -> None:
    """Assert package version is set and core modules import properly."""
    assert psx_predictor.__version__ == "0.1.0"
    assert AppConfig is not None
    assert SettingsConfig is not None
    assert StockConfig is not None


def test_default_config_loads() -> None:
    """Verify default repository YAML configurations load without errors."""
    config = load_config()
    assert config is not None
    assert len(config.stocks) >= 10

    symbols = [s.symbol for s in config.stocks]
    assert "OGDC" in symbols
    assert "PPL" in symbols
    assert "ENGRO" in symbols
    assert "LUCK" in symbols

    # Verify field values of OGDC
    ogdc = config.get_stock("OGDC")
    assert ogdc is not None
    assert ogdc.yahoo_ticker == "OGDC.KA"
    assert ogdc.sector == "Oil & Gas Exploration"
    assert ogdc.enabled is True


def test_stock_symbol_validation() -> None:
    """Verify stock symbol normalization to uppercase and error on invalid values."""
    stock = StockConfig(
        symbol="  ogdc  ",
        name="Oil and Gas",
        sector="Energy",
        yahoo_ticker="OGDC.KA",
    )
    assert stock.symbol == "OGDC"

    # Empty symbol should fail
    with pytest.raises(ValidationError):
        StockConfig(
            symbol="",
            name="Empty",
            sector="Sector",
            yahoo_ticker="EMPTY.KA",
        )

    # Non-alphanumeric symbol should fail
    with pytest.raises(ValidationError):
        StockConfig(
            symbol="OGDC@123",
            name="Invalid",
            sector="Sector",
            yahoo_ticker="INVALID.KA",
        )


def test_stock_immutability(sample_stock_config: StockConfig) -> None:
    """Verify StockConfig models are frozen/immutable."""
    with pytest.raises(ValidationError):
        # Setting attributes on frozen model should raise ValidationError
        sample_stock_config.symbol = "NEW"  # type: ignore[misc]


def test_get_enabled_stocks(sample_app_config: AppConfig) -> None:
    """Verify get_enabled_stocks returns only stocks where enabled=True."""
    enabled = sample_app_config.get_enabled_stocks()
    assert len(enabled) == 1
    assert enabled[0].symbol == "OGDC"

    # Search case-insensitively
    stock = sample_app_config.get_stock("engro")
    assert stock is not None
    assert stock.symbol == "ENGRO"
    assert stock.enabled is False

    # Non-existent stock returns None
    assert sample_app_config.get_stock("NONEXISTENT") is None


def test_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify environment variables override default settings values."""
    custom_data = str(tmp_path / "custom_data")
    monkeypatch.setenv("PSX_DATA_DIR", custom_data)
    monkeypatch.setenv("PSX_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("PSX_TIMEZONE", "Asia/Karachi")

    config = load_config()
    assert str(config.settings.data_dir) == custom_data
    assert config.settings.log_level == "DEBUG"
    assert config.settings.timezone == "Asia/Karachi"


def test_invalid_log_level() -> None:
    """Verify invalid log level strings raise ValidationError."""
    with pytest.raises(ValidationError):
        SettingsConfig(log_level="SUPER_VERBOSE")


def test_cli_version(cli_runner: CliRunner) -> None:
    """Test CLI version command."""
    result = cli_runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_config_show(cli_runner: CliRunner) -> None:
    """Test CLI config show command displays universe and settings."""
    result = cli_runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "OGDC" in result.output
    assert "PPL" in result.output
    assert "ENGRO" in result.output
    assert "Active Settings" in result.output
    assert "ENABLED" in result.output


def test_cli_config_show_custom_dir(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test CLI config show with custom config directory."""
    empty_dir = tmp_path / "empty_config"
    empty_dir.mkdir()

    result = cli_runner.invoke(app, ["config", "show", "--config-dir", str(empty_dir)])
    assert result.exit_code == 0
    assert "0 active / 0 total" in result.output
