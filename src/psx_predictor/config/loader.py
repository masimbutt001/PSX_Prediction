"""Configuration file loader with environment variable override support."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

from psx_predictor.config.schema import AppConfig, SettingsConfig, StockConfig


def find_default_config_dir() -> Path:
    """Locate the default configuration directory relative to project root."""
    # Try relative to current working directory
    cwd_config = Path.cwd() / "config"
    if cwd_config.exists() and (cwd_config / "stocks.yaml").exists():
        return cwd_config

    # Try relative to package directory
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent
    pkg_config = pkg_root / "config"
    if pkg_config.exists() and (pkg_config / "stocks.yaml").exists():
        return pkg_config

    return Path("config")


def _apply_env_overrides(settings_dict: dict[str, Any]) -> dict[str, Any]:
    """Override configuration values with PSX_ prefixed environment variables."""
    overrides = {
        "PSX_DATA_DIR": "data_dir",
        "PSX_RAW_DIR": "raw_dir",
        "PSX_PROCESSED_DIR": "processed_dir",
        "PSX_FEATURES_DIR": "features_dir",
        "PSX_PREDICTIONS_DIR": "predictions_dir",
        "PSX_MODELS_DIR": "models_dir",
        "PSX_REPORTS_DIR": "reports_dir",
        "PSX_LOG_LEVEL": "log_level",
        "PSX_TIMEZONE": "timezone",
    }
    for env_var, key in overrides.items():
        val = os.getenv(env_var)
        if val is not None and val.strip():
            settings_dict[key] = val.strip()
    return settings_dict


def load_config(config_dir: Optional[Path] = None) -> AppConfig:
    """Load application configuration and stock universe from YAML files.

    Args:
        config_dir: Optional custom path to directory containing YAML files.

    Returns:
        Validated AppConfig instance.
    """
    target_dir = config_dir or find_default_config_dir()

    settings_file = target_dir / "settings.yaml"
    stocks_file = target_dir / "stocks.yaml"

    settings_data: dict[str, Any] = {}
    if settings_file.exists():
        with open(settings_file, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            if isinstance(raw, dict):
                extracted = raw.get("settings", raw)
                if isinstance(extracted, dict):
                    settings_data = extracted
    else:
        logger.warning(f"Settings file not found at {settings_file}. Using defaults.")

    settings_data = _apply_env_overrides(settings_data)
    settings_config = SettingsConfig(**settings_data)

    stocks_list: list[StockConfig] = []
    if stocks_file.exists():
        with open(stocks_file, "r", encoding="utf-8") as f:
            raw_stocks = yaml.safe_load(f)
            if isinstance(raw_stocks, dict):
                raw_list = raw_stocks.get("stocks", [])
                if isinstance(raw_list, list):
                    for item in raw_list:
                        if isinstance(item, dict):
                            stocks_list.append(StockConfig(**item))
    else:
        logger.warning(f"Stocks file not found at {stocks_file}. Stock universe is empty.")

    return AppConfig(settings=settings_config, stocks=stocks_list)
