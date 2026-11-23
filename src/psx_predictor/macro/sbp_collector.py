"""Collector for State Bank of Pakistan (SBP) policy rates, KIBOR, and PBS CPI inflation.

The figures are read from ``config/macro_rates.yaml`` so they can be updated by hand when SBP
announces a policy decision or PBS publishes CPI, without changing code.
"""

from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

from psx_predictor.config.loader import find_default_config_dir
from psx_predictor.macro.schemas import MacroDataPoint, MacroIndicatorType

MACRO_RATES_FILENAME = "macro_rates.yaml"


def load_macro_rates(
    rates_file: Optional[Path] = None,
) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float, float]], float]:
    """Load SBP policy rates, CPI releases and the KIBOR spread from YAML.

    Args:
        rates_file: Path to the YAML file. Defaults to ``<config dir>/macro_rates.yaml``.

    Returns:
        (policy_rates, cpi_releases, kibor_spread) where policy_rates holds
        (observation_date, release_date, rate_pct) and cpi_releases holds
        (observation_date, release_date, yoy_pct, mom_pct).
    """
    path = rates_file or (find_default_config_dir() / MACRO_RATES_FILENAME)
    if not path.exists():
        raise FileNotFoundError(
            f"Macro rates file not found: {path}. "
            "Create it from the repository's config/macro_rates.yaml."
        )
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    rates = [
        (str(r["observation_date"]), str(r["release_date"]), float(r["rate_pct"]))
        for r in data.get("sbp_policy_rates") or []
    ]
    cpi = [
        (
            str(r["observation_date"]),
            str(r["release_date"]),
            float(r["yoy_pct"]),
            float(r["mom_pct"]),
        )
        for r in data.get("cpi_releases") or []
    ]
    spread = float(data.get("kibor_spread_over_policy", 0.65))
    return rates, cpi, spread


class SBPMacroCollector:
    """Collects State Bank of Pakistan policy decisions, KIBOR, and PBS CPI data points."""

    def __init__(self, timeout: float = 10.0, rates_file: Optional[Path] = None) -> None:
        self.timeout = timeout
        self.rates_file = rates_file
        self._loaded: Optional[
            tuple[list[tuple[str, str, float]], list[tuple[str, str, float, float]], float]
        ] = None

    def _rates(
        self,
    ) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float, float]], float]:
        if self._loaded is None:
            self._loaded = load_macro_rates(self.rates_file)
        return self._loaded

    def fetch_policy_rates(self) -> list[MacroDataPoint]:
        """Retrieve SBP policy rate announcement history with exact release dates."""
        points: list[MacroDataPoint] = []
        policy_rates, _, _ = self._rates()
        for obs, rel, rate in policy_rates:
            points.append(
                MacroDataPoint(
                    indicator=MacroIndicatorType.SBP_POLICY_RATE.value,
                    observation_date=obs,
                    public_release_date=rel,
                    value=rate,
                    unit="PERCENT",
                    frequency="EVENT",
                    source="State Bank of Pakistan",
                    notes="SBP Target Policy Rate announced via MPC Statement",
                )
            )
        logger.info(f"Loaded {len(points)} SBP policy rate decisions.")
        return points

    def fetch_kibor_rates(self) -> list[MacroDataPoint]:
        """Estimate 6-Month KIBOR as the policy rate plus the configured spread.

        This is an approximation, not collected KIBOR data.
        """
        points: list[MacroDataPoint] = []
        policy_rates, _, spread = self._rates()
        for obs, rel, rate in policy_rates:
            kibor_val = round(rate + spread, 2)
            points.append(
                MacroDataPoint(
                    indicator=MacroIndicatorType.KIBOR_6M.value,
                    observation_date=obs,
                    public_release_date=rel,
                    value=kibor_val,
                    unit="PERCENT",
                    frequency="DAILY",
                    source="Estimated from SBP policy rate",
                    notes=f"Estimated 6-Month KIBOR: policy rate + {spread} spread",
                )
            )
        logger.info(f"Loaded {len(points)} KIBOR 6M data points.")
        return points

    def fetch_cpi_inflation(self) -> list[MacroDataPoint]:
        """Retrieve monthly headline CPI inflation figures with public release dates."""
        points: list[MacroDataPoint] = []
        _, cpi_releases, _ = self._rates()
        for obs, rel, yoy, mom in cpi_releases:
            # YoY CPI
            points.append(
                MacroDataPoint(
                    indicator=MacroIndicatorType.CPI_YOY.value,
                    observation_date=obs,
                    public_release_date=rel,
                    value=yoy,
                    unit="PERCENT",
                    frequency="MONTHLY",
                    source="Pakistan Bureau of Statistics",
                    notes=f"National CPI YoY for period ending {obs}",
                )
            )
            # MoM CPI
            points.append(
                MacroDataPoint(
                    indicator=MacroIndicatorType.CPI_MOM.value,
                    observation_date=obs,
                    public_release_date=rel,
                    value=mom,
                    unit="PERCENT",
                    frequency="MONTHLY",
                    source="Pakistan Bureau of Statistics",
                    notes=f"National CPI MoM for period ending {obs}",
                )
            )
        logger.info(f"Loaded {len(points)} PBS CPI inflation data points.")
        return points

    def fetch_all(self) -> list[MacroDataPoint]:
        """Collect all SBP and PBS macroeconomic data points."""
        return (
            self.fetch_policy_rates()
            + self.fetch_kibor_rates()
            + self.fetch_cpi_inflation()
        )
