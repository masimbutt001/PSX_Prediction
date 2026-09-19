"""Data models and schemas for macroeconomic indicators and daily features."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class MacroIndicatorType(str, Enum):
    """Supported macroeconomic indicator series."""

    USD_PKR = "USD_PKR"
    SBP_POLICY_RATE = "SBP_POLICY_RATE"
    KIBOR_6M = "KIBOR_6M"
    CPI_YOY = "CPI_YOY"
    CPI_MOM = "CPI_MOM"
    BRENT_CRUDE = "BRENT_CRUDE"
    KSE_100 = "KSE_100"


class MacroDataPoint(BaseModel):
    """Individual macroeconomic reading with strict observation and release timestamps."""

    indicator: str
    observation_date: str = Field(description="Date/period the data represents (YYYY-MM-DD)")
    public_release_date: str = Field(
        description="Earliest date the observation became public knowledge (YYYY-MM-DD)"
    )
    value: float
    unit: str = "PERCENT"
    frequency: str = "DAILY"  # DAILY, MONTHLY, EVENT
    source: str = "SBP"
    notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class DailyMacroFeatures(BaseModel):
    """Point-in-time daily macroeconomic feature vector for a PSX trading session."""

    session_date: str
    usd_pkr: float
    usd_pkr_return_1d: float = 0.0
    sbp_policy_rate: float
    kibor_6m: float
    cpi_yoy: float
    brent_crude: float
    brent_return_1d: float = 0.0
    kse100_index: float
    kse100_return_1d: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
