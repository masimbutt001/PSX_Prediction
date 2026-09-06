"""Optional live integration tests for external collectors.

Run explicitly with:
    uv run pytest tests/integration/test_live_collector.py -m external -v
"""

import datetime

import pytest

from psx_predictor.collectors.composite import CompositeCollector
from psx_predictor.collectors.scs_collector import SCSTradeCollector


@pytest.mark.external
def test_live_scs_collector_ogdc():
    """Live connectivity test verifying SCS Trade API returns real OGDC data."""
    collector = SCSTradeCollector()
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=14)

    df = collector.fetch_historical(
        symbol="OGDC",
        start_date=start_date,
        end_date=end_date,
        save_raw_payload=False,
    )

    assert not df.empty
    assert df["symbol"].iloc[0] == "OGDC"
    assert "close" in df.columns
    assert "volume" in df.columns
    assert df["close"].iloc[-1] > 0


@pytest.mark.external
def test_live_composite_collector():
    """Live connectivity test verifying Composite collector fetches and normalizes."""
    collector = CompositeCollector()
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=7)

    df = collector.fetch_historical(
        symbol="OGDC",
        start_date=start_date,
        end_date=end_date,
        save_raw_payload=False,
    )

    assert not df.empty
    assert len(df) > 0
    assert "trade_date" in df.columns
    assert "adjusted_close" in df.columns
