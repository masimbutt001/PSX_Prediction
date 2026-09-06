"""Unit tests for Phase 02 collectors: SCS Trade, Yahoo Finance, DPS, and Composite."""

import datetime
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest
from typer.testing import CliRunner

from psx_predictor.cli.main import app
from psx_predictor.collectors.base import (
    CollectorError,
    EmptyResponseError,
    calculate_circuit_locks,
)
from psx_predictor.collectors.composite import CompositeCollector
from psx_predictor.collectors.dps_collector import DPSCollector
from psx_predictor.collectors.scs_collector import SCSTradeCollector
from psx_predictor.collectors.yahoo_collector import YahooCollector


# ---------------------------------------------------------------------------
# 1. Circuit Breaker Calculations
# ---------------------------------------------------------------------------
def test_calculate_circuit_locks():
    """Verify circuit lock limit calculations (+/-7.5% flat price action)."""
    df = pd.DataFrame(
        [
            {"close": 100.0, "high": 102.0, "low": 99.0},
            {"close": 107.5, "high": 107.5, "low": 107.5},  # Upper lock (+7.5%)
            {"close": 108.0, "high": 110.0, "low": 105.0},  # Normal trading
            {"close": 99.9, "high": 99.9, "low": 99.9},  # Lower lock (-7.5%)
        ]
    )

    result = calculate_circuit_locks(df)
    assert bool(result["is_upper_lock"].iloc[1]) is True
    assert bool(result["is_lower_lock"].iloc[1]) is False

    assert bool(result["is_upper_lock"].iloc[2]) is False
    assert bool(result["is_lower_lock"].iloc[2]) is False

    assert bool(result["is_upper_lock"].iloc[3]) is False
    assert bool(result["is_lower_lock"].iloc[3]) is True


# ---------------------------------------------------------------------------
# 2. SCS Trade Collector Tests
# ---------------------------------------------------------------------------
def test_scs_parse_ms_date():
    """Verify parsing of Microsoft JSON date strings."""
    parsed = SCSTradeCollector.parse_ms_date("/Date(1705258800000)/")
    assert isinstance(parsed, datetime.date)
    assert parsed.year == 2024
    assert parsed.month == 1

    # Test ISO format fallback
    fallback = SCSTradeCollector.parse_ms_date("2024-05-15")
    assert fallback == datetime.date(2024, 5, 15)


def test_scs_collector_fetch_and_normalize(tmp_path):
    """Verify SCS Trade collector fetches and normalizes raw JSON data."""
    mock_payload = {
        "d": [
            {
                "trading_Date": "/Date(1705258800000)/",
                "trading_open": 128.6,
                "trading_high": 133.8,
                "trading_low": 128.51,
                "trading_close": 130.29,
                "trading_vol": 18594025,
                "trading_change": 2.76,
            },
            {
                "trading_Date": "/Date(1705345200000)/",
                "trading_open": 130.5,
                "trading_high": 132.0,
                "trading_low": 129.0,
                "trading_close": 131.0,
                "trading_vol": 12000000,
                "trading_change": 0.71,
            },
        ]
    }

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload
    mock_resp.raise_for_status.return_value = None
    mock_client.post.return_value = mock_resp

    collector = SCSTradeCollector(client=mock_client)
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2024, 1, 20)

    df = collector.fetch_historical(
        symbol="OGDC",
        start_date=start_date,
        end_date=end_date,
        save_raw_payload=False,
    )

    assert not df.empty
    assert len(df) == 2
    assert "symbol" in df.columns
    assert df["symbol"].iloc[0] == "OGDC"
    assert "close" in df.columns
    assert "volume" in df.columns
    assert df["close"].iloc[0] == 130.29
    assert df["volume"].iloc[0] == 18594025


def test_scs_collector_empty_response():
    """Verify SCS collector raises EmptyResponseError on empty 'd' list."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"d": []}
    mock_resp.raise_for_status.return_value = None
    mock_client.post.return_value = mock_resp

    collector = SCSTradeCollector(client=mock_client)
    with pytest.raises(EmptyResponseError):
        collector.fetch_historical(
            symbol="UNKNOWN",
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 10),
            save_raw_payload=False,
        )


def test_scs_collector_retries_on_network_error():
    """Verify exponential backoff retries when encountering network failures."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "d": [
            {
                "trading_Date": "/Date(1705258800000)/",
                "trading_open": 100.0,
                "trading_high": 105.0,
                "trading_low": 99.0,
                "trading_close": 102.0,
                "trading_vol": 50000,
            }
        ]
    }
    mock_resp.raise_for_status.return_value = None

    # Fail on first 2 calls, succeed on 3rd
    mock_client.post.side_effect = [
        httpx.ConnectError("Connection refused"),
        httpx.ReadTimeout("Socket timeout"),
        mock_resp,
    ]

    with patch("time.sleep", return_value=None):
        collector = SCSTradeCollector(client=mock_client)
        df = collector.fetch_historical(
            symbol="OGDC",
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 10),
            save_raw_payload=False,
        )
        assert len(df) == 1
        assert mock_client.post.call_count == 3


# ---------------------------------------------------------------------------
# 3. Yahoo Finance Collector Tests
# ---------------------------------------------------------------------------
def test_yahoo_collector_symbol_resolution():
    """Verify PSX ticker resolution to .KA format."""
    collector = YahooCollector()
    assert collector.resolve_yahoo_symbol("OGDC") == "OGDC.KA"
    assert collector.resolve_yahoo_symbol("PPL.KA") == "PPL.KA"


def test_yahoo_collector_fetch_and_normalize():
    """Verify Yahoo collector normalizes yfinance OHLCV DataFrame."""
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    mock_yf_df = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 101.0],
            "High": [105.0, 104.0, 103.0],
            "Low": [98.0, 100.0, 99.0],
            "Close": [102.0, 103.0, 100.0],
            "Adj Close": [101.5, 102.5, 99.5],
            "Volume": [100000, 150000, 80000],
            "Dividends": [0.0, 2.5, 0.0],
            "Stock Splits": [0.0, 0.0, 0.0],
        },
        index=dates,
    )

    collector = YahooCollector()
    with patch.object(collector, "fetch_raw", return_value=mock_yf_df):
        df = collector.fetch_historical(
            symbol="OGDC",
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 5),
            save_raw_payload=False,
        )

        assert len(df) == 3
        assert df["symbol"].iloc[0] == "OGDC"
        assert df["adjusted_close"].iloc[0] == 101.5
        assert df["dividend_amount"].iloc[1] == 2.5
        assert "is_upper_lock" in df.columns


# ---------------------------------------------------------------------------
# 4. PSX Data Portal (DPS) Collector Tests
# ---------------------------------------------------------------------------
def test_dps_collector_timeseries_json():
    """Verify DPS collector correctly parses JSON timeseries payload."""
    mock_payload = {
        "type": "timeseries_json",
        "payload": [
            [1704067200, 100.0, 105.0, 98.0, 102.0, 500000],
            [1704153600, 102.0, 104.0, 101.0, 103.5, 600000],
        ],
    }

    collector = DPSCollector()
    with patch.object(collector, "fetch_raw", return_value=mock_payload):
        df = collector.fetch_historical(
            symbol="OGDC",
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 5),
            save_raw_payload=False,
        )

        assert len(df) == 2
        assert df["symbol"].iloc[0] == "OGDC"
        assert df["close"].iloc[0] == 102.0
        assert df["volume"].iloc[0] == 500000


def test_dps_collector_html_fallback():
    """Verify DPS collector parses HTML table fallback."""
    sample_html = """
    <table>
        <tr><th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Vol</th></tr>
        <tr><td>2024-01-02</td><td>100.00</td><td>105.00</td><td>99.00</td><td>103.00</td><td>120000</td></tr>
    </table>
    """
    mock_payload = {"type": "html_table", "payload": sample_html}

    collector = DPSCollector()
    with patch.object(collector, "fetch_raw", return_value=mock_payload):
        df = collector.fetch_historical(
            symbol="OGDC",
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 5),
            save_raw_payload=False,
        )

        assert len(df) == 1
        assert df["close"].iloc[0] == 103.0


# ---------------------------------------------------------------------------
# 5. Composite Collector Tests
# ---------------------------------------------------------------------------
def test_composite_collector_fallback_on_error():
    """Verify composite collector falls back to secondary when primary fails."""
    # Sub-collector 1: fails
    mock_coll_1 = MagicMock(spec=SCSTradeCollector)
    mock_coll_1.source_name = "scs"
    mock_coll_1.fetch_historical.side_effect = CollectorError("SCS Service unavailable")

    # Sub-collector 2: succeeds
    mock_coll_2 = MagicMock(spec=YahooCollector)
    mock_coll_2.source_name = "yahoo"
    mock_coll_2.fetch_historical.return_value = pd.DataFrame(
        [
            {
                "symbol": "OGDC",
                "trade_date": datetime.date(2024, 1, 2),
                "open": 100.0,
                "high": 105.0,
                "low": 98.0,
                "close": 102.0,
                "adjusted_close": 102.0,
                "volume": 200000,
                "dividend_amount": 0.0,
                "split_ratio": 1.0,
                "is_upper_lock": False,
                "is_lower_lock": False,
            }
        ]
    )

    composite = CompositeCollector(collectors=[mock_coll_1, mock_coll_2])
    df = composite.fetch_historical(
        symbol="OGDC",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 1, 5),
    )

    assert len(df) == 1
    assert df["close"].iloc[0] == 102.0
    assert mock_coll_1.fetch_historical.called
    assert mock_coll_2.fetch_historical.called


def test_composite_collector_all_fail_raises():
    """Verify CollectorError raised if all underlying providers fail."""
    mock_coll = MagicMock(spec=SCSTradeCollector)
    mock_coll.source_name = "scs"
    mock_coll.fetch_historical.side_effect = CollectorError("Connection timeout")

    composite = CompositeCollector(collectors=[mock_coll])
    with pytest.raises(CollectorError, match="All providers failed"):
        composite.fetch_historical(
            symbol="OGDC",
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2024, 1, 5),
        )


# ---------------------------------------------------------------------------
# 6. CLI Data Fetch Command Tests
# ---------------------------------------------------------------------------
def test_cli_data_fetch_dry_run(cli_runner: CliRunner):
    """Test 'psx data fetch' command executes properly in dry-run mode."""
    mock_df = pd.DataFrame(
        [
            {
                "symbol": "OGDC",
                "trade_date": datetime.date(2024, 1, 15),
                "open": 128.0,
                "high": 132.0,
                "low": 127.5,
                "close": 130.0,
                "adjusted_close": 130.0,
                "volume": 1500000,
                "dividend_amount": 0.0,
                "split_ratio": 1.0,
                "is_upper_lock": False,
                "is_lower_lock": False,
            }
        ]
    )

    with patch(
        "psx_predictor.collectors.composite.CompositeCollector.fetch_historical",
        return_value=mock_df,
    ):
        result = cli_runner.invoke(
            app,
            ["data", "fetch", "--symbol", "OGDC", "--source", "composite", "--dry-run"],
        )
        assert result.exit_code == 0
        assert "OGDC" in result.output
        assert "PKR 130.00" in result.output
        assert "Dry run active" in result.output
