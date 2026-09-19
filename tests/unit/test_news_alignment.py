"""Unit tests for Phase 13: PSX Market Session Calendar and News Alignment."""

import datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from psx_predictor.cli.main import app
from psx_predictor.news.calendar import (
    PSXMarketCalendar,
)
from psx_predictor.news.session_aligner import (
    NewsSessionAligner,
    SessionAlignment,
)
from psx_predictor.storage.parquet_io import write_parquet_atomic


@pytest.fixture
def test_calendar() -> PSXMarketCalendar:
    """Fixture providing an isolated PSXMarketCalendar instance."""
    return PSXMarketCalendar()


@pytest.fixture
def temp_aligner(tmp_path: Path, test_calendar: PSXMarketCalendar) -> NewsSessionAligner:
    """Fixture providing NewsSessionAligner configured with temporary storage."""
    raw_news = tmp_path / "raw" / "news"
    raw_news.mkdir(parents=True, exist_ok=True)
    raw_ann = tmp_path / "raw" / "announcements"
    raw_ann.mkdir(parents=True, exist_ok=True)
    proc_news = tmp_path / "processed" / "news"
    proc_news.mkdir(parents=True, exist_ok=True)
    feat_news = tmp_path / "features" / "news"
    feat_news.mkdir(parents=True, exist_ok=True)
    proc_prices = tmp_path / "processed" / "prices"
    proc_prices.mkdir(parents=True, exist_ok=True)

    storage_paths = {
        "root": tmp_path,
        "raw_news": raw_news,
        "raw_announcements": raw_ann,
        "processed_news": proc_news,
        "features_news": feat_news,
        "processed_prices": proc_prices,
    }

    return NewsSessionAligner(calendar=test_calendar, storage_paths=storage_paths)


# --- Calendar Unit Tests ---


def test_calendar_trading_days(test_calendar: PSXMarketCalendar) -> None:
    """Verify regular weekdays are identified as trading days and weekends are excluded."""
    # 2026-09-07 is Monday, 2026-09-11 is Friday, 2026-09-12 is Saturday, 2026-09-13 is Sunday
    monday = datetime.date(2026, 9, 7)
    friday = datetime.date(2026, 9, 11)
    saturday = datetime.date(2026, 9, 12)
    sunday = datetime.date(2026, 9, 13)

    assert test_calendar.is_trading_day(monday) is True
    assert test_calendar.is_trading_day(friday) is True
    assert test_calendar.is_trading_day(saturday) is False
    assert test_calendar.is_trading_day(sunday) is False


def test_calendar_fixed_holidays(test_calendar: PSXMarketCalendar) -> None:
    """Verify fixed annual national holidays are marked as closed."""
    kashmir_day = datetime.date(2026, 2, 5)  # Feb 5
    pakistan_day = datetime.date(2026, 3, 23)  # Mar 23
    labour_day = datetime.date(2026, 5, 1)  # May 1
    independence_day = datetime.date(2026, 8, 14)  # Aug 14
    quaideazam_day = datetime.date(2026, 12, 25)  # Dec 25

    for holiday in [kashmir_day, pakistan_day, labour_day, independence_day, quaideazam_day]:
        assert test_calendar.is_holiday(holiday) is True
        assert test_calendar.is_trading_day(holiday) is False


def test_calendar_next_previous_trading_day(test_calendar: PSXMarketCalendar) -> None:
    """Verify next_trading_day and previous_trading_day jump over weekends and holidays."""
    friday = datetime.date(2026, 9, 11)
    monday = datetime.date(2026, 9, 14)
    assert test_calendar.next_trading_day(friday) == monday
    assert test_calendar.previous_trading_day(monday) == friday

    # Holiday jump: 2026-03-23 is Monday (Pakistan Day)
    tuesday_after = datetime.date(2026, 3, 24)
    # 2026-03-20 to 2026-03-22 is Eid-ul-Fitr, 2026-03-23 is Pakistan Day
    # next trading day from Thu 2026-03-19 jumps to Tue 2026-03-24
    thursday_before = datetime.date(2026, 3, 19)
    assert test_calendar.next_trading_day(thursday_before) == tuesday_after


# --- Session Alignment Unit Tests ---


def test_after_hours_news_aligned_to_next_day(temp_aligner: NewsSessionAligner) -> None:
    """Article at 17:00 on Tuesday must have session_target_date as Wednesday."""
    # 2026-09-08 is Tuesday
    tuesday_1700 = "2026-09-08T17:00:00+05:00"
    alignment: SessionAlignment = temp_aligner.align_timestamp(tuesday_1700)

    assert alignment.session_date == datetime.date(2026, 9, 9)  # Wednesday
    assert alignment.session_type == "REGULAR"
    assert alignment.is_intraday is False
    assert "2026-09-09T09:15:00" in alignment.usable_from


def test_friday_prayer_break_aligned_to_session_2(temp_aligner: NewsSessionAligner) -> None:
    """Article at 13:00 on Friday must align to Friday Session 2 (14:30 PKT)."""
    # 2026-09-11 is Friday
    friday_1300 = "2026-09-11T13:00:00+05:00"
    alignment: SessionAlignment = temp_aligner.align_timestamp(friday_1300)

    assert alignment.session_date == datetime.date(2026, 9, 11)  # Friday
    assert alignment.session_type == "FRIDAY_SESSION_2"
    assert alignment.is_intraday is False  # During prayer break exchange is paused
    assert "2026-09-11T14:30:00" in alignment.usable_from


def test_friday_session_1_intraday_aligned_to_session_2(temp_aligner: NewsSessionAligner) -> None:
    """Article at 10:30 on Friday during Session 1 aligns to Friday Session 2."""
    friday_1030 = "2026-09-11T10:30:00+05:00"
    alignment = temp_aligner.align_timestamp(friday_1030)

    assert alignment.session_date == datetime.date(2026, 9, 11)
    assert alignment.session_type == "FRIDAY_SESSION_2"
    assert alignment.is_intraday is True
    assert "2026-09-11T14:30:00" in alignment.usable_from


def test_friday_after_hours_aligned_to_monday(temp_aligner: NewsSessionAligner) -> None:
    """Article at 18:00 on Friday must align to Monday."""
    friday_1800 = "2026-09-11T18:00:00+05:00"
    alignment = temp_aligner.align_timestamp(friday_1800)

    assert alignment.session_date == datetime.date(2026, 9, 14)  # Monday
    assert alignment.session_type == "REGULAR"
    assert alignment.is_intraday is False
    assert "2026-09-14T09:15:00" in alignment.usable_from


def test_weekend_news_aligned_to_monday(temp_aligner: NewsSessionAligner) -> None:
    """Article at 14:00 on Sunday must align to Monday."""
    # 2026-09-13 is Sunday
    sunday_1400 = "2026-09-13T14:00:00+05:00"
    alignment = temp_aligner.align_timestamp(sunday_1400)

    assert alignment.session_date == datetime.date(2026, 9, 14)  # Monday
    assert alignment.session_type == "REGULAR"
    assert alignment.is_intraday is False
    assert "2026-09-14T09:15:00" in alignment.usable_from


def test_pre_market_weekday_aligned_to_today(temp_aligner: NewsSessionAligner) -> None:
    """Article published at 08:15 on Thursday maps to Thursday Regular session."""
    # 2026-09-10 is Thursday
    thursday_0815 = "2026-09-10T08:15:00+05:00"
    alignment = temp_aligner.align_timestamp(thursday_0815)

    assert alignment.session_date == datetime.date(2026, 9, 10)
    assert alignment.session_type == "REGULAR"
    assert alignment.is_intraday is False
    assert "2026-09-10T09:15:00" in alignment.usable_from


def test_holiday_news_aligned_to_next_trading_day(temp_aligner: NewsSessionAligner) -> None:
    """News published on a national holiday aligns to the next active trading day."""
    # Independence Day Aug 14, 2026 is Friday -> Next trading day is Monday Aug 17, 2026
    holiday_article = "2026-08-14T11:00:00+05:00"
    alignment = temp_aligner.align_timestamp(holiday_article)

    assert alignment.session_date == datetime.date(2026, 8, 17)  # Monday
    assert alignment.session_type == "REGULAR"
    assert alignment.is_intraday is False


# --- Zero Lookahead & Point-in-Time Feature Aggregation Tests ---


def test_point_in_time_zero_lookahead_leakage(temp_aligner: NewsSessionAligner) -> None:
    """Signals published after session open (09:15 PKT) must NEVER leak into 24h features."""
    signals_data = [
        # Signal 1: Published at 08:30 on Wed 2026-09-09 (before open) -> SHOULD be included
        {
            "signal_id": "sig_pre",
            "primary_symbol": "OGDC",
            "matched_symbols": ["OGDC"],
            "published_at": "2026-09-09T08:30:00+05:00",
            "sentiment_score": 0.8,
            "sentiment_label": "POSITIVE",
            "event_category": "DISCOVERY",
        },
        # Signal 2: Published at 10:30 on Wed 2026-09-09 (intraday) -> MUST NOT leak into Wed!
        {
            "signal_id": "sig_intra",
            "primary_symbol": "OGDC",
            "matched_symbols": ["OGDC"],
            "published_at": "2026-09-09T10:30:00+05:00",
            "sentiment_score": -0.9,
            "sentiment_label": "NEGATIVE",
            "event_category": "PRODUCTION_CHANGE",
        },
    ]
    signals_df = pd.DataFrame(signals_data)
    trading_dates = [datetime.date(2026, 9, 9), datetime.date(2026, 9, 10)]

    features = temp_aligner.compute_daily_session_features(
        signals_df=signals_df,
        symbol="OGDC",
        trading_dates=trading_dates,
    )

    row_wed = features[features["session_date"] == "2026-09-09"].iloc[0]
    row_thu = features[features["session_date"] == "2026-09-10"].iloc[0]

    # Wednesday session: ONLY sig_pre (08:30) is eligible. sig_intra (10:30) MUST NOT be present!
    assert row_wed["news_count_24h"] == 1
    assert row_wed["sentiment_24h"] == 0.8
    assert row_wed["positive_count_24h"] == 1
    assert row_wed["negative_count_24h"] == 0
    assert row_wed["has_discovery_announcement_today"] == 1

    # Thursday session: sig_intra (Wed 10:30) is now within the past 24h window
    # So on Thursday morning, sig_intra is captured!
    assert row_thu["news_count_24h"] == 1
    assert row_thu["sentiment_24h"] == -0.9
    assert row_thu["negative_count_24h"] == 1


def test_feature_aggregation_calculation(temp_aligner: NewsSessionAligner) -> None:
    """Verify exact numerical values of sentiment_24h, sentiment_72h, and event indicators."""
    # Thursday 2026-09-10 Session open is 2026-09-10 09:15:00 PKT
    # 24h window: 2026-09-09 09:15:00 to 2026-09-10 09:15:00
    # 72h window: 2026-09-07 09:15:00 to 2026-09-10 09:15:00
    signals_data = [
        # Signal A: in 24h window
        {
            "signal_id": "sig_a",
            "primary_symbol": "PPL",
            "matched_symbols": ["PPL"],
            "published_at": "2026-09-09T18:00:00+05:00",
            "sentiment_score": 0.60,
            "sentiment_label": "POSITIVE",
            "event_category": "EARNINGS",
        },
        # Signal B: in 24h window
        {
            "signal_id": "sig_b",
            "primary_symbol": "PPL",
            "matched_symbols": ["PPL"],
            "published_at": "2026-09-10T07:00:00+05:00",
            "sentiment_score": 0.40,
            "sentiment_label": "POSITIVE",
            "event_category": "DIVIDEND",
        },
        # Signal C: older than 24h, but within 72h (Tuesday morning)
        {
            "signal_id": "sig_c",
            "primary_symbol": "PPL",
            "matched_symbols": ["PPL"],
            "published_at": "2026-09-08T10:00:00+05:00",
            "sentiment_score": -0.80,
            "sentiment_label": "NEGATIVE",
            "event_category": "PRODUCTION_CHANGE",
        },
    ]
    signals_df = pd.DataFrame(signals_data)
    target_dates = [datetime.date(2026, 9, 10)]

    features = temp_aligner.compute_daily_session_features(
        signals_df=signals_df,
        symbol="PPL",
        trading_dates=target_dates,
    )
    row = features.iloc[0]

    # 24h: Signals A & B -> count = 2, mean sentiment = (0.60 + 0.40) / 2 = 0.50
    assert row["news_count_24h"] == 2
    assert row["positive_count_24h"] == 2
    assert row["negative_count_24h"] == 0
    assert row["sentiment_24h"] == 0.50
    assert row["has_earnings_announcement_today"] == 1
    assert row["has_dividend_announcement_today"] == 1

    # 72h: Signals A, B, C -> count = 3, mean sentiment = (0.60 + 0.40 - 0.80) / 3 = 0.0667
    assert row["news_count_72h"] == 3
    assert abs(row["sentiment_72h"] - round(0.2 / 3, 4)) < 1e-4


# --- CLI Command Test ---


def test_cli_news_align_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that psx news align executes and reports metrics via CliRunner."""
    from psx_predictor.storage.paths import ensure_directories

    monkeypatch.setenv("PSX_DATA_DIR", str(tmp_path))
    paths = ensure_directories(tmp_path)

    # Seed mock processed signals
    sample_signals = pd.DataFrame(
        [
            {
                "signal_id": "test_sig_1",
                "raw_id": "art_1",
                "source": "Dawn Business",
                "published_at": "2026-09-08T17:00:00+05:00",
                "headline": "OGDC announces bumper discovery in Sukkur block",
                "matched_symbols": ["OGDC"],
                "primary_symbol": "OGDC",
                "event_category": "DISCOVERY",
                "sentiment_score": 0.75,
                "sentiment_label": "POSITIVE",
                "confidence": 0.90,
                "positive_tokens": ["discovery", "bumper"],
                "negative_tokens": [],
                "processed_at": "2026-09-08T17:05:00+05:00",
            },
            {
                "signal_id": "test_sig_2",
                "raw_id": "art_2",
                "source": "PSX DPS",
                "published_at": "2026-09-11T13:00:00+05:00",
                "headline": "PPL declares quarterly interim cash dividend",
                "matched_symbols": ["PPL"],
                "primary_symbol": "PPL",
                "event_category": "DIVIDEND",
                "sentiment_score": 0.65,
                "sentiment_label": "POSITIVE",
                "confidence": 0.95,
                "positive_tokens": ["dividend"],
                "negative_tokens": [],
                "processed_at": "2026-09-11T13:02:00+05:00",
            },
        ]
    )
    sig_file = paths["processed_news"] / "news_signals.parquet"
    write_parquet_atomic(sample_signals, sig_file)

    runner = CliRunner()
    result = runner.invoke(app, ["news", "align", "--symbols", "OGDC,PPL"])

    assert result.exit_code == 0
    assert "PSX Market Session Alignment Summary" in result.stdout
    assert "Total Signals Aligned" in result.stdout
    assert "Session Window Distribution" in result.stdout

    # Verify aligned signals file and features were persisted
    aligned_file = paths["processed_news"] / "aligned_signals.parquet"
    assert aligned_file.exists()
    aligned_df = pd.read_parquet(aligned_file)
    assert len(aligned_df) == 2
    assert "session_target_date" in aligned_df.columns
    assert "usable_from" in aligned_df.columns

    daily_feat_file = paths["features_news"] / "daily_news_features.parquet"
    assert daily_feat_file.exists()
    feat_df = pd.read_parquet(daily_feat_file)
    assert not feat_df.empty
    assert "sentiment_24h" in feat_df.columns
    assert "news_count_24h" in feat_df.columns
