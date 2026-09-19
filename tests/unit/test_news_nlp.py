"""Unit tests for Phase 12: News Classification, Entity Matching, and Sentiment NLP."""

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from psx_predictor.cli.main import app
from psx_predictor.news.classifier import EventCategory, EventClassifier
from psx_predictor.news.entity_matcher import EntityMatcher
from psx_predictor.news.processor import NewsNLPProcessor
from psx_predictor.news.sentiment import FinancialSentimentAnalyzer
from psx_predictor.storage.parquet_io import read_parquet, write_parquet_atomic


@pytest.fixture
def temp_storage(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "raw_news": tmp_path / "raw" / "news",
        "raw_announcements": tmp_path / "raw" / "announcements",
        "processed_news": tmp_path / "processed" / "news",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


class TestEntityMatcher:
    def test_official_names_and_aliases(self) -> None:
        matcher = EntityMatcher()

        # OGDC variations
        assert matcher.match(
            "Oil & Gas Development Company Limited announced quarterly accounts"
        ) == ["OGDC"]
        assert matcher.match("OGDCL reports new production milestone") == ["OGDC"]
        assert matcher.match("Oil and Gas Development discovered gas") == ["OGDC"]

        # PPL & Hubco & Engro
        assert matcher.match("Pakistan Petroleum partners with Hubco on green energy") == [
            "HUBC",
            "PPL",
        ]
        assert matcher.match("Engro Corp and Fauji Fertilizer in discussion") == ["ENGRO", "FFC"]

    def test_avoid_common_word_false_positives(self) -> None:
        matcher = EntityMatcher()

        # "luck" as lowercase noun should NOT match LUCK ticker
        assert matcher.match("Wish you good luck on your trading journey today") == []

        # "system" as lowercase word should NOT match SYS ticker
        assert matcher.match("The operating system update was applied successfully") == []

        # But uppercase tickers or explicit company names DO match
        assert matcher.match("LUCK expands clinker production capacity") == ["LUCK"]
        assert matcher.match("Lucky Cement posts record export numbers") == ["LUCK"]
        assert matcher.match("SYS signs $20M export contract in Gulf") == ["SYS"]
        assert matcher.match("Systems Limited board meeting on Monday") == ["SYS"]


class TestEventClassifier:
    def test_all_event_categories(self) -> None:
        classifier = EventClassifier()

        cat, conf = classifier.classify(
            "OGDC reports quarterly profit after tax of PKR 45 billion and higher EPS"
        )
        assert cat == EventCategory.EARNINGS.value
        assert conf >= 0.6

        cat, conf = classifier.classify("FFC declares interim cash dividend of PKR 5.50 per share")
        assert cat == EventCategory.DIVIDEND.value

        cat, conf = classifier.classify("Company announces 15% bonus shares for shareholders")
        assert cat == EventCategory.BONUS_ISSUE.value

        cat, conf = classifier.classify(
            "Hydrocarbon discovery of gas and condensate with flow rate of 20 MMSCFD"
        )
        assert cat == EventCategory.DISCOVERY.value

        cat, conf = classifier.classify(
            "Temporary plant shutdown for annual maintenance turnaround"
        )
        assert cat == EventCategory.PRODUCTION_CHANGE.value

        cat, conf = classifier.classify(
            "SECP issues directive and show-cause notice regarding insider trading"
        )
        assert cat == EventCategory.REGULATORY.value

        cat, conf = classifier.classify(
            "SBP Monetary Policy Committee keeps policy rate unchanged as CPI inflation eases"
        )
        assert cat == EventCategory.MACRO_INTEREST_RATE.value

        cat, conf = classifier.classify("Local trade delegation visits international exhibition")
        assert cat == EventCategory.OTHER.value


class TestFinancialSentimentAnalyzer:
    def test_positive_sentiment(self) -> None:
        analyzer = FinancialSentimentAnalyzer()
        res = analyzer.analyze("Record profit surge and dividend increase boost investor optimism")
        assert res.score > 0.10
        assert res.label == "POSITIVE"
        assert "profit" in res.positive_tokens
        assert "surge" in res.positive_tokens

    def test_negative_sentiment(self) -> None:
        analyzer = FinancialSentimentAnalyzer()
        res = analyzer.analyze(
            "Massive slump and sharp loss as plant shutdown causes severe revenue plunge"
        )
        assert res.score < -0.10
        assert res.label == "NEGATIVE"
        assert "loss" in res.negative_tokens
        assert "plunge" in res.negative_tokens

    def test_negation_awareness(self) -> None:
        analyzer = FinancialSentimentAnalyzer()

        # "not a loss" should NOT be classified as negative
        res_not_loss = analyzer.analyze("Company confirmed it did not suffer a loss")
        assert res_not_loss.score >= 0.0

        # "no dividend" should NOT be classified as positive
        res_no_div = analyzer.analyze("Board decided on no dividend for the second quarter")
        assert res_no_div.score <= 0.0

    def test_neutral_sentiment(self) -> None:
        analyzer = FinancialSentimentAnalyzer()
        res = analyzer.analyze("The annual general meeting of shareholders will be held in Karachi")
        assert res.label == "NEUTRAL"
        assert abs(res.score) < 0.05


class TestNewsNLPProcessor:
    def test_end_to_end_processing_and_deduplication(self, temp_storage: dict[str, Path]) -> None:
        # 1. Create synthetic raw news
        news_df = pd.DataFrame(
            [
                {
                    "article_id": "art_001",
                    "source": "Business Recorder",
                    "headline": "OGDC discovers major hydrocarbon reserves in Sindh",
                    "summary": "Gas discovery test flow rate reached 18 MMSCFD.",
                    "url": "https://brecorder.com/1",
                    "published_at": "2026-09-10T10:00:00",
                },
                {
                    "article_id": "art_002",
                    "source": "Dawn Business",
                    "headline": "SBP keeps policy rate steady amid inflation decline",
                    "summary": "Monetary policy committee maintains rate at 15 percent.",
                    "url": "https://dawn.com/2",
                    "published_at": "2026-09-10T11:00:00",
                },
            ]
        )
        write_parquet_atomic(news_df, temp_storage["raw_news"] / "news.parquet")

        # 2. Create synthetic raw announcements
        ann_df = pd.DataFrame(
            [
                {
                    "announcement_id": "ann_001",
                    "symbol": "PPL",
                    "company_name": "Pakistan Petroleum Limited",
                    "title": "Financial Results for year ended June 30, 2026 with cash dividend",
                    "announcement_date": "2026-09-10",
                    "announcement_time": "3:00 PM",
                    "document_url": "https://dps.psx.com.pk/1.pdf",
                    "source": "PSX DPS",
                }
            ]
        )
        write_parquet_atomic(ann_df, temp_storage["raw_announcements"] / "announcements.parquet")

        # 3. Process
        processor = NewsNLPProcessor(storage_paths=temp_storage)
        res1 = processor.process_all(dry_run=False)

        assert res1.total_news_inspected == 2
        assert res1.total_announcements_inspected == 1
        assert res1.new_signals_created == 3
        assert res1.duplicates_skipped == 0

        signals_file = temp_storage["processed_news"] / "news_signals.parquet"
        assert signals_file.exists()
        sig_df1 = read_parquet(signals_file)
        assert len(sig_df1) == 3

        # Check art_001 entity and category
        art1_sig = [s for s in res1.signals if s.raw_id == "art_001"][0]
        assert "OGDC" in art1_sig.matched_symbols
        assert art1_sig.event_category == EventCategory.DISCOVERY.value
        assert art1_sig.sentiment_label == "POSITIVE"

        # Check ann_001 category
        ann1_sig = [s for s in res1.signals if s.raw_id == "ann_001"][0]
        assert ann1_sig.primary_symbol == "PPL"
        assert ann1_sig.event_category in (
            EventCategory.DIVIDEND.value,
            EventCategory.EARNINGS.value,
        )

        # 4. Re-run: verify 100% deduplication
        res2 = processor.process_all(dry_run=False)
        assert res2.new_signals_created == 0
        assert res2.duplicates_skipped == 3
        sig_df2 = read_parquet(signals_file)
        assert len(sig_df2) == 3  # Zero duplicate rows written


class TestCLIIntegration:
    def test_cli_news_process_and_status(self) -> None:
        runner = CliRunner()
        res_dry = runner.invoke(app, ["news", "process", "--dry-run"])
        assert res_dry.exit_code == 0
        assert "News NLP Signal Extraction Summary" in res_dry.stdout

        res_status = runner.invoke(app, ["news", "status"])
        assert res_status.exit_code == 0
        assert "Storage Status" in res_status.stdout
        assert "Processed" in res_status.stdout
