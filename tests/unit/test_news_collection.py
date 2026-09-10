"""Unit tests for Phase 11: News and Corporate Announcements Collection."""

from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from psx_predictor.cli.main import app
from psx_predictor.news.dps_announcements import DPSAnnouncementsCollector
from psx_predictor.news.rss_collector import RSSNewsCollector
from psx_predictor.news.schemas import (
    RawAnnouncement,
    RawNewsArticle,
    compute_announcement_id,
    compute_news_id,
)
from psx_predictor.storage.parquet_io import read_parquet

SAMPLE_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Business Recorder - Markets</title>
    <link>https://www.brecorder.com/feeds/markets</link>
    <description>Financial Markets News</description>
    <item>
      <title>KSE-100 index gains 500 points amid buying surge</title>
      <link>https://www.brecorder.com/news/10001</link>
      <description>&lt;p&gt;PSX rally &lt;b&gt;Thursday&lt;/b&gt;.&lt;/p&gt;</description>
      <pubDate>Thu, 10 Sep 2026 14:30:00 +0500</pubDate>
      <guid>https://www.brecorder.com/news/10001</guid>
    </item>
    <item>
      <title>SBP maintains policy rate at 15 percent</title>
      <link>https://www.brecorder.com/news/10002</link>
      <description>The Monetary Policy Committee announced the rate decision.</description>
      <pubDate>Thu, 10 Sep 2026 16:00:00 +0500</pubDate>
      <guid>https://www.brecorder.com/news/10002</guid>
    </item>
  </channel>
</rss>
"""

SAMPLE_ATOM_XML = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Dawn Business Feed</title>
  <entry>
    <title>IMF mission to visit Islamabad for review</title>
    <link href="https://www.dawn.com/news/20001" />
    <summary>Talks will focus on revenue and structural benchmarks.</summary>
    <published>2026-09-10T11:00:00Z</published>
    <id>urn:dawn:news:20001</id>
  </entry>
</feed>
"""

SAMPLE_DPS_HTML = """
<html>
<body>
<table class="table">
  <thead>
    <tr>
      <th>DATE</th><th>TIME</th><th>SYMBOL</th><th>NAME</th><th>TITLE</th><th>ACTION</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Sep 10, 2026</td>
      <td>3:24 PM</td>
      <td>OGDC</td>
      <td>Oil &amp; Gas Development Company Limited</td>
      <td>Financial Results for the Year Ended June 30, 2026</td>
      <td><a href="/download/document/282509.pdf">ViewPDF</a></td>
    </tr>
    <tr>
      <td>Sep 10, 2026</td>
      <td>2:15 PM</td>
      <td>PPL</td>
      <td>Pakistan Petroleum Limited</td>
      <td>Notice of Annual General Meeting</td>
      <td><a href="/download/document/282510.pdf">ViewPDF</a></td>
    </tr>
  </tbody>
</table>
</body>
</html>
"""


@pytest.fixture
def temp_storage(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "raw_news": tmp_path / "raw" / "news",
        "raw_announcements": tmp_path / "raw" / "announcements",
        "processed_prices": tmp_path / "processed" / "prices",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


class TestNewsSchemas:
    def test_news_article_schema_and_hashing(self) -> None:
        expected_id = compute_news_id("Dawn", "OGDC Discovers Gas", "2026-09-10")
        assert len(expected_id) == 64

        art = RawNewsArticle(
            source="Dawn",
            headline="OGDC Discovers Gas",
            summary="New well test yields 15 MMSCFD",
            url="https://dawn.com/news/123",
            published_at="2026-09-10",
        )
        assert art.article_id == expected_id
        assert art.summary == "New well test yields 15 MMSCFD"
        assert art.to_dict()["article_id"] == expected_id

    def test_announcement_schema_and_hashing(self) -> None:
        expected_id = compute_announcement_id("OGDC", "Board Meeting", "2026-09-10", "3:00 PM")
        assert len(expected_id) == 64

        ann = RawAnnouncement(
            symbol="OGDC",
            company_name="Oil & Gas Dev Co",
            title="Board Meeting",
            announcement_date="2026-09-10",
            announcement_time="3:00 PM",
            document_url="https://dps.psx.com.pk/doc/1.pdf",
        )
        assert ann.announcement_id == expected_id
        assert ann.symbol == "OGDC"
        assert ann.document_url == "https://dps.psx.com.pk/doc/1.pdf"


class TestRSSCollector:
    def test_parse_rss_20_feed(self) -> None:
        collector = RSSNewsCollector()
        articles = collector.parse_feed_xml(SAMPLE_RSS_XML, source_name="Test RSS")
        assert len(articles) == 2

        a1 = articles[0]
        assert a1.headline == "KSE-100 index gains 500 points amid buying surge"
        assert a1.url == "https://www.brecorder.com/news/10001"
        # Verify HTML stripped from description
        assert "<p>" not in a1.summary
        assert "Thursday" in a1.summary
        assert a1.source == "Test RSS"
        assert a1.article_id is not None

    def test_parse_atom_feed(self) -> None:
        collector = RSSNewsCollector()
        articles = collector.parse_feed_xml(SAMPLE_ATOM_XML, source_name="Test Atom")
        assert len(articles) == 1

        a = articles[0]
        assert a.headline == "IMF mission to visit Islamabad for review"
        assert a.url == "https://www.dawn.com/news/20001"
        assert "revenue" in a.summary
        assert a.source == "Test Atom"

    @respx.mock
    def test_news_deduplication_persistence(self, temp_storage: dict[str, Path]) -> None:
        feed_url = "https://example.com/rss.xml"
        respx.get(feed_url).mock(return_value=httpx.Response(200, text=SAMPLE_RSS_XML))

        collector = RSSNewsCollector(
            feeds={"Test Feed": feed_url},
            storage_paths=temp_storage,
        )

        # First ingestion
        res1 = collector.fetch_all(dry_run=False)
        assert res1.total_fetched == 2
        assert res1.new_articles_added == 2
        assert res1.duplicates_skipped == 0

        news_file = temp_storage["raw_news"] / "news.parquet"
        assert news_file.exists()
        df1 = read_parquet(news_file)
        assert len(df1) == 2

        # Second ingestion: same feed content
        res2 = collector.fetch_all(dry_run=False)
        assert res2.total_fetched == 2
        assert res2.new_articles_added == 0
        assert res2.duplicates_skipped == 2

        df2 = read_parquet(news_file)
        assert len(df2) == 2  # No duplicate rows written


class TestDPSAnnouncementsCollector:
    def test_parse_html_table(self) -> None:
        collector = DPSAnnouncementsCollector()
        announcements = collector.parse_html_table(SAMPLE_DPS_HTML)
        assert len(announcements) == 2

        ann1 = announcements[0]
        assert ann1.symbol == "OGDC"
        assert ann1.company_name == "Oil & Gas Development Company Limited"
        assert "Financial Results" in ann1.title
        assert ann1.announcement_date == "2026-09-10"
        assert ann1.announcement_time == "3:24 PM"
        assert ann1.document_url == "https://dps.psx.com.pk/download/document/282509.pdf"

        ann2 = announcements[1]
        assert ann2.symbol == "PPL"
        assert ann2.document_url == "https://dps.psx.com.pk/download/document/282510.pdf"

    @respx.mock
    def test_announcements_deduplication_persistence(self, temp_storage: dict[str, Path]) -> None:
        respx.post("https://dps.psx.com.pk/announcements").mock(
            return_value=httpx.Response(200, text=SAMPLE_DPS_HTML)
        )

        collector = DPSAnnouncementsCollector(storage_paths=temp_storage)

        # First ingestion
        res1 = collector.fetch_announcements(dry_run=False)
        assert res1.total_fetched == 2
        assert res1.new_announcements_added == 2
        assert res1.duplicates_skipped == 0

        ann_file = temp_storage["raw_announcements"] / "announcements.parquet"
        assert ann_file.exists()
        df1 = read_parquet(ann_file)
        assert len(df1) == 2

        # Second ingestion: identical announcements
        res2 = collector.fetch_announcements(dry_run=False)
        assert res2.total_fetched == 2
        assert res2.new_announcements_added == 0
        assert res2.duplicates_skipped == 2

        df2 = read_parquet(ann_file)
        assert len(df2) == 2  # Idempotent: exactly 2 records


class TestNewsCLI:
    def test_news_cli_help_and_status(self) -> None:
        runner = CliRunner()
        res_help = runner.invoke(app, ["news", "--help"])
        assert res_help.exit_code == 0
        assert "fetch" in res_help.stdout
        assert "status" in res_help.stdout

        res_status = runner.invoke(app, ["news", "status"])
        assert res_status.exit_code == 0
        assert "News & Corporate Announcements Storage Status" in res_status.stdout
