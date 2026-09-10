"""Financial news RSS feed collector for Pakistani business news outlets."""

import datetime
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger

from psx_predictor.config.loader import load_config
from psx_predictor.news.schemas import RawNewsArticle
from psx_predictor.storage.parquet_io import read_parquet, write_parquet_atomic
from psx_predictor.storage.paths import ensure_directories

DEFAULT_RSS_FEEDS: dict[str, str] = {
    "Business Recorder - Latest": "https://www.brecorder.com/feeds/latest-news",
    "Business Recorder - Markets": "https://www.brecorder.com/feeds/markets",
    "Dawn - Business": "https://www.dawn.com/feeds/business",
}


@dataclass
class RSSCollectionResult:
    """Summary of an RSS news collection execution."""

    total_fetched: int = 0
    new_articles_added: int = 0
    duplicates_skipped: int = 0
    feed_stats: dict[str, int] = field(default_factory=dict)
    articles: list[RawNewsArticle] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_fetched": self.total_fetched,
            "new_articles_added": self.new_articles_added,
            "duplicates_skipped": self.duplicates_skipped,
            "feed_stats": self.feed_stats,
        }


class RSSNewsCollector:
    """Collector ingesting financial news from reputable Pakistani media RSS feeds."""

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }

    def __init__(
        self,
        feeds: Optional[dict[str, str]] = None,
        storage_paths: Optional[dict[str, Path]] = None,
        client: Optional[httpx.Client] = None,
        timeout: float = 15.0,
    ) -> None:
        self.feeds = feeds or DEFAULT_RSS_FEEDS
        if storage_paths is None:
            cfg = load_config()
            self.storage_paths = ensure_directories(cfg.settings.data_dir)
        else:
            self.storage_paths = storage_paths

        self._client = client
        self.timeout = timeout
        self.news_file = self.storage_paths["raw_news"] / "news.parquet"

    def _clean_text(self, raw_html: str) -> str:
        """Strip HTML tags and normalize whitespace."""
        if not raw_html:
            return ""
        soup = BeautifulSoup(raw_html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return " ".join(text.split())

    def _standardize_date(self, date_str: str) -> str:
        """Convert standard RFC 2822 or ISO dates to ISO format."""
        cleaned = date_str.strip()
        try:
            dt = parsedate_to_datetime(cleaned)
            return dt.isoformat()
        except Exception:
            pass

        try:
            dt_iso = datetime.datetime.fromisoformat(cleaned)
            return dt_iso.isoformat()
        except Exception:
            pass

        return cleaned

    def parse_feed_xml(self, xml_content: bytes | str, source_name: str) -> list[RawNewsArticle]:
        """Parse RSS 2.0 or Atom XML content into a list of RawNewsArticle instances."""
        if isinstance(xml_content, str):
            xml_bytes = xml_content.encode("utf-8")
        else:
            xml_bytes = xml_content

        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as e:
            logger.warning(f"[{source_name}] XML parsing failed: {e}")
            return []

        articles: list[RawNewsArticle] = []

        # 1. Standard RSS 2.0 items (<rss><channel><item>...)
        items = root.findall(".//item")
        if items:
            for item in items:
                title = item.findtext("title") or ""
                link = item.findtext("link") or ""
                desc = item.findtext("description") or ""
                pub_date = item.findtext("pubDate") or ""
                guid = item.findtext("guid") or ""
                author = item.findtext("author") or item.findtext(
                    "{http://purl.org/dc/elements/1.1/}creator"
                )

                if not title.strip() or not link.strip():
                    continue

                clean_summary = self._clean_text(desc)
                std_date = self._standardize_date(pub_date)

                articles.append(
                    RawNewsArticle(
                        source=source_name,
                        headline=title.strip(),
                        summary=clean_summary,
                        url=link.strip(),
                        published_at=std_date,
                        author=author.strip() if author else None,
                        raw_guid=guid.strip() if guid else None,
                    )
                )
            return articles

        # 2. Atom feed entries (<feed><entry>...)
        entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")
        if not entries:
            entries = root.findall(".//entry")

        for entry in entries:
            title = (
                entry.findtext("{http://www.w3.org/2005/Atom}title")
                or entry.findtext("title")
                or ""
            )
            link_elem = entry.find("{http://www.w3.org/2005/Atom}link")
            if link_elem is None:
                link_elem = entry.find("link")
            link = link_elem.get("href", "") if link_elem is not None else ""
            summary = (
                entry.findtext("{http://www.w3.org/2005/Atom}summary")
                or entry.findtext("summary")
                or entry.findtext("{http://www.w3.org/2005/Atom}content")
                or entry.findtext("content")
                or ""
            )
            pub_date = (
                entry.findtext("{http://www.w3.org/2005/Atom}published")
                or entry.findtext("published")
                or entry.findtext("{http://www.w3.org/2005/Atom}updated")
                or entry.findtext("updated")
                or ""
            )
            guid = entry.findtext("{http://www.w3.org/2005/Atom}id") or entry.findtext("id") or ""

            if not title.strip():
                continue

            clean_summary = self._clean_text(summary)
            std_date = self._standardize_date(pub_date)

            articles.append(
                RawNewsArticle(
                    source=source_name,
                    headline=title.strip(),
                    summary=clean_summary,
                    url=str(link).strip(),
                    published_at=std_date,
                    author=None,
                    raw_guid=guid.strip() if guid else None,
                )
            )

        return articles

    def fetch_all(self, dry_run: bool = False) -> RSSCollectionResult:
        """Fetch all configured RSS feeds, deduplicate, and persist to Parquet."""
        result = RSSCollectionResult()
        existing_ids: set[str] = set()

        # Load existing IDs to prevent duplicates
        if self.news_file.exists():
            existing_df = read_parquet(self.news_file)
            if not existing_df.empty and "article_id" in existing_df.columns:
                existing_ids = set(existing_df["article_id"].dropna().astype(str))

        should_close = False
        client = self._client
        if client is None:
            client = httpx.Client(
                headers=self.DEFAULT_HEADERS,
                timeout=self.timeout,
                follow_redirects=True,
            )
            should_close = True

        all_new_articles: list[RawNewsArticle] = []

        try:
            for feed_name, feed_url in self.feeds.items():
                logger.info(f"Fetching RSS feed: '{feed_name}' ({feed_url})...")
                try:
                    resp = client.get(feed_url)
                    if resp.status_code != 200:
                        logger.warning(
                            f"Feed '{feed_name}' returned HTTP {resp.status_code}. Skipping."
                        )
                        result.feed_stats[feed_name] = 0
                        continue

                    parsed_articles = self.parse_feed_xml(resp.content, source_name=feed_name)
                    result.feed_stats[feed_name] = len(parsed_articles)
                    result.total_fetched += len(parsed_articles)

                    for art in parsed_articles:
                        if art.article_id in existing_ids:
                            result.duplicates_skipped += 1
                        else:
                            existing_ids.add(art.article_id)
                            all_new_articles.append(art)
                            result.new_articles_added += 1

                except Exception as e:
                    logger.error(f"Error fetching RSS feed '{feed_name}': {e}")
                    result.feed_stats[feed_name] = 0

        finally:
            if should_close:
                client.close()

        result.articles = all_new_articles

        # Persist new articles atomically
        if all_new_articles and not dry_run:
            new_df = pd.DataFrame([a.to_dict() for a in all_new_articles])
            if self.news_file.exists():
                existing_df = read_parquet(self.news_file)
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["article_id"], keep="last")
            else:
                combined = new_df

            write_parquet_atomic(combined, self.news_file)
            logger.info(
                f"Atomically wrote {len(all_new_articles)} new articles to {self.news_file}"
            )

        return result
