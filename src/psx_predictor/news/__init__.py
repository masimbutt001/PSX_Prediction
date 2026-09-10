"""PSX Predictor news package: financial RSS collectors and DPS corporate announcements."""

from psx_predictor.news.dps_announcements import (
    DPSAnnouncementsCollector,
    DPSAnnouncementsResult,
)
from psx_predictor.news.rss_collector import (
    DEFAULT_RSS_FEEDS,
    RSSCollectionResult,
    RSSNewsCollector,
)
from psx_predictor.news.schemas import (
    RawAnnouncement,
    RawNewsArticle,
    compute_announcement_id,
    compute_news_id,
)

__all__ = [
    "RawNewsArticle",
    "RawAnnouncement",
    "compute_news_id",
    "compute_announcement_id",
    "RSSNewsCollector",
    "RSSCollectionResult",
    "DEFAULT_RSS_FEEDS",
    "DPSAnnouncementsCollector",
    "DPSAnnouncementsResult",
]
