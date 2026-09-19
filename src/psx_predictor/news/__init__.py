"""PSX Predictor news package: collectors, entity matching, classification, and sentiment NLP."""

from psx_predictor.news.classifier import (
    EventCategory,
    EventClassifier,
)
from psx_predictor.news.dps_announcements import (
    DPSAnnouncementsCollector,
    DPSAnnouncementsResult,
)
from psx_predictor.news.entity_matcher import (
    EntityMatcher,
)
from psx_predictor.news.processor import (
    NewsNLPProcessor,
    NLPProcessingResult,
    ProcessedNewsSignal,
    compute_signal_id,
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
from psx_predictor.news.sentiment import (
    FinancialSentimentAnalyzer,
    SentimentResult,
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
    "EntityMatcher",
    "EventCategory",
    "EventClassifier",
    "FinancialSentimentAnalyzer",
    "SentimentResult",
    "ProcessedNewsSignal",
    "NewsNLPProcessor",
    "NLPProcessingResult",
    "compute_signal_id",
]
