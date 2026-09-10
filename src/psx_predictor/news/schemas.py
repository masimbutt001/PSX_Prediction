"""Pydantic schemas and deterministic content-hashing invariants for news and announcements."""

import datetime
import hashlib
from typing import Any, Optional

from pydantic import BaseModel, Field


def compute_news_id(source: str, headline: str, published_at: str) -> str:
    """Compute deterministic SHA256 identifier for a news article."""
    raw_key = f"{source.strip().lower()}:{headline.strip().lower()}:{published_at.strip()}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def compute_announcement_id(
    symbol: str,
    title: str,
    announcement_date: str,
    announcement_time: str,
) -> str:
    """Compute deterministic SHA256 identifier for an official exchange announcement."""
    raw_key = (
        f"{symbol.strip().upper()}:{title.strip().lower()}:"
        f"{announcement_date.strip()}:{announcement_time.strip().lower()}"
    )
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class RawNewsArticle(BaseModel):
    """Immutable data contract representing a raw collected financial news article."""

    article_id: str = Field(default="")
    source: str
    headline: str
    summary: str = ""
    url: str
    published_at: str
    fetched_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    author: Optional[str] = None
    raw_guid: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        """Ensure article_id is deterministically populated if omitted."""
        if not self.article_id:
            self.article_id = compute_news_id(
                source=self.source,
                headline=self.headline,
                published_at=self.published_at,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize article model to dictionary."""
        return self.model_dump()


class RawAnnouncement(BaseModel):
    """Immutable data contract representing an official PSX corporate announcement."""

    announcement_id: str = Field(default="")
    symbol: str
    company_name: str
    title: str
    category: str = "Company Announcement"
    announcement_date: str  # YYYY-MM-DD or formatted date
    announcement_time: str  # e.g. "3:24 PM"
    document_url: Optional[str] = None
    source: str = "PSX DPS"
    fetched_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def model_post_init(self, __context: Any) -> None:
        """Ensure announcement_id is deterministically populated if omitted."""
        if not self.announcement_id:
            self.announcement_id = compute_announcement_id(
                symbol=self.symbol,
                title=self.title,
                announcement_date=self.announcement_date,
                announcement_time=self.announcement_time,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize announcement model to dictionary."""
        return self.model_dump()
