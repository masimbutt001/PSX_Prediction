"""NLP processing pipeline transforming raw text into structured quantitative signals."""

import datetime
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

from psx_predictor.config.loader import load_config
from psx_predictor.news.classifier import EventClassifier
from psx_predictor.news.entity_matcher import EntityMatcher
from psx_predictor.news.sentiment import FinancialSentimentAnalyzer
from psx_predictor.storage.parquet_io import read_parquet, write_parquet_atomic
from psx_predictor.storage.paths import ensure_directories


def compute_signal_id(raw_id: str, primary_symbol: str, category: str) -> str:
    """Compute deterministic SHA256 identifier for a processed news signal."""
    raw_key = f"{raw_id.strip()}:{primary_symbol.strip().upper()}:{category.strip().upper()}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class ProcessedNewsSignal(BaseModel):
    """Structured quantitative signal extracted from financial news or company notices."""

    signal_id: str = Field(default="")
    raw_id: str
    source: str
    published_at: str
    headline: str
    matched_symbols: list[str] = Field(default_factory=list)
    primary_symbol: Optional[str] = None
    event_category: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_label: str  # "POSITIVE", "NEGATIVE", "NEUTRAL"
    confidence: float = Field(ge=0.0, le=1.0)
    positive_tokens: list[str] = Field(default_factory=list)
    negative_tokens: list[str] = Field(default_factory=list)
    processed_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

    def model_post_init(self, __context: Any) -> None:
        if not self.signal_id:
            sym_key = self.primary_symbol or "MACRO"
            self.signal_id = compute_signal_id(
                raw_id=self.raw_id,
                primary_symbol=sym_key,
                category=self.event_category,
            )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


@dataclass
class NLPProcessingResult:
    """Summary of batch NLP extraction execution."""

    total_news_inspected: int = 0
    total_announcements_inspected: int = 0
    new_signals_created: int = 0
    duplicates_skipped: int = 0
    symbol_distribution: dict[str, int] = field(default_factory=dict)
    category_distribution: dict[str, int] = field(default_factory=dict)
    sentiment_distribution: dict[str, int] = field(default_factory=dict)
    signals: list[ProcessedNewsSignal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_news_inspected": self.total_news_inspected,
            "total_announcements_inspected": self.total_announcements_inspected,
            "new_signals_created": self.new_signals_created,
            "duplicates_skipped": self.duplicates_skipped,
            "symbol_distribution": self.symbol_distribution,
            "category_distribution": self.category_distribution,
            "sentiment_distribution": self.sentiment_distribution,
        }


class NewsNLPProcessor:
    """Extracts entities, event categories, and sentiment signals from raw texts."""

    def __init__(self, storage_paths: Optional[dict[str, Path]] = None) -> None:
        if storage_paths is None:
            cfg = load_config()
            self.storage_paths = ensure_directories(cfg.settings.data_dir)
        else:
            self.storage_paths = storage_paths

        self.matcher = EntityMatcher()
        self.classifier = EventClassifier()
        self.sentiment = FinancialSentimentAnalyzer()

        self.news_file = self.storage_paths["raw_news"] / "news.parquet"
        self.announcements_file = self.storage_paths["raw_announcements"] / "announcements.parquet"
        self.signals_file = self.storage_paths["processed_news"] / "news_signals.parquet"

    def process_all(self, dry_run: bool = False) -> NLPProcessingResult:
        """Process all raw news articles and corporate announcements into structured signals."""
        result = NLPProcessingResult()
        existing_ids: set[str] = set()

        if self.signals_file.exists():
            existing_df = read_parquet(self.signals_file)
            if not existing_df.empty and "signal_id" in existing_df.columns:
                existing_ids = set(existing_df["signal_id"].dropna().astype(str))

        new_signals: list[ProcessedNewsSignal] = []

        # 1. Process Raw News Articles
        if self.news_file.exists():
            news_df = read_parquet(self.news_file)
            result.total_news_inspected = len(news_df)

            for _, row in news_df.iterrows():
                headline = str(row.get("headline", ""))
                summary = str(row.get("summary", ""))
                full_text = f"{headline}. {summary}" if summary else headline

                matched_symbols = self.matcher.match(full_text)
                primary_sym = matched_symbols[0] if matched_symbols else None
                event_cat, cat_conf = self.classifier.classify(full_text)
                sent = self.sentiment.analyze(full_text)

                combined_conf = round((cat_conf + sent.confidence) / 2.0, 2)
                raw_id = str(row.get("article_id", ""))

                signal = ProcessedNewsSignal(
                    raw_id=raw_id,
                    source=str(row.get("source", "RSS")),
                    published_at=str(row.get("published_at", "")),
                    headline=headline,
                    matched_symbols=matched_symbols,
                    primary_symbol=primary_sym,
                    event_category=event_cat,
                    sentiment_score=sent.score,
                    sentiment_label=sent.label,
                    confidence=combined_conf,
                    positive_tokens=sent.positive_tokens,
                    negative_tokens=sent.negative_tokens,
                )

                if signal.signal_id in existing_ids:
                    result.duplicates_skipped += 1
                else:
                    existing_ids.add(signal.signal_id)
                    new_signals.append(signal)
                    result.new_signals_created += 1

                    # Update statistics
                    sym_key = primary_sym or "MACRO/GENERAL"
                    result.symbol_distribution[sym_key] = (
                        result.symbol_distribution.get(sym_key, 0) + 1
                    )
                    result.category_distribution[event_cat] = (
                        result.category_distribution.get(event_cat, 0) + 1
                    )
                    result.sentiment_distribution[sent.label] = (
                        result.sentiment_distribution.get(sent.label, 0) + 1
                    )

        # 2. Process Raw Corporate Announcements
        if self.announcements_file.exists():
            ann_df = read_parquet(self.announcements_file)
            result.total_announcements_inspected = len(ann_df)

            for _, row in ann_df.iterrows():
                title = str(row.get("title", ""))
                sym = str(row.get("symbol", "")).upper()
                date_str = str(row.get("announcement_date", ""))
                time_str = str(row.get("announcement_time", ""))
                pub_at = f"{date_str} {time_str}".strip()

                matched_symbols = [sym] if sym else []
                # Also check title for additional partner/acquired symbols
                extra_syms = self.matcher.match(title)
                for s in extra_syms:
                    if s not in matched_symbols:
                        matched_symbols.append(s)

                event_cat, cat_conf = self.classifier.classify(title)
                sent = self.sentiment.analyze(title)

                combined_conf = round((cat_conf + sent.confidence) / 2.0, 2)
                raw_id = str(row.get("announcement_id", ""))

                signal = ProcessedNewsSignal(
                    raw_id=raw_id,
                    source="PSX DPS",
                    published_at=pub_at,
                    headline=title,
                    matched_symbols=matched_symbols,
                    primary_symbol=sym or None,
                    event_category=event_cat,
                    sentiment_score=sent.score,
                    sentiment_label=sent.label,
                    confidence=combined_conf,
                    positive_tokens=sent.positive_tokens,
                    negative_tokens=sent.negative_tokens,
                )

                if signal.signal_id in existing_ids:
                    result.duplicates_skipped += 1
                else:
                    existing_ids.add(signal.signal_id)
                    new_signals.append(signal)
                    result.new_signals_created += 1

                    sym_key = sym or "GENERAL"
                    result.symbol_distribution[sym_key] = (
                        result.symbol_distribution.get(sym_key, 0) + 1
                    )
                    result.category_distribution[event_cat] = (
                        result.category_distribution.get(event_cat, 0) + 1
                    )
                    result.sentiment_distribution[sent.label] = (
                        result.sentiment_distribution.get(sent.label, 0) + 1
                    )

        result.signals = new_signals

        # Persist structured signals atomically
        if new_signals and not dry_run:
            new_df = pd.DataFrame([s.to_dict() for s in new_signals])
            if self.signals_file.exists():
                existing_df = read_parquet(self.signals_file)
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["signal_id"], keep="last")
            else:
                combined = new_df

            write_parquet_atomic(combined, self.signals_file)
            logger.info(f"Atomically wrote {len(new_signals)} new signals to {self.signals_file}")

        return result
