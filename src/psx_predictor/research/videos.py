"""Research presentation videos made with docs/research/antigravity_video_prompt.md.

Videos live in data/research/videos/ as <SYMBOL>_<YYYY-MM-DD>.mp4 with an optional .srt
next to them, plus index.json listing them. Files on disk are the source of truth:
index.json only adds details (duration, style, language) and may be missing or stale.
"""

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from psx_predictor.research.store import FILE_NAME_PATTERN


@dataclass
class ResearchVideo:
    symbol: str
    research_date: datetime.date
    video_path: Path
    subtitles_path: Optional[Path] = None
    duration_sec: Optional[float] = None
    style: Optional[str] = None
    language: Optional[str] = None

    @property
    def key(self) -> tuple[str, datetime.date]:
        return (self.symbol, self.research_date)


def _read_index(videos_dir: Path) -> dict[str, dict[str, Any]]:
    """index.json entries keyed by video file name; {} when missing or unreadable."""
    index_file = videos_dir / "index.json"
    if not index_file.exists():
        return {}
    try:
        data = json.loads(index_file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        logger.warning(f"Cannot read {index_file}: {exc}")
        return {}
    if isinstance(data, dict):
        data = data.get("videos", [])
    entries: dict[str, dict[str, Any]] = {}
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and entry.get("video"):
                entries[Path(str(entry["video"])).name] = entry
    return entries


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def list_research_videos(research_dir: Path) -> list[ResearchVideo]:
    """All videos in <research_dir>/videos/, newest research date first."""
    videos_dir = Path(research_dir) / "videos"
    if not videos_dir.exists():
        return []
    index = _read_index(videos_dir)
    videos = []
    for mp4 in videos_dir.glob("*.mp4"):
        match = FILE_NAME_PATTERN.match(mp4.stem)
        if not match:
            continue
        try:
            date = datetime.date.fromisoformat(match.group("date"))
        except ValueError:
            continue
        entry = index.get(mp4.name, {})
        srt = mp4.with_suffix(".srt")
        if entry.get("subtitles"):
            named = videos_dir / Path(str(entry["subtitles"])).name
            srt = named if named.exists() else srt
        videos.append(
            ResearchVideo(
                symbol=match.group("symbol"),
                research_date=date,
                video_path=mp4,
                subtitles_path=srt if srt.exists() else None,
                duration_sec=_as_float(entry.get("duration_sec")),
                style=entry.get("style"),
                language=entry.get("language"),
            )
        )
    videos.sort(key=lambda v: (v.research_date, v.symbol), reverse=True)
    return videos


def find_research_video(
    research_dir: Path, symbol: str, research_date: Optional[datetime.date]
) -> Optional[ResearchVideo]:
    """The video for one research report, or None."""
    if research_date is None:
        return None
    sym = symbol.strip().upper()
    for video in list_research_videos(research_dir):
        if video.symbol == sym and video.research_date == research_date:
            return video
    return None
