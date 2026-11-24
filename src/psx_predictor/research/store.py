"""Discovery, loading and validation of research files in data/research/."""

import datetime
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from pydantic import ValidationError

from psx_predictor.config.loader import load_config
from psx_predictor.research.schemas import (
    EXPECTED_TOP_LEVEL_KEYS,
    OPTIONAL_TOP_LEVEL_KEYS,
    RESEARCH_SCHEMA_VERSION,
    ResearchReport,
)
from psx_predictor.storage.paths import ensure_directories

FILE_NAME_PATTERN = re.compile(r"^(?P<symbol>[A-Z0-9]+)_(?P<date>\d{4}-\d{2}-\d{2})$")


@dataclass
class ResearchFile:
    """A research JSON file on disk with its parse and validation outcome."""

    path: Path
    symbol: str
    research_date: Optional[datetime.date]
    report: Optional[ResearchReport] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.report is not None and not self.errors

    @property
    def markdown_path(self) -> Path:
        return self.path.with_suffix(".md")

    def read_markdown(self) -> Optional[str]:
        md = self.markdown_path
        return md.read_text(encoding="utf-8") if md.exists() else None


def _format_validation_error(exc: ValidationError) -> list[str]:
    messages = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        messages.append(f"{loc}: {err['msg']}")
    return messages


def validate_research_data(
    data: Any, expected_symbol: Optional[str] = None, expected_date: Optional[datetime.date] = None
) -> tuple[Optional[ResearchReport], list[str], list[str]]:
    """Validate parsed JSON against the research schema.

    Returns:
        (report or None, errors, warnings). A report with errors is not used for features.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, dict):
        return None, ["Top level must be a JSON object."], warnings

    try:
        report = ResearchReport.model_validate(data)
    except ValidationError as exc:
        return None, _format_validation_error(exc), warnings

    if report.schema_version != RESEARCH_SCHEMA_VERSION:
        warnings.append(
            f"schema_version is '{report.schema_version}', app expects "
            f"'{RESEARCH_SCHEMA_VERSION}'."
        )

    symbol = report.symbol.strip().upper()
    if expected_symbol and symbol != expected_symbol:
        errors.append(f"symbol '{report.symbol}' does not match file name '{expected_symbol}'.")
    if expected_date and report.research_date != expected_date:
        errors.append(
            f"research_date {report.research_date} does not match file name date "
            f"{expected_date}."
        )

    extra = sorted(set(data.keys()) - EXPECTED_TOP_LEVEL_KEYS)
    if extra:
        warnings.append(f"Unknown top-level keys ignored: {', '.join(extra)}")
    missing = sorted(EXPECTED_TOP_LEVEL_KEYS - OPTIONAL_TOP_LEVEL_KEYS - set(data.keys()))
    if missing:
        warnings.append(f"Missing keys filled with defaults: {', '.join(missing)}")

    if report.data_as_of and report.data_as_of > report.research_date:
        errors.append(
            f"data_as_of {report.data_as_of} is after research_date {report.research_date}."
        )

    future_news = [n.date for n in report.news if n.date > report.research_date]
    if future_news:
        errors.append(
            f"{len(future_news)} news item(s) dated after research_date (look-ahead)."
        )

    scores = report.scores.model_dump()
    null_scores = [k for k, v in scores.items() if v is None]
    if null_scores:
        warnings.append(f"Scores left empty: {', '.join(null_scores)}")

    source_ids = {s.id for s in report.sources}
    referenced: set[str] = set()
    for section in (
        report.earnings,
        report.dividends.history,
        report.upcoming_events,
        report.catalysts,
        report.risks,
        report.news,
        report.analyst_views,
    ):
        for item in section:
            referenced.update(item.source_ids)
    for single in (report.fundamentals, report.sector_view, report.management):
        referenced.update(single.source_ids)
    unknown = sorted(referenced - source_ids)
    if unknown:
        warnings.append(f"source_ids not found in sources: {', '.join(unknown)}")
    if not report.sources:
        warnings.append("No sources listed.")

    return report, errors, warnings


class ResearchStore:
    """Reads research JSON files from data/research/."""

    def __init__(
        self,
        storage_paths: Optional[dict[str, Path]] = None,
        research_dir: Optional[Path] = None,
    ) -> None:
        if research_dir is not None:
            self.research_dir = Path(research_dir)
        else:
            if storage_paths is None:
                config = load_config()
                storage_paths = ensure_directories(config.settings.data_dir)
            research_dir = storage_paths.get("research")
            if research_dir is None and "root" in storage_paths:
                research_dir = Path(storage_paths["root"]) / "research"
            # Without a known data root there is no research folder; scans return nothing.
            self.research_dir = Path(research_dir) if research_dir else Path("__no_research__")

    def load_file(self, path: Path) -> ResearchFile:
        """Parse and validate a single research JSON file."""
        path = Path(path)
        match = FILE_NAME_PATTERN.match(path.stem)
        if match:
            name_symbol = match.group("symbol")
            try:
                name_date: Optional[datetime.date] = datetime.date.fromisoformat(
                    match.group("date")
                )
            except ValueError:
                name_date = None
        else:
            name_symbol = path.stem.split("_")[0].upper()
            name_date = None

        rf = ResearchFile(path=path, symbol=name_symbol, research_date=name_date)
        if not match or name_date is None:
            rf.errors.append("File name must be <SYMBOL>_<YYYY-MM-DD>.json")

        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            rf.errors.append(f"Cannot read JSON: {exc}")
            return rf

        report, errors, warnings = validate_research_data(
            data, expected_symbol=name_symbol if match else None, expected_date=name_date
        )
        rf.report = report
        rf.errors.extend(errors)
        rf.warnings.extend(warnings)
        if report is not None:
            rf.symbol = report.symbol.strip().upper()
            rf.research_date = report.research_date
        return rf

    def scan(self, symbol: Optional[str] = None) -> list[ResearchFile]:
        """Load every research file (valid or not), newest first."""
        if not self.research_dir.exists():
            return []
        pattern = f"{symbol.strip().upper()}_*.json" if symbol else "*.json"
        files = [self.load_file(p) for p in sorted(self.research_dir.glob(pattern))]
        if symbol:
            files = [f for f in files if f.symbol == symbol.strip().upper()]
        files.sort(
            key=lambda f: (f.research_date or datetime.date.min, f.path.name), reverse=True
        )
        return files

    def load_reports(self, symbol: str) -> list[ResearchReport]:
        """Valid reports for a symbol, oldest first. Invalid files are skipped with a warning."""
        reports = []
        for rf in self.scan(symbol):
            if rf.is_valid and rf.report is not None:
                reports.append(rf.report)
            else:
                logger.warning(f"Skipping invalid research file {rf.path.name}: {rf.errors}")
        reports.sort(key=lambda r: r.research_date)
        return reports

    def latest(
        self, symbol: str, as_of: Optional[datetime.date] = None
    ) -> Optional[ResearchReport]:
        """Newest valid report for a symbol dated on or before as_of (default: any)."""
        reports = self.load_reports(symbol)
        if as_of is not None:
            reports = [r for r in reports if r.research_date <= as_of]
        return reports[-1] if reports else None
