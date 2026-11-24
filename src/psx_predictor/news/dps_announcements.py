"""PSX Data Portal (DPS) official company announcements collector and parser."""

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
import pandas as pd
from bs4 import BeautifulSoup, Tag
from loguru import logger

from psx_predictor.config.loader import load_config
from psx_predictor.news.schemas import RawAnnouncement
from psx_predictor.storage.parquet_io import read_parquet, write_parquet_atomic
from psx_predictor.storage.paths import ensure_directories


@dataclass
class DPSAnnouncementsResult:
    """Summary of a PSX DPS announcements collection execution."""

    total_fetched: int = 0
    new_announcements_added: int = 0
    duplicates_skipped: int = 0
    symbol_filter: Optional[str] = None
    announcements: list[RawAnnouncement] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_fetched": self.total_fetched,
            "new_announcements_added": self.new_announcements_added,
            "duplicates_skipped": self.duplicates_skipped,
            "symbol_filter": self.symbol_filter,
        }


class DPSAnnouncementsCollector:
    """Collector fetching official corporate announcements from PSX Data Portal."""

    ANNOUNCEMENTS_URL = "https://dps.psx.com.pk/announcements"
    BASE_URL = "https://dps.psx.com.pk"
    DEFAULT_TIMEOUT = 15.0
    DATE_ONLY_TIME = "11:59 PM"

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://dps.psx.com.pk/announcements",
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "*/*",
    }

    def __init__(
        self,
        storage_paths: Optional[dict[str, Path]] = None,
        client: Optional[httpx.Client] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if storage_paths is None:
            cfg = load_config()
            self.storage_paths = ensure_directories(cfg.settings.data_dir)
        else:
            self.storage_paths = storage_paths

        self._client = client
        self.timeout = timeout
        self.announcements_file = self.storage_paths["raw_announcements"] / "announcements.parquet"

    def _parse_date(self, raw_date_str: str) -> str:
        """Standardize various PSX date formats (e.g. 'Sep 10, 2026') to 'YYYY-MM-DD'."""
        cleaned = raw_date_str.strip()
        for fmt in ("%b %d, %Y", "%B %d, %Y", "%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d"):
            try:
                dt = datetime.datetime.strptime(cleaned, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return cleaned

    def parse_html_table(self, html_text: str) -> list[RawAnnouncement]:
        """Parse the HTML table returned by PSX DPS announcements endpoint."""
        soup = BeautifulSoup(html_text, "html.parser")
        table = soup.find("table")
        if isinstance(table, Tag):
            rows = table.find_all("tr")
        else:
            rows = soup.find_all("tr")

        announcements: list[RawAnnouncement] = []

        for row in rows:
            if not isinstance(row, Tag):
                continue
            cells = row.find_all("td")
            if len(cells) < 5:
                # Header row or empty row
                continue

            raw_date = cells[0].get_text(strip=True)
            time_str = cells[1].get_text(strip=True)
            sym = cells[2].get_text(strip=True).upper()
            company_name = cells[3].get_text(strip=True)
            title = cells[4].get_text(strip=True)

            if not sym or not title:
                continue

            # Extract PDF document link if present
            doc_url: Optional[str] = None
            for a_tag in row.find_all("a"):
                href = a_tag.get("href", "")
                if "/download/document/" in href or href.endswith(".pdf"):
                    doc_url = f"{self.BASE_URL}{href}" if href.startswith("/") else href
                    break

            std_date = self._parse_date(raw_date)

            announcements.append(
                RawAnnouncement(
                    symbol=sym,
                    company_name=company_name,
                    title=title,
                    category="Company Announcement",
                    announcement_date=std_date,
                    announcement_time=time_str,
                    document_url=doc_url,
                    source="PSX DPS",
                )
            )

        return announcements

    def parse_company_page(self, html_text: str, symbol: str) -> list[RawAnnouncement]:
        """Parse the Announcements tabs on a PSX Data Portal company page (/company/<SYMBOL>)."""
        soup = BeautifulSoup(html_text, "html.parser")
        sym = symbol.strip().upper()
        name_tag = soup.find(class_="quote__name")
        company_name = name_tag.get_text(strip=True) if isinstance(name_tag, Tag) else sym

        section = soup.find(id="announcements")
        if not isinstance(section, Tag):
            return []

        announcements: list[RawAnnouncement] = []
        for panel in section.find_all("div", class_="tabs__panel"):
            if not isinstance(panel, Tag):
                continue
            category = str(panel.get("data-name") or "Company Announcement").strip()
            for row in panel.find_all("tr"):
                if not isinstance(row, Tag):
                    continue
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue
                title = cells[1].get_text(" ", strip=True)
                std_date = self._parse_date(cells[0].get_text(strip=True))
                if not title or len(std_date) != 10:
                    continue
                doc_url: Optional[str] = None
                for a_tag in row.find_all("a"):
                    href = str(a_tag.get("href", ""))
                    if "/download/document/" in href or href.endswith(".pdf"):
                        doc_url = f"{self.BASE_URL}{href}" if href.startswith("/") else href
                        break
                announcements.append(
                    RawAnnouncement(
                        symbol=sym,
                        company_name=company_name,
                        title=title,
                        category=category,
                        announcement_date=std_date,
                        # Company pages give no time. Treat the item as released after the
                        # close so it only feeds the next session (no look-ahead).
                        announcement_time=self.DATE_ONLY_TIME,
                        document_url=doc_url,
                        source="PSX DPS company page",
                    )
                )
        return announcements

    def fetch_company_announcements(
        self, symbols: list[str], dry_run: bool = False
    ) -> DPSAnnouncementsResult:
        """Read announcements from each company's PSX Data Portal page, then dedupe and save.

        Used when the announcements search endpoint refuses requests (it returns HTTP 403
        to scripts), because company pages are plain HTML that still loads.
        """
        result = DPSAnnouncementsResult(symbol_filter=",".join(symbols) or None)
        should_close = False
        client = self._client
        if client is None:
            headers = {k: v for k, v in self.DEFAULT_HEADERS.items() if k != "X-Requested-With"}
            client = httpx.Client(headers=headers, timeout=self.timeout, follow_redirects=True)
            should_close = True

        parsed: list[RawAnnouncement] = []
        failures: list[str] = []
        try:
            for raw_sym in symbols:
                sym = raw_sym.strip().upper()
                try:
                    resp = client.get(f"{self.BASE_URL}/company/{sym}")
                    resp.raise_for_status()
                    parsed.extend(self.parse_company_page(resp.text, sym))
                except Exception as exc:
                    logger.warning(f"Could not read PSX company page for {sym}: {exc}")
                    failures.append(sym)
        finally:
            if should_close:
                client.close()

        if symbols and len(failures) == len(symbols):
            raise RuntimeError(f"PSX company pages unreachable for {', '.join(failures)}")
        result.total_fetched = len(parsed)
        self._store_new(parsed, result, dry_run)
        return result

    def _store_new(
        self, parsed: list[RawAnnouncement], result: DPSAnnouncementsResult, dry_run: bool
    ) -> None:
        """Skip announcements already saved and atomically append the rest."""
        existing_ids: set[str] = set()
        if self.announcements_file.exists():
            existing_df = read_parquet(self.announcements_file)
            if not existing_df.empty and "announcement_id" in existing_df.columns:
                existing_ids = set(existing_df["announcement_id"].dropna().astype(str))

        new_items: list[RawAnnouncement] = []
        for ann in parsed:
            if ann.announcement_id in existing_ids:
                result.duplicates_skipped += 1
            else:
                existing_ids.add(ann.announcement_id)
                new_items.append(ann)
                result.new_announcements_added += 1
        result.announcements = new_items

        if new_items and not dry_run:
            new_df = pd.DataFrame([a.to_dict() for a in new_items])
            if self.announcements_file.exists():
                existing_df = read_parquet(self.announcements_file)
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["announcement_id"], keep="last")
            else:
                combined = new_df
            write_parquet_atomic(combined, self.announcements_file)
            logger.info(
                f"Atomically wrote {len(new_items)} new announcements to {self.announcements_file}"
            )

    def fetch_announcements(
        self,
        symbol: Optional[str] = None,
        count: int = 50,
        offset: int = 0,
        date_from: str = "",
        date_to: str = "",
        dry_run: bool = False,
    ) -> DPSAnnouncementsResult:
        """Fetch latest corporate announcements from PSX DPS, deduplicate, and persist."""
        clean_sym = symbol.strip().upper() if symbol else ""
        result = DPSAnnouncementsResult(symbol_filter=clean_sym or None)

        existing_ids: set[str] = set()
        if self.announcements_file.exists():
            existing_df = read_parquet(self.announcements_file)
            if not existing_df.empty and "announcement_id" in existing_df.columns:
                existing_ids = set(existing_df["announcement_id"].dropna().astype(str))

        payload = {
            "type": "C",  # Companies Announcements
            "symbol": clean_sym,
            "query": "",
            "count": count,
            "offset": offset,
            "date_from": date_from,
            "date_to": date_to,
        }

        should_close = False
        client = self._client
        if client is None:
            client = httpx.Client(
                headers=self.DEFAULT_HEADERS,
                timeout=self.timeout,
                follow_redirects=True,
            )
            should_close = True

        all_new_announcements: list[RawAnnouncement] = []

        try:
            logger.info(
                f"Fetching PSX announcements (Symbol: '{clean_sym or 'ALL'}', Count: {count})..."
            )
            resp = client.post(self.ANNOUNCEMENTS_URL, data=payload)
            if resp.status_code != 200:
                logger.warning(
                    f"DPS announcements endpoint returned HTTP {resp.status_code}; "
                    "reading company pages instead."
                )
                fallback_symbols = (
                    [clean_sym]
                    if clean_sym
                    else [s.symbol for s in load_config().get_enabled_stocks()]
                )
                if should_close:
                    client.close()
                    should_close = False
                return self.fetch_company_announcements(fallback_symbols, dry_run=dry_run)

            parsed_list = self.parse_html_table(resp.text)
            result.total_fetched = len(parsed_list)

            for ann in parsed_list:
                if ann.announcement_id in existing_ids:
                    result.duplicates_skipped += 1
                else:
                    existing_ids.add(ann.announcement_id)
                    all_new_announcements.append(ann)
                    result.new_announcements_added += 1

        except Exception as e:
            logger.error(f"Error fetching DPS announcements: {e}")
            raise
        finally:
            if should_close:
                client.close()

        result.announcements = all_new_announcements

        # Atomically write to Parquet
        if all_new_announcements and not dry_run:
            new_df = pd.DataFrame([a.to_dict() for a in all_new_announcements])
            if self.announcements_file.exists():
                existing_df = read_parquet(self.announcements_file)
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["announcement_id"], keep="last")
            else:
                combined = new_df

            write_parquet_atomic(combined, self.announcements_file)
            logger.info(
                f"Atomically wrote {len(all_new_announcements)} new announcements "
                f"to {self.announcements_file}"
            )

        return result
