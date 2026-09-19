"""Market session aligner mapping news and announcements to PSX sessions without lookahead bias."""

import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dateutil import parser as date_parser
from loguru import logger

from psx_predictor.config.loader import load_config
from psx_predictor.news.calendar import (
    FRIDAY_BREAK_CLOSE,
    FRIDAY_BREAK_OPEN,
    FRIDAY_S1_CLOSE,
    FRIDAY_S1_OPEN,
    FRIDAY_S2_CLOSE,
    FRIDAY_S2_OPEN,
    PKT_TZ,
    REGULAR_CLOSE,
    REGULAR_OPEN,
    PSXMarketCalendar,
)
from psx_predictor.storage.parquet_io import read_parquet, write_parquet_atomic
from psx_predictor.storage.paths import ensure_directories, get_storage_paths


@dataclass
class SessionAlignment:
    """Temporal mapping of a news article to its earliest actionable PSX trading session."""

    session_date: datetime.date
    session_type: str  # "REGULAR", "FRIDAY_SESSION_1", "FRIDAY_SESSION_2"
    usable_from: str  # ISO timestamp with PKT timezone
    is_intraday: bool  # True if published during live market trading hours
    published_pkt: str  # Original publication timestamp normalized to PKT
    hours_to_session: float  # Buffer hours from publication until actionable trading

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_date": self.session_date.isoformat(),
            "session_type": self.session_type,
            "usable_from": self.usable_from,
            "is_intraday": self.is_intraday,
            "published_pkt": self.published_pkt,
            "hours_to_session": round(self.hours_to_session, 2),
        }


@dataclass
class AlignmentSummary:
    """Summary of batch market session alignment execution."""

    total_signals_aligned: int = 0
    session_type_distribution: dict[str, int] = field(default_factory=dict)
    intraday_signals_count: int = 0
    after_hours_signals_count: int = 0
    trading_dates_covered: int = 0
    earliest_session: Optional[str] = None
    latest_session: Optional[str] = None
    daily_feature_records: int = 0
    symbols_covered: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_to_pkt(dt_input: Any) -> datetime.datetime:
    """Safely parse arbitrary date/time input into a timezone-aware PKT datetime."""
    if isinstance(dt_input, datetime.datetime):
        if dt_input.tzinfo is None:
            # Assume local Pakistan Standard Time if timezone is unspecified
            return dt_input.replace(tzinfo=PKT_TZ)
        return dt_input.astimezone(PKT_TZ)

    if isinstance(dt_input, datetime.date) and not isinstance(dt_input, datetime.datetime):
        # Default date-only inputs to 00:00:00 PKT
        return datetime.datetime.combine(dt_input, datetime.time(0, 0), tzinfo=PKT_TZ)

    s = str(dt_input).strip()
    if not s:
        return datetime.datetime.now(PKT_TZ)

    try:
        parsed = date_parser.parse(s)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=PKT_TZ)
        return parsed.astimezone(PKT_TZ)
    except Exception:
        # Fallback to current timestamp
        return datetime.datetime.now(PKT_TZ)


class NewsSessionAligner:
    """Maps publication timestamps to actionable PSX trading sessions without lookahead."""

    def __init__(
        self,
        calendar: Optional[PSXMarketCalendar] = None,
        storage_paths: Optional[dict[str, Path]] = None,
    ) -> None:
        self.calendar = calendar or PSXMarketCalendar()

        if storage_paths is None:
            cfg = load_config()
            self.storage_paths = get_storage_paths(cfg.settings.data_dir)
        else:
            self.storage_paths = storage_paths

        ensure_directories(self.storage_paths["root"])
        self.signals_file = self.storage_paths["processed_news"] / "news_signals.parquet"
        self.aligned_file = self.storage_paths["processed_news"] / "aligned_signals.parquet"
        self.features_dir = self.storage_paths.get(
            "features_news", self.storage_paths["root"] / "features" / "news"
        )
        self.features_dir.mkdir(parents=True, exist_ok=True)

    def align_timestamp(self, published_at: Any) -> SessionAlignment:
        """Map a single publication timestamp to the earliest actionable PSX session.

        Trading Hours & Split Session Rules:
        - Weekends & Holidays: Closed -> earliest next trading day open.
        - Friday Split:
          - < 09:15: Usable for Friday Session 1 (09:15).
          - 09:15 to 12:00 (Session 1): Usable for Friday Session 2 (14:30), is_intraday=True.
          - 12:00 to 14:30 (Prayer Break): Usable for Friday Session 2 (14:30), is_intraday=False.
          - 14:30 to 16:30 (Session 2): Usable for Monday, is_intraday=True.
          - > 16:30 (After Hours): Usable for Monday, is_intraday=False.
        - Mon - Thu Regular:
          - < 09:15: Usable for Today Regular Session (09:15).
          - 09:15 to 15:30 (Market Session): Actionable for Next Trading Day Open, is_intraday=True.
          - > 15:30 (After Hours): Usable for Next Trading Day Open, is_intraday=False.

        Args:
            published_at: Timestamp (str, datetime, or date).

        Returns:
            SessionAlignment object with actionable session details.
        """
        dt_pkt = parse_to_pkt(published_at)
        date_pkt = dt_pkt.date()
        time_pkt = dt_pkt.time()

        # Case 1: Market closed (Weekend or Exchange Holiday)
        if not self.calendar.is_trading_day(date_pkt):
            target_date = self.calendar.next_trading_day(date_pkt)
            if target_date.weekday() == 4:
                session_type = "FRIDAY_SESSION_1"
                session_open = FRIDAY_S1_OPEN
            else:
                session_type = "REGULAR"
                session_open = REGULAR_OPEN

            usable_from_dt = datetime.datetime.combine(target_date, session_open, tzinfo=PKT_TZ)
            hours_buffer = max(0.0, (usable_from_dt - dt_pkt).total_seconds() / 3600.0)

            return SessionAlignment(
                session_date=target_date,
                session_type=session_type,
                usable_from=usable_from_dt.isoformat(),
                is_intraday=False,
                published_pkt=dt_pkt.isoformat(),
                hours_to_session=hours_buffer,
            )

        # Case 2: Friday Split Trading Schedule
        if date_pkt.weekday() == 4:
            if time_pkt < FRIDAY_S1_OPEN:
                target_date = date_pkt
                session_type = "FRIDAY_SESSION_1"
                session_open = FRIDAY_S1_OPEN
                is_intraday = False
            elif FRIDAY_S1_OPEN <= time_pkt < FRIDAY_S1_CLOSE:
                # During Friday Session 1 -> actionable for Friday Session 2
                target_date = date_pkt
                session_type = "FRIDAY_SESSION_2"
                session_open = FRIDAY_S2_OPEN
                is_intraday = True
            elif FRIDAY_BREAK_OPEN <= time_pkt < FRIDAY_BREAK_CLOSE:
                # During Friday Prayer Break -> actionable for Friday Session 2
                target_date = date_pkt
                session_type = "FRIDAY_SESSION_2"
                session_open = FRIDAY_S2_OPEN
                is_intraday = False
            elif FRIDAY_S2_OPEN <= time_pkt < FRIDAY_S2_CLOSE:
                # During Friday Session 2 -> actionable for Monday
                target_date = self.calendar.next_trading_day(date_pkt)
                session_type = "REGULAR"
                session_open = REGULAR_OPEN
                is_intraday = True
            else:
                # Friday After Hours (> 16:30) -> actionable for Monday
                target_date = self.calendar.next_trading_day(date_pkt)
                session_type = "REGULAR"
                session_open = REGULAR_OPEN
                is_intraday = False

            usable_from_dt = datetime.datetime.combine(target_date, session_open, tzinfo=PKT_TZ)
            hours_buffer = max(0.0, (usable_from_dt - dt_pkt).total_seconds() / 3600.0)

            return SessionAlignment(
                session_date=target_date,
                session_type=session_type,
                usable_from=usable_from_dt.isoformat(),
                is_intraday=is_intraday,
                published_pkt=dt_pkt.isoformat(),
                hours_to_session=hours_buffer,
            )

        # Case 3: Mon - Thu Regular Trading Schedule
        if time_pkt < REGULAR_OPEN:
            target_date = date_pkt
            session_type = "REGULAR"
            session_open = REGULAR_OPEN
            is_intraday = False
        elif REGULAR_OPEN <= time_pkt <= REGULAR_CLOSE:
            # During market hours -> actionable for next session
            target_date = self.calendar.next_trading_day(date_pkt)
            session_type = "FRIDAY_SESSION_1" if target_date.weekday() == 4 else "REGULAR"
            session_open = FRIDAY_S1_OPEN if target_date.weekday() == 4 else REGULAR_OPEN
            is_intraday = True
        else:
            # After market hours (> 15:30) -> actionable for next session
            target_date = self.calendar.next_trading_day(date_pkt)
            session_type = "FRIDAY_SESSION_1" if target_date.weekday() == 4 else "REGULAR"
            session_open = FRIDAY_S1_OPEN if target_date.weekday() == 4 else REGULAR_OPEN
            is_intraday = False

        usable_from_dt = datetime.datetime.combine(target_date, session_open, tzinfo=PKT_TZ)
        hours_buffer = max(0.0, (usable_from_dt - dt_pkt).total_seconds() / 3600.0)

        return SessionAlignment(
            session_date=target_date,
            session_type=session_type,
            usable_from=usable_from_dt.isoformat(),
            is_intraday=is_intraday,
            published_pkt=dt_pkt.isoformat(),
            hours_to_session=hours_buffer,
        )

    def align_dataframe(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """Align all signals in a DataFrame to their respective PSX trading sessions."""
        if signals_df.empty:
            return signals_df

        aligned_records = []
        for _, row in signals_df.iterrows():
            pub_raw = row.get("published_at", "")
            alignment = self.align_timestamp(pub_raw)
            aligned_records.append(
                {
                    "session_target_date": alignment.session_date.isoformat(),
                    "session_type": alignment.session_type,
                    "usable_from": alignment.usable_from,
                    "is_intraday": alignment.is_intraday,
                    "published_pkt": alignment.published_pkt,
                    "hours_to_session": alignment.hours_to_session,
                }
            )

        align_df = pd.DataFrame(aligned_records, index=signals_df.index)
        return pd.concat([signals_df, align_df], axis=1)

    def compute_daily_session_features(
        self,
        signals_df: pd.DataFrame,
        symbol: str,
        trading_dates: list[datetime.date],
    ) -> pd.DataFrame:
        """Compute point-in-time rolling news features for a symbol across trading sessions.

        Zero Lookahead Invariant:
        At the start of session date D (09:15 PKT), we strictly aggregate signals
        published BEFORE session open:
        - 24h window: (session_open - 24h, session_open]
        - 72h window: (session_open - 72h, session_open]

        Features:
        - sentiment_24h
        - sentiment_72h
        - news_count_24h
        - news_count_72h
        - positive_count_24h
        - negative_count_24h
        - has_earnings_announcement_today
        - has_dividend_announcement_today
        - has_discovery_announcement_today

        Args:
            signals_df: DataFrame of processed/aligned signals.
            symbol: Target stock symbol.
            trading_dates: Ordered list of trading session dates.

        Returns:
            DataFrame with one row per session date and computed feature columns.
        """
        clean_sym = symbol.strip().upper()

        # Filter signals relevant to this symbol
        if signals_df.empty:
            sym_signals = pd.DataFrame()
        else:
            # Check primary_symbol or matched_symbols list/array
            def is_relevant(row: Any) -> bool:
                prim = str(row.get("primary_symbol", "")).upper()
                if prim == clean_sym:
                    return True
                matched = row.get("matched_symbols", [])
                if isinstance(matched, (list, tuple)):
                    return clean_sym in [str(m).upper() for m in matched]
                return False

            mask = signals_df.apply(is_relevant, axis=1)
            sym_signals = signals_df[mask].copy()

        # Parse published_at into datetime with PKT timezone
        if not sym_signals.empty:
            sym_signals["pub_dt"] = sym_signals["published_at"].apply(parse_to_pkt)
            sym_signals["sentiment_score"] = pd.to_numeric(
                sym_signals["sentiment_score"], errors="coerce"
            ).fillna(0.0)
            sym_signals["event_category"] = sym_signals["event_category"].astype(str).str.upper()

        rows = []
        for dt in trading_dates:
            # Session open cutoff for date dt
            session_open = datetime.datetime.combine(dt, REGULAR_OPEN, tzinfo=PKT_TZ)
            win_24h = session_open - datetime.timedelta(hours=24)
            win_72h = session_open - datetime.timedelta(hours=72)

            if sym_signals.empty:
                rows.append(
                    {
                        "symbol": clean_sym,
                        "session_date": dt.isoformat(),
                        "sentiment_24h": 0.0,
                        "sentiment_72h": 0.0,
                        "news_count_24h": 0,
                        "news_count_72h": 0,
                        "positive_count_24h": 0,
                        "negative_count_24h": 0,
                        "has_earnings_announcement_today": 0,
                        "has_dividend_announcement_today": 0,
                        "has_discovery_announcement_today": 0,
                    }
                )
                continue

            # Zero look-ahead: pub_dt < session_open
            sig_24h = sym_signals[
                (sym_signals["pub_dt"] > win_24h) & (sym_signals["pub_dt"] <= session_open)
            ]
            sig_72h = sym_signals[
                (sym_signals["pub_dt"] > win_72h) & (sym_signals["pub_dt"] <= session_open)
            ]

            cnt_24h = len(sig_24h)
            cnt_72h = len(sig_72h)

            sent_24h = float(sig_24h["sentiment_score"].mean()) if cnt_24h > 0 else 0.0
            sent_72h = float(sig_72h["sentiment_score"].mean()) if cnt_72h > 0 else 0.0

            pos_24h = int((sig_24h["sentiment_score"] > 0.05).sum()) if cnt_24h > 0 else 0
            neg_24h = int((sig_24h["sentiment_score"] < -0.05).sum()) if cnt_24h > 0 else 0

            has_earnings = (
                1 if (cnt_24h > 0 and (sig_24h["event_category"] == "EARNINGS").any()) else 0
            )
            has_dividend = (
                1 if (cnt_24h > 0 and (sig_24h["event_category"] == "DIVIDEND").any()) else 0
            )
            has_discovery = (
                1 if (cnt_24h > 0 and (sig_24h["event_category"] == "DISCOVERY").any()) else 0
            )

            rows.append(
                {
                    "symbol": clean_sym,
                    "session_date": dt.isoformat(),
                    "sentiment_24h": round(sent_24h, 4),
                    "sentiment_72h": round(sent_72h, 4),
                    "news_count_24h": cnt_24h,
                    "news_count_72h": cnt_72h,
                    "positive_count_24h": pos_24h,
                    "negative_count_24h": neg_24h,
                    "has_earnings_announcement_today": has_earnings,
                    "has_dividend_announcement_today": has_dividend,
                    "has_discovery_announcement_today": has_discovery,
                }
            )

        return pd.DataFrame(rows)

    def align_and_persist(
        self,
        target_symbols: Optional[list[str]] = None,
        dry_run: bool = False,
    ) -> AlignmentSummary:
        """Align all processed news signals and build point-in-time daily session features.

        Args:
            target_symbols: Optional subset of symbols to compute daily features for.
            dry_run: If True, executes calculations without persisting to disk.

        Returns:
            AlignmentSummary with execution statistics.
        """
        summary = AlignmentSummary()

        if not self.signals_file.exists():
            logger.warning(
                f"No signals found at {self.signals_file}. Run 'psx news process' first."
            )
            return summary

        signals_df = read_parquet(self.signals_file)
        if signals_df.empty:
            logger.warning("Processed signals parquet is empty.")
            return summary

        # 1. Temporal Session Alignment
        aligned_df = self.align_dataframe(signals_df)
        summary.total_signals_aligned = len(aligned_df)

        for s_type in aligned_df["session_type"].dropna():
            summary.session_type_distribution[str(s_type)] = (
                summary.session_type_distribution.get(str(s_type), 0) + 1
            )

        summary.intraday_signals_count = int(aligned_df["is_intraday"].sum())
        summary.after_hours_signals_count = (
            summary.total_signals_aligned - summary.intraday_signals_count
        )

        sorted_dates = sorted(aligned_df["session_target_date"].dropna().unique())
        summary.trading_dates_covered = len(sorted_dates)
        if sorted_dates:
            summary.earliest_session = str(sorted_dates[0])
            summary.latest_session = str(sorted_dates[-1])

        # 2. Daily Feature Generation
        # Determine universe of symbols
        if target_symbols:
            symbols = [s.strip().upper() for s in target_symbols]
        else:
            # Check existing processed prices or matched signals
            symbols_set: set[str] = set()
            for prim in aligned_df["primary_symbol"].dropna():
                if prim and prim.upper() not in {"MACRO", "GENERAL", "OTHER"}:
                    symbols_set.add(prim.upper())

            # Also inspect prices dir for universe tickers
            prices_dir = self.storage_paths["processed_prices"]
            if prices_dir.exists():
                for p in prices_dir.glob("*.parquet"):
                    symbols_set.add(p.stem.upper())

            symbols = (
                sorted(symbols_set)
                if symbols_set
                else ["OGDC", "PPL", "LUCK", "SYS", "UBL", "MCB", "HUBC"]
            )

        summary.symbols_covered = symbols

        # Determine trading dates interval for daily features
        if sorted_dates:
            earliest_dt = datetime.date.fromisoformat(sorted_dates[0])
            latest_dt = datetime.date.fromisoformat(sorted_dates[-1])
            # Extend 5 trading days beyond latest signal for forward prediction
            future_dt = latest_dt
            for _ in range(5):
                future_dt = self.calendar.next_trading_day(future_dt)
            all_trading_days = self.calendar.get_trading_days_between(earliest_dt, future_dt)
        else:
            today = datetime.date.today()
            all_trading_days = [today]

        all_features_dfs = []
        for sym in symbols:
            feat_df = self.compute_daily_session_features(
                signals_df=aligned_df,
                symbol=sym,
                trading_dates=all_trading_days,
            )
            all_features_dfs.append(feat_df)

            # Persist per-symbol news features
            if not dry_run:
                sym_path = self.features_dir / f"{sym}.parquet"
                write_parquet_atomic(feat_df, sym_path)

        if all_features_dfs:
            combined_features = pd.concat(all_features_dfs, ignore_index=True)
            summary.daily_feature_records = len(combined_features)
        else:
            combined_features = pd.DataFrame()

        # 3. Persist Datasets
        if not dry_run:
            write_parquet_atomic(aligned_df, self.aligned_file)
            logger.info(f"Persisted {len(aligned_df)} aligned signals to {self.aligned_file}")

            combined_feat_file = self.features_dir / "daily_news_features.parquet"
            if not combined_features.empty:
                write_parquet_atomic(combined_features, combined_feat_file)
                logger.info(
                    f"Persisted {len(combined_features)} daily news feature rows "
                    f"across {len(symbols)} symbols to {combined_feat_file}"
                )

        return summary
