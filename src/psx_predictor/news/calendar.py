"""Pakistan Stock Exchange (PSX) market session calendar and schedule rules."""

import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

# PSX official market operating timezone
PKT_TZ = ZoneInfo("Asia/Karachi")

# Standard Trading Hours (Pakistan Standard Time)
REGULAR_OPEN = datetime.time(9, 15)
REGULAR_CLOSE = datetime.time(15, 30)

FRIDAY_S1_OPEN = datetime.time(9, 15)
FRIDAY_S1_CLOSE = datetime.time(12, 0)
FRIDAY_BREAK_OPEN = datetime.time(12, 0)
FRIDAY_BREAK_CLOSE = datetime.time(14, 30)
FRIDAY_S2_OPEN = datetime.time(14, 30)
FRIDAY_S2_CLOSE = datetime.time(16, 30)

# Standard Fixed Annual National / PSX Holidays
FIXED_HOLIDAYS: set[tuple[int, int]] = {
    (2, 5),  # Kashmir Solidarity Day (Feb 5)
    (3, 23),  # Pakistan Day (Mar 23)
    (5, 1),  # Labour Day (May 1)
    (8, 14),  # Independence Day (Aug 14)
    (11, 9),  # Iqbal Day (Nov 9)
    (12, 25),  # Quaid-e-Azam Day / Christmas (Dec 25)
}

# Known Lunar / Declared Exchange Holidays by Year (YYYY-MM-DD)
DECLARED_HOLIDAYS: set[datetime.date] = {
    # 2024
    datetime.date(2024, 4, 10),
    datetime.date(2024, 4, 11),
    datetime.date(2024, 4, 12),  # Eid-ul-Fitr
    datetime.date(2024, 6, 17),
    datetime.date(2024, 6, 18),
    datetime.date(2024, 6, 19),  # Eid-ul-Azha
    datetime.date(2024, 7, 16),
    datetime.date(2024, 7, 17),  # Ashura
    datetime.date(2024, 9, 16),  # Eid Milad-un-Nabi
    # 2025
    datetime.date(2025, 3, 31),
    datetime.date(2025, 4, 1),
    datetime.date(2025, 4, 2),  # Eid-ul-Fitr
    datetime.date(2025, 6, 6),
    datetime.date(2025, 6, 7),
    datetime.date(2025, 6, 8),  # Eid-ul-Azha
    datetime.date(2025, 7, 5),
    datetime.date(2025, 7, 6),  # Ashura
    datetime.date(2025, 9, 5),  # Eid Milad-un-Nabi
    # 2026
    datetime.date(2026, 3, 20),
    datetime.date(2026, 3, 21),
    datetime.date(2026, 3, 22),  # Eid-ul-Fitr
    datetime.date(2026, 5, 27),
    datetime.date(2026, 5, 28),
    datetime.date(2026, 5, 29),  # Eid-ul-Azha
    datetime.date(2026, 6, 25),
    datetime.date(2026, 6, 26),  # Ashura
    datetime.date(2026, 8, 25),  # Eid Milad-un-Nabi
}


class PSXMarketCalendar:
    """PSX market session calendar managing trading hours, Friday split schedules, and holidays."""

    def __init__(self, custom_holidays: Optional[set[datetime.date]] = None) -> None:
        """Initialize the PSX market calendar.

        Args:
            custom_holidays: Optional set of extra holiday dates to merge.
        """
        self.tz = PKT_TZ
        self._holidays: set[datetime.date] = set(DECLARED_HOLIDAYS)
        if custom_holidays:
            self._holidays.update(custom_holidays)

    def add_holiday(self, holiday_date: datetime.date) -> None:
        """Register a new holiday dynamically."""
        self._holidays.add(holiday_date)

    def remove_holiday(self, holiday_date: datetime.date) -> None:
        """Remove a holiday date if needed for backtesting or testing."""
        self._holidays.discard(holiday_date)

    def is_holiday(self, dt: datetime.date) -> bool:
        """Check if a date is a registered PSX holiday.

        Args:
            dt: Calendar date.

        Returns:
            True if date is a fixed national or declared Islamic/PSX holiday.
        """
        if (dt.month, dt.day) in FIXED_HOLIDAYS:
            return True
        return dt in self._holidays

    def is_trading_day(self, dt: datetime.date) -> bool:
        """Check if the given date is an active PSX trading day.

        Active trading days are Monday to Friday excluding official holidays.

        Args:
            dt: Calendar date.

        Returns:
            True if PSX is open for trading on this date.
        """
        # Saturday = 5, Sunday = 6
        if dt.weekday() >= 5:
            return False
        return not self.is_holiday(dt)

    def next_trading_day(self, dt: datetime.date) -> datetime.date:
        """Find the earliest trading day strictly after the given date.

        Args:
            dt: Reference date.

        Returns:
            Next active trading day.
        """
        current = dt + datetime.timedelta(days=1)
        while not self.is_trading_day(current):
            current += datetime.timedelta(days=1)
        return current

    def previous_trading_day(self, dt: datetime.date) -> datetime.date:
        """Find the latest trading day strictly before the given date.

        Args:
            dt: Reference date.

        Returns:
            Previous active trading day.
        """
        current = dt - datetime.timedelta(days=1)
        while not self.is_trading_day(current):
            current -= datetime.timedelta(days=1)
        return current

    def get_trading_days_between(
        self, start_date: datetime.date, end_date: datetime.date
    ) -> list[datetime.date]:
        """Get list of active trading days within a closed interval [start_date, end_date]."""
        days: list[datetime.date] = []
        cur = start_date
        while cur <= end_date:
            if self.is_trading_day(cur):
                days.append(cur)
            cur += datetime.timedelta(days=1)
        return days

    def get_sessions_for_date(self, dt: datetime.date) -> list[dict[str, Any]]:
        """Retrieve operational trading session windows for a given date.

        Args:
            dt: Calendar date.

        Returns:
            List of session descriptions with 'session_type', 'open', and 'close' times.
        """
        if not self.is_trading_day(dt):
            return []

        # Friday split session
        if dt.weekday() == 4:
            return [
                {
                    "session_type": "FRIDAY_SESSION_1",
                    "open": FRIDAY_S1_OPEN,
                    "close": FRIDAY_S1_CLOSE,
                },
                {
                    "session_type": "FRIDAY_SESSION_2",
                    "open": FRIDAY_S2_OPEN,
                    "close": FRIDAY_S2_CLOSE,
                },
            ]

        # Mon - Thu regular session
        return [
            {
                "session_type": "REGULAR",
                "open": REGULAR_OPEN,
                "close": REGULAR_CLOSE,
            }
        ]
