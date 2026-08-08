"""Conversions between calendar dates and Julian Dates.

Astronomical algorithms count time as a single continuous number of days --- the
Julian Date --- because calendars are full of discontinuities (leap years, the
Gregorian reform, month lengths) that make date arithmetic error-prone.

The conversions here use the Fliegel & Van Flandern algorithm, valid for any
date in the Gregorian calendar.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from .constants import J2000_JD, JULIAN_CENTURY_DAYS

__all__ = [
    "julian_date",
    "julian_date_from_datetime",
    "datetime_from_julian_date",
    "julian_centuries_since_j2000",
    "days_since_j2000",
]


def julian_date(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: float = 0.0,
) -> float:
    """Return the Julian Date of a Gregorian calendar instant (UTC).

    Args:
        year: Astronomical year numbering (1 BC is year 0, 2 BC is year -1).
        month: 1-12.
        day: 1-31.
        hour: 0-23.
        minute: 0-59.
        second: 0-59.999...

    Returns:
        Julian Date as a float.

    Example:
        >>> julian_date(2000, 1, 1, 12)
        2451545.0
    """
    # Shift the year so that a "year" starts in March. This makes the leap day
    # the last day of the year, which removes the special case from the
    # day-count arithmetic below.
    if month <= 2:
        year -= 1
        month += 12

    a = math.floor(year / 100)
    # Gregorian calendar correction (skipped century leap years).
    b = 2 - a + math.floor(a / 4)

    jd_midnight = (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + day
        + b
        - 1524.5
    )
    day_fraction = (hour + minute / 60.0 + second / 3600.0) / 24.0
    return jd_midnight + day_fraction


def julian_date_from_datetime(moment: datetime) -> float:
    """Return the Julian Date of a :class:`datetime`.

    Naive datetimes are assumed to be UTC; aware datetimes are converted.
    """
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc)
    return julian_date(
        moment.year,
        moment.month,
        moment.day,
        moment.hour,
        moment.minute,
        moment.second + moment.microsecond / 1e6,
    )


def datetime_from_julian_date(jd: float) -> datetime:
    """Inverse of :func:`julian_date`, returning a UTC-aware :class:`datetime`."""
    jd_shifted = jd + 0.5
    integer_part = math.floor(jd_shifted)
    fraction = jd_shifted - integer_part

    if integer_part >= 2_299_161:  # Gregorian calendar
        alpha = math.floor((integer_part - 1_867_216.25) / 36_524.25)
        integer_part += 1 + alpha - math.floor(alpha / 4)

    b = integer_part + 1524
    c = math.floor((b - 122.1) / 365.25)
    d = math.floor(365.25 * c)
    e = math.floor((b - d) / 30.6001)

    day = b - d - math.floor(30.6001 * e)
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715

    seconds_total = fraction * 86400.0
    hour, remainder = divmod(seconds_total, 3600.0)
    minute, second = divmod(remainder, 60.0)
    microsecond = round((second - int(second)) * 1e6)

    # Rounding the microseconds can push us to exactly 1e6; normalise.
    if microsecond >= 1_000_000:
        microsecond -= 1_000_000
        second += 1

    return datetime(
        year,
        month,
        int(day),
        int(hour),
        int(minute),
        int(second),
        microsecond,
        tzinfo=timezone.utc,
    )


def days_since_j2000(jd: float | np.ndarray) -> float | np.ndarray:
    """Days elapsed since the J2000.0 epoch (negative before it)."""
    return jd - J2000_JD


def julian_centuries_since_j2000(jd: float | np.ndarray) -> float | np.ndarray:
    """Julian centuries elapsed since J2000.0.

    This is the time variable ``T`` used by JPL's orbital element rate tables.
    """
    return (jd - J2000_JD) / JULIAN_CENTURY_DAYS
