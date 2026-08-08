"""Tests for calendar <-> Julian Date conversions.

Reference Julian Dates below are standard published values; the leap-year and
Gregorian-reform cases are the ones where naive implementations break.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orrery.constants import J2000_JD, JULIAN_CENTURY_DAYS
from orrery.timescales import (
    datetime_from_julian_date,
    days_since_j2000,
    julian_centuries_since_j2000,
    julian_date,
    julian_date_from_datetime,
)

# (year, month, day, hour, expected JD) --- standard reference epochs.
KNOWN_EPOCHS = [
    (2000, 1, 1, 12, 2451545.0),  # J2000.0 itself
    (1999, 1, 1, 0, 2451179.5),
    (1987, 1, 27, 0, 2446822.5),
    (1987, 6, 19, 12, 2446966.0),
    (1988, 1, 27, 0, 2447187.5),
    (1988, 6, 19, 12, 2447332.0),  # 1988 is a leap year
    (1900, 1, 1, 0, 2415020.5),  # 1900 is *not* a leap year (century rule)
    (2000, 3, 1, 0, 2451604.5),  # 2000 *is* a leap year (400-year rule)
    (1970, 1, 1, 0, 2440587.5),  # Unix epoch
    (2026, 7, 26, 0, 2461247.5),
]


class TestJulianDate:
    @pytest.mark.parametrize(("year", "month", "day", "hour", "expected"), KNOWN_EPOCHS)
    def test_known_epochs(self, year, month, day, hour, expected):
        assert julian_date(year, month, day, hour) == pytest.approx(expected, abs=1e-9)

    def test_half_day_offset(self):
        """Julian Days start at noon, so midnight always lands on a .5 fraction."""
        assert julian_date(2026, 7, 26, 0) % 1.0 == pytest.approx(0.5)
        assert julian_date(2026, 7, 26, 12) % 1.0 == pytest.approx(0.0)

    def test_one_day_apart(self):
        assert julian_date(2026, 7, 27) - julian_date(2026, 7, 26) == pytest.approx(1.0)

    def test_sub_day_resolution(self):
        base = julian_date(2026, 7, 26, 0, 0, 0)
        one_hour = julian_date(2026, 7, 26, 1, 0, 0)
        one_minute = julian_date(2026, 7, 26, 0, 1, 0)

        assert one_hour - base == pytest.approx(1.0 / 24.0)
        assert one_minute - base == pytest.approx(1.0 / 1440.0)


class TestRoundTrip:
    @pytest.mark.parametrize(
        "moment",
        [
            datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            datetime(1969, 7, 20, 20, 17, 40, tzinfo=timezone.utc),  # Apollo 11 landing
            datetime(2026, 7, 26, 15, 42, 13, tzinfo=timezone.utc),
            datetime(2100, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
            datetime(1800, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        ],
    )
    def test_datetime_round_trip(self, moment):
        recovered = datetime_from_julian_date(julian_date_from_datetime(moment))
        assert abs((recovered - moment).total_seconds()) < 1e-3

    def test_naive_datetime_treated_as_utc(self):
        naive = datetime(2026, 7, 26, 12, 0, 0)
        aware = datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone.utc)
        assert julian_date_from_datetime(naive) == pytest.approx(
            julian_date_from_datetime(aware)
        )

    def test_timezone_is_converted_not_ignored(self):
        """A -03:00 local time is three hours *later* in UTC."""
        local = datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone(timedelta(hours=-3)))
        as_utc = datetime(2026, 7, 26, 15, 0, 0, tzinfo=timezone.utc)
        assert julian_date_from_datetime(local) == pytest.approx(
            julian_date_from_datetime(as_utc)
        )


class TestEpochOffsets:
    def test_j2000_is_the_origin(self):
        assert days_since_j2000(J2000_JD) == 0.0
        assert julian_centuries_since_j2000(J2000_JD) == 0.0

    def test_one_century(self):
        assert julian_centuries_since_j2000(J2000_JD + JULIAN_CENTURY_DAYS) == pytest.approx(1.0)

    def test_negative_before_j2000(self):
        assert julian_centuries_since_j2000(julian_date(1900, 1, 1)) < 0.0
