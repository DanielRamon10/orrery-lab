"""Print the state of the solar system for a given date.

Usage::

    python scripts/solar_system_report.py                # today
    python scripts/solar_system_report.py 2026-12-25
    python scripts/solar_system_report.py 1969-07-20

This is the human-readable face of phase 1: everything printed here is computed
from the orbital elements, not looked up in a table.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from orrery import PLANET_NAMES, PLANETS, body_state, julian_date_from_datetime
from orrery.constants import AU_KM


def parse_date(text: str | None) -> datetime:
    """Accept ``YYYY-MM-DD`` (optionally with ``THH:MM``), defaulting to now."""
    if text is None:
        return datetime.now(timezone.utc)
    for pattern in ("%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise SystemExit(f"could not parse date {text!r}; expected YYYY-MM-DD")


def ecliptic_longitude_deg(position: np.ndarray) -> float:
    """Angle in the ecliptic plane, measured from the vernal equinox."""
    return float(np.degrees(np.arctan2(position[1], position[0])) % 360.0)


def ecliptic_latitude_deg(position: np.ndarray) -> float:
    """Angle above or below the ecliptic plane."""
    in_plane = float(np.hypot(position[0], position[1]))
    return float(np.degrees(np.arctan2(position[2], in_plane)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", nargs="?", help="UTC date, YYYY-MM-DD (default: now)")
    parser.add_argument(
        "--include-pluto", action="store_true", help="add Pluto to the table"
    )
    args = parser.parse_args()

    moment = parse_date(args.date)
    jd = julian_date_from_datetime(moment)

    bodies = list(PLANET_NAMES) + (["pluto"] if args.include_pluto else [])

    print()
    print(f"Solar system state  |  {moment:%Y-%m-%d %H:%M} UTC  |  JD {jd:.5f}")
    print("=" * 104)
    header = (
        f"{'Body':<9}{'r (AU)':>10}{'r (10^6 km)':>13}{'speed':>10}"
        f"{'lon':>9}{'lat':>8}{'a (AU)':>10}{'e':>8}{'incl':>8}{'period':>14}"
    )
    print(header)
    print(f"{'':<9}{'':>10}{'':>13}{'km/s':>10}{'deg':>9}{'deg':>8}{'':>10}{'':>8}{'deg':>8}{'years':>14}")
    print("-" * 104)

    for name in bodies:
        elements = PLANETS[name]
        state = body_state(name, jd)
        position = state.position

        distance_au = float(state.distance_au)
        print(
            f"{elements.name:<9}"
            f"{distance_au:>10.4f}"
            f"{distance_au * AU_KM / 1e6:>13.2f}"
            f"{float(state.speed_km_per_s):>10.2f}"
            f"{ecliptic_longitude_deg(position):>9.2f}"
            f"{ecliptic_latitude_deg(position):>8.2f}"
            f"{elements.semi_major_axis_au:>10.4f}"
            f"{elements.eccentricity:>8.4f}"
            f"{elements.inclination_deg:>8.2f}"
            f"{elements.period_days / 365.25:>14.3f}"
        )

    print("-" * 104)
    print()

    # A couple of derived facts that are easy to check against the news.
    earth = body_state("earth", jd)
    mars = body_state("mars", jd)
    separation = float(np.linalg.norm(mars.position - earth.position))
    light_minutes = separation * AU_KM / 299_792.458 / 60.0

    print(f"Earth-Mars separation : {separation:.4f} AU  ({light_minutes:.1f} light-minutes)")
    print(f"Earth orbital energy  : {float(earth.specific_orbital_energy()):.6e} AU^2/day^2")
    print(
        "Earth angular momentum: "
        f"{float(np.linalg.norm(earth.specific_angular_momentum)):.9f} AU^2/day"
    )
    print()


if __name__ == "__main__":
    main()
