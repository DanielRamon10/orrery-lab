"""Keplerian orbital elements of the major planets, and how they drift.

Six numbers fully describe an orbit:

===============  ============================================================
``a``            Semi-major axis --- the size of the ellipse.
``e``            Eccentricity --- how squashed it is (0 = circle).
``i``            Inclination --- tilt away from the ecliptic plane.
``Omega``        Longitude of the ascending node --- where it crosses upward.
``omega_bar``    Longitude of perihelion --- where the closest approach is.
``L``            Mean longitude --- where the planet is *along* the orbit.
===============  ============================================================

The first five are the shape and orientation of the ellipse; only the last one
depends on time in the two-body problem.

In reality the planets tug on each other, so even the "fixed" five slowly drift.
The table below therefore stores each element *and its rate of change per Julian
century*, following JPL's "Keplerian Elements for Approximate Positions of the
Major Planets" (E. M. Standish, Solar System Dynamics Group). The linear-rate
form is accurate to roughly an arcminute over 1800-2050 --- far better than the
eye can tell in a 3D scene, and cheap enough to run every animation frame.

Note that Earth's entry is really the **Earth-Moon barycentre**, which is what
the source table tabulates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import GM_SUN
from .kepler import orbital_period
from .timescales import julian_centuries_since_j2000

__all__ = ["OrbitalElements", "PLANETS", "PLANET_NAMES", "elements_at"]


@dataclass(frozen=True)
class OrbitalElements:
    """Keplerian elements at J2000.0 plus their linear drift per century.

    Angles are stored in **degrees**, distances in **astronomical units**, to
    match the published source table. Conversion to radians happens once, inside
    :meth:`at`, so the raw numbers stay auditable against the JPL document.
    """

    name: str
    semi_major_axis_au: float
    eccentricity: float
    inclination_deg: float
    mean_longitude_deg: float
    longitude_of_perihelion_deg: float
    longitude_of_ascending_node_deg: float

    # Rates of change, per Julian century.
    semi_major_axis_rate: float = 0.0
    eccentricity_rate: float = 0.0
    inclination_rate: float = 0.0
    mean_longitude_rate: float = 0.0
    longitude_of_perihelion_rate: float = 0.0
    longitude_of_ascending_node_rate: float = 0.0

    def at(self, jd: float | np.ndarray) -> dict[str, np.ndarray]:
        """Propagate the elements to a Julian Date (or an array of them).

        Returns:
            Dict of arrays with keys ``a`` (AU), ``e`` (dimensionless), and
            ``i``, ``Omega``, ``omega``, ``M`` in **radians**, where

            * ``omega`` is the *argument* of perihelion, ``omega_bar - Omega``
            * ``M`` is the mean anomaly, ``L - omega_bar``

            Those two conversions are the only subtlety in using this table:
            the published columns are longitudes measured from the reference
            direction, while the geometry in :mod:`orrery.ephemeris` needs the
            argument of perihelion and the mean anomaly.
        """
        centuries = np.asarray(julian_centuries_since_j2000(jd), dtype=float)

        a = self.semi_major_axis_au + self.semi_major_axis_rate * centuries
        e = self.eccentricity + self.eccentricity_rate * centuries
        inclination = self.inclination_deg + self.inclination_rate * centuries
        mean_longitude = self.mean_longitude_deg + self.mean_longitude_rate * centuries
        perihelion = (
            self.longitude_of_perihelion_deg + self.longitude_of_perihelion_rate * centuries
        )
        node = (
            self.longitude_of_ascending_node_deg
            + self.longitude_of_ascending_node_rate * centuries
        )

        return {
            "a": a,
            "e": e,
            "i": np.radians(inclination),
            "Omega": np.radians(node),
            "omega": np.radians(perihelion - node),
            "M": np.radians(mean_longitude - perihelion),
        }

    @property
    def period_days(self) -> float:
        """Orbital period at J2000.0, in days, from Kepler's third law."""
        return float(orbital_period(self.semi_major_axis_au, GM_SUN))

    @property
    def perihelion_au(self) -> float:
        """Closest approach to the Sun, ``a (1 - e)``."""
        return self.semi_major_axis_au * (1.0 - self.eccentricity)

    @property
    def aphelion_au(self) -> float:
        """Farthest recession from the Sun, ``a (1 + e)``."""
        return self.semi_major_axis_au * (1.0 + self.eccentricity)


# ---------------------------------------------------------------------------
# JPL approximate elements, epoch J2000.0, heliocentric ecliptic frame.
# Source: https://ssd.jpl.nasa.gov/planets/approx_pos.html (Table 1, 1800-2050)
#
# Column order in each pair of rows below matches the dataclass field order:
#   a, e, i, L, omega_bar, Omega   then the six rates.
# ---------------------------------------------------------------------------

PLANETS: dict[str, OrbitalElements] = {
    "mercury": OrbitalElements(
        name="Mercury",
        semi_major_axis_au=0.38709927,
        eccentricity=0.20563593,
        inclination_deg=7.00497902,
        mean_longitude_deg=252.25032350,
        longitude_of_perihelion_deg=77.45779628,
        longitude_of_ascending_node_deg=48.33076593,
        semi_major_axis_rate=0.00000037,
        eccentricity_rate=0.00001906,
        inclination_rate=-0.00594749,
        mean_longitude_rate=149472.67411175,
        longitude_of_perihelion_rate=0.16047689,
        longitude_of_ascending_node_rate=-0.12534081,
    ),
    "venus": OrbitalElements(
        name="Venus",
        semi_major_axis_au=0.72333566,
        eccentricity=0.00677672,
        inclination_deg=3.39467605,
        mean_longitude_deg=181.97909950,
        longitude_of_perihelion_deg=131.60246718,
        longitude_of_ascending_node_deg=76.67984255,
        semi_major_axis_rate=0.00000390,
        eccentricity_rate=-0.00004107,
        inclination_rate=-0.00078890,
        mean_longitude_rate=58517.81538729,
        longitude_of_perihelion_rate=0.00268329,
        longitude_of_ascending_node_rate=-0.27769418,
    ),
    "earth": OrbitalElements(
        name="Earth",  # strictly the Earth-Moon barycentre
        semi_major_axis_au=1.00000261,
        eccentricity=0.01671123,
        inclination_deg=-0.00001531,
        mean_longitude_deg=100.46457166,
        longitude_of_perihelion_deg=102.93768193,
        longitude_of_ascending_node_deg=0.0,
        semi_major_axis_rate=0.00000562,
        eccentricity_rate=-0.00004392,
        inclination_rate=-0.01294668,
        mean_longitude_rate=35999.37244981,
        longitude_of_perihelion_rate=0.32327364,
        longitude_of_ascending_node_rate=0.0,
    ),
    "mars": OrbitalElements(
        name="Mars",
        semi_major_axis_au=1.52371034,
        eccentricity=0.09339410,
        inclination_deg=1.84969142,
        mean_longitude_deg=-4.55343205,
        longitude_of_perihelion_deg=-23.94362959,
        longitude_of_ascending_node_deg=49.55953891,
        semi_major_axis_rate=0.00001847,
        eccentricity_rate=0.00007882,
        inclination_rate=-0.00813131,
        mean_longitude_rate=19140.30268499,
        longitude_of_perihelion_rate=0.44441088,
        longitude_of_ascending_node_rate=-0.29257343,
    ),
    "jupiter": OrbitalElements(
        name="Jupiter",
        semi_major_axis_au=5.20288700,
        eccentricity=0.04838624,
        inclination_deg=1.30439695,
        mean_longitude_deg=34.39644051,
        longitude_of_perihelion_deg=14.72847983,
        longitude_of_ascending_node_deg=100.47390909,
        semi_major_axis_rate=-0.00011607,
        eccentricity_rate=-0.00013253,
        inclination_rate=-0.00183714,
        mean_longitude_rate=3034.74612775,
        longitude_of_perihelion_rate=0.21252668,
        longitude_of_ascending_node_rate=0.20469106,
    ),
    "saturn": OrbitalElements(
        name="Saturn",
        semi_major_axis_au=9.53667594,
        eccentricity=0.05386179,
        inclination_deg=2.48599187,
        mean_longitude_deg=49.95424423,
        longitude_of_perihelion_deg=92.59887831,
        longitude_of_ascending_node_deg=113.66242448,
        semi_major_axis_rate=-0.00125060,
        eccentricity_rate=-0.00050991,
        inclination_rate=0.00193609,
        mean_longitude_rate=1222.49362201,
        longitude_of_perihelion_rate=-0.41897216,
        longitude_of_ascending_node_rate=-0.28867794,
    ),
    "uranus": OrbitalElements(
        name="Uranus",
        semi_major_axis_au=19.18916464,
        eccentricity=0.04725744,
        inclination_deg=0.77263783,
        mean_longitude_deg=313.23810451,
        longitude_of_perihelion_deg=170.95427630,
        longitude_of_ascending_node_deg=74.01692503,
        semi_major_axis_rate=-0.00196176,
        eccentricity_rate=-0.00004397,
        inclination_rate=-0.00242939,
        mean_longitude_rate=428.48202785,
        longitude_of_perihelion_rate=0.40805281,
        longitude_of_ascending_node_rate=0.04240589,
    ),
    "neptune": OrbitalElements(
        name="Neptune",
        semi_major_axis_au=30.06992276,
        eccentricity=0.00859048,
        inclination_deg=1.77004347,
        mean_longitude_deg=-55.12002969,
        longitude_of_perihelion_deg=44.96476227,
        longitude_of_ascending_node_deg=131.78422574,
        semi_major_axis_rate=0.00026291,
        eccentricity_rate=0.00005105,
        inclination_rate=0.00035372,
        mean_longitude_rate=218.45945325,
        longitude_of_perihelion_rate=-0.32241464,
        longitude_of_ascending_node_rate=-0.00508664,
    ),
    # Kept for completeness: no longer a planet, but the most interesting orbit
    # in the table (high eccentricity and inclination, 3:2 resonance with
    # Neptune) and therefore a useful stress test for the Kepler solver.
    "pluto": OrbitalElements(
        name="Pluto",
        semi_major_axis_au=39.48211675,
        eccentricity=0.24882730,
        inclination_deg=17.14001206,
        mean_longitude_deg=238.92903833,
        longitude_of_perihelion_deg=224.06891629,
        longitude_of_ascending_node_deg=110.30393684,
        semi_major_axis_rate=-0.00031596,
        eccentricity_rate=0.00005170,
        inclination_rate=0.00004818,
        mean_longitude_rate=145.20780515,
        longitude_of_perihelion_rate=-0.04062942,
        longitude_of_ascending_node_rate=-0.01183482,
    ),
}

#: The eight planets, in order of distance from the Sun.
PLANET_NAMES: tuple[str, ...] = (
    "mercury",
    "venus",
    "earth",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
)


def elements_at(body: str, jd: float | np.ndarray) -> dict[str, np.ndarray]:
    """Propagated elements for a named body.

    Args:
        body: Lower-case body key, e.g. ``"mars"``.
        jd: Julian Date, scalar or array.

    Raises:
        KeyError: If the body is not in :data:`PLANETS`.
    """
    key = body.lower()
    if key not in PLANETS:
        raise KeyError(f"unknown body {body!r}; known bodies: {sorted(PLANETS)}")
    return PLANETS[key].at(jd)
