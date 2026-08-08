"""orrery-lab --- celestial mechanics, statistics and machine learning on real sky data.

An *orrery* is a mechanical model of the solar system, the kind with brass arms
and hand-cranked gears. This package is the computational equivalent: it turns
published orbital elements into real 3D positions, so that everything built on
top --- the browser scene, the statistics, the machine-learning models --- rests
on physics rather than on decoration.

Quick start::

    from orrery import body_state, julian_date

    jd = julian_date(2026, 7, 26)
    mars = body_state("mars", jd)
    print(mars.distance_au, mars.speed_km_per_s)

Layout
------
``constants``           Physical constants, masses, radii, frame definitions.
``timescales``          Calendar dates <-> Julian Dates.
``kepler``              Kepler's equation and anomaly conversions.
``elements``            Orbital elements of the planets and their secular drift.
``ephemeris``           Elements -> 3D state vectors, orbit sampling, frame rotations.
``nbody``               Gravitational N-body integrators and conservation diagnostics.
``initial_conditions``  Ephemeris -> barycentric starting states for the integrator.
"""

from __future__ import annotations

from .constants import AU_KM, GM_SUN, J2000_JD
from .elements import PLANET_NAMES, PLANETS, OrbitalElements, elements_at
from .ephemeris import (
    StateVector,
    body_state,
    ecliptic_to_equatorial,
    orbit_path,
    state_from_elements,
    system_state,
)
from .initial_conditions import SystemState, solar_system_state, two_body_state
from .kepler import (
    eccentric_from_true_anomaly,
    mean_motion,
    orbital_period,
    radius_from_eccentric,
    solve_kepler,
    true_anomaly_from_eccentric,
)
from .nbody import (
    INTEGRATORS,
    Trajectory,
    accelerations,
    integrate,
    total_angular_momentum,
    total_energy,
    total_linear_momentum,
)
from .timescales import (
    datetime_from_julian_date,
    julian_centuries_since_j2000,
    julian_date,
    julian_date_from_datetime,
)

__version__ = "0.1.0"

__all__ = [
    # constants
    "AU_KM",
    "GM_SUN",
    "J2000_JD",
    # time
    "julian_date",
    "julian_date_from_datetime",
    "datetime_from_julian_date",
    "julian_centuries_since_j2000",
    # kepler
    "solve_kepler",
    "true_anomaly_from_eccentric",
    "eccentric_from_true_anomaly",
    "radius_from_eccentric",
    "mean_motion",
    "orbital_period",
    # elements
    "OrbitalElements",
    "PLANETS",
    "PLANET_NAMES",
    "elements_at",
    # ephemeris
    "StateVector",
    "state_from_elements",
    "body_state",
    "system_state",
    "orbit_path",
    "ecliptic_to_equatorial",
    # n-body
    "accelerations",
    "integrate",
    "total_energy",
    "total_angular_momentum",
    "total_linear_momentum",
    "Trajectory",
    "INTEGRATORS",
    "SystemState",
    "solar_system_state",
    "two_body_state",
    "__version__",
]
