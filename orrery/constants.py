"""Physical and astronomical constants.

Values follow IAU 2015 resolutions and the JPL DE440 planetary ephemeris.
Every constant states its unit, because mixing unit systems is the single most
common source of silent errors in celestial mechanics code.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

#: Julian Date of the J2000.0 epoch (2000-01-01 12:00:00 TT).
J2000_JD = 2451545.0

#: Days in a Julian century, the unit used by JPL's element rate tables.
JULIAN_CENTURY_DAYS = 36525.0

#: Days in a Julian year.
JULIAN_YEAR_DAYS = 365.25

SECONDS_PER_DAY = 86400.0

# ---------------------------------------------------------------------------
# Length
# ---------------------------------------------------------------------------

#: Astronomical unit in kilometres (IAU 2012 exact definition).
AU_KM = 149_597_870.7
AU_M = AU_KM * 1000.0

# ---------------------------------------------------------------------------
# Gravity
# ---------------------------------------------------------------------------

#: Gaussian gravitational constant, in radians per day.
#:
#: This is the historical constant ``k`` that defines the AU-day-solar mass
#: system. Its square *is* the heliocentric ``GM``, which is why the Sun's mass
#: never appears explicitly in classical planetary theory.
GAUSSIAN_K = 0.01720209895

#: Heliocentric gravitational parameter GM_sun, in AU^3 / day^2.
#:
#: With this value, the mean motion of an orbit is ``n = sqrt(GM_SUN / a**3)``
#: for ``a`` in AU and ``n`` in radians per day.
GM_SUN = GAUSSIAN_K**2  # 2.9591220828559115e-04

#: Heliocentric gravitational parameter in km^3 / s^2 (DE440).
GM_SUN_KM3_S2 = 1.32712440041279419e11

#: Newtonian constant of gravitation, m^3 kg^-1 s^-2 (CODATA 2018).
G_SI = 6.67430e-11

#: Solar mass in kilograms, derived from GM_sun and G.
M_SUN_KG = GM_SUN_KM3_S2 * 1e9 / G_SI

# ---------------------------------------------------------------------------
# Reference frames
# ---------------------------------------------------------------------------

#: Mean obliquity of the ecliptic at J2000.0, in degrees.
#:
#: The angle between the ecliptic plane (Earth's orbital plane, natural for
#: solar-system work) and the equatorial plane (natural for star catalogues).
#: Phase 6 needs this to place Gaia stars in the same frame as the planets.
OBLIQUITY_J2000_DEG = 23.439291111111111
OBLIQUITY_J2000_RAD = math.radians(OBLIQUITY_J2000_DEG)

# ---------------------------------------------------------------------------
# Body masses
# ---------------------------------------------------------------------------

#: Sun-to-body mass ratios (dimensionless), from the DE440 header.
#:
#: Terrestrial planets are quoted as planet + satellites, matching the
#: barycentric convention used by the approximate element tables.
SUN_TO_BODY_MASS_RATIO: dict[str, float] = {
    "mercury": 6_023_657.9,
    "venus": 408_523.72,
    "earth": 328_900.5596,  # Earth-Moon system
    "mars": 3_098_703.6,
    "jupiter": 1047.348644,
    "saturn": 3497.9018,
    "uranus": 22_902.98,
    "neptune": 19_412.26,
    "pluto": 136_045_556.0,
}

#: Gravitational parameter GM of each body, in AU^3 / day^2.
GM_BODY: dict[str, float] = {
    name: GM_SUN / ratio for name, ratio in SUN_TO_BODY_MASS_RATIO.items()
}

# ---------------------------------------------------------------------------
# Physical sizes (used by the 3D renderer, not by the dynamics)
# ---------------------------------------------------------------------------

#: Mean/volumetric radius in kilometres (IAU 2015 nominal values).
MEAN_RADIUS_KM: dict[str, float] = {
    "sun": 695_700.0,
    "mercury": 2439.7,
    "venus": 6051.8,
    "earth": 6371.0,
    "mars": 3389.5,
    "jupiter": 69_911.0,
    "saturn": 58_232.0,
    "uranus": 25_362.0,
    "neptune": 24_622.0,
    "pluto": 1188.3,
}

#: Sidereal rotation period in days; negative means retrograde rotation.
ROTATION_PERIOD_DAYS: dict[str, float] = {
    "sun": 25.38,
    "mercury": 58.646,
    "venus": -243.025,
    "earth": 0.99726968,
    "mars": 1.02595676,
    "jupiter": 0.41354,
    "saturn": 0.44401,
    "uranus": -0.71833,
    "neptune": 0.67125,
    "pluto": -6.3872,
}

#: Obliquity of each body's rotation axis to its orbital plane, in degrees.
AXIAL_TILT_DEG: dict[str, float] = {
    "sun": 7.25,
    "mercury": 0.034,
    "venus": 177.36,
    "earth": 23.4393,
    "mars": 25.19,
    "jupiter": 3.13,
    "saturn": 26.73,
    "uranus": 97.77,
    "neptune": 28.32,
    "pluto": 122.53,
}
