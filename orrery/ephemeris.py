"""From orbital elements to 3D positions and velocities.

Pipeline
--------
1. Propagate the elements to the requested date (:mod:`orrery.elements`).
2. Solve Kepler's equation for the eccentric anomaly (:mod:`orrery.kepler`).
3. Place the body on its ellipse in the **perifocal frame** --- a flat 2D frame
   whose x-axis points at perihelion and whose origin is the Sun::

       x' = a (cos E - e)
       y' = a sqrt(1 - e^2) sin E

4. Rotate that flat orbit into the shared 3D ecliptic frame with three
   rotations, ``R = Rz(Omega) . Rx(i) . Rz(omega)``:

   * ``Rz(omega)`` spins the ellipse within its own plane so perihelion lands
     in the right direction,
   * ``Rx(i)`` tips the plane by the inclination,
   * ``Rz(Omega)`` swings the tipped plane around to its ascending node.

Because the same matrix rotates velocity as well as position, differentiating
the perifocal coordinates with respect to time is enough to get full state
vectors --- which is exactly what the N-body integrator in phase 3 needs as
initial conditions.

Units: positions in AU, velocities in AU/day, angles in radians.
"""

from __future__ import annotations

import numpy as np

from .constants import GM_SUN, OBLIQUITY_J2000_RAD
from .elements import PLANET_NAMES, PLANETS
from .kepler import mean_motion, solve_kepler

__all__ = [
    "StateVector",
    "state_from_elements",
    "body_state",
    "system_state",
    "orbit_path",
    "ecliptic_to_equatorial",
]


class StateVector:
    """Position and velocity of a body, with a few derived quantities.

    Attributes:
        position: Shape ``(..., 3)`` array in AU, heliocentric ecliptic J2000.
        velocity: Shape ``(..., 3)`` array in AU/day, same frame.
    """

    __slots__ = ("position", "velocity")

    def __init__(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)

    @property
    def distance_au(self) -> np.ndarray:
        """Heliocentric distance, ``|r|``."""
        return np.linalg.norm(self.position, axis=-1)

    @property
    def speed_au_per_day(self) -> np.ndarray:
        """Orbital speed, ``|v|``."""
        return np.linalg.norm(self.velocity, axis=-1)

    @property
    def speed_km_per_s(self) -> np.ndarray:
        """Orbital speed in km/s, the unit textbooks quote."""
        from .constants import AU_KM, SECONDS_PER_DAY

        return self.speed_au_per_day * AU_KM / SECONDS_PER_DAY

    @property
    def specific_angular_momentum(self) -> np.ndarray:
        """``h = r x v``.

        Conserved in any central force field, so a constant ``h`` along an orbit
        is a direct numerical check of Kepler's second law.
        """
        return np.cross(self.position, self.velocity)

    def specific_orbital_energy(self, gm: float = GM_SUN) -> np.ndarray:
        """``E = v^2/2 - GM/r``.

        Negative for a bound (elliptical) orbit. Also conserved, which makes it
        the standard accuracy diagnostic for an N-body integrator.
        """
        return 0.5 * self.speed_au_per_day**2 - gm / self.distance_au

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"StateVector(position={np.array2string(self.position, precision=6)}, "
            f"velocity={np.array2string(self.velocity, precision=9)})"
        )


def _perifocal_to_ecliptic_matrix(
    inclination: np.ndarray,
    node: np.ndarray,
    argument_of_perihelion: np.ndarray,
) -> np.ndarray:
    """Build ``Rz(Omega) . Rx(i) . Rz(omega)`` for arrays of angles.

    Returns:
        Shape ``(..., 3, 3)`` rotation matrices.
    """
    cos_i, sin_i = np.cos(inclination), np.sin(inclination)
    cos_n, sin_n = np.cos(node), np.sin(node)
    cos_w, sin_w = np.cos(argument_of_perihelion), np.sin(argument_of_perihelion)

    # Written out explicitly rather than as three matrix products: it is the form
    # published by JPL, so it can be checked line by line against the source, and
    # it avoids allocating three intermediate arrays per body per frame.
    matrix = np.empty(np.broadcast(cos_i, cos_n, cos_w).shape + (3, 3), dtype=float)

    matrix[..., 0, 0] = cos_w * cos_n - sin_w * sin_n * cos_i
    matrix[..., 0, 1] = -sin_w * cos_n - cos_w * sin_n * cos_i
    matrix[..., 0, 2] = sin_n * sin_i

    matrix[..., 1, 0] = cos_w * sin_n + sin_w * cos_n * cos_i
    matrix[..., 1, 1] = -sin_w * sin_n + cos_w * cos_n * cos_i
    matrix[..., 1, 2] = -cos_n * sin_i

    matrix[..., 2, 0] = sin_w * sin_i
    matrix[..., 2, 1] = cos_w * sin_i
    matrix[..., 2, 2] = cos_i

    return matrix


def state_from_elements(
    semi_major_axis: np.ndarray | float,
    eccentricity: np.ndarray | float,
    inclination: np.ndarray | float,
    node: np.ndarray | float,
    argument_of_perihelion: np.ndarray | float,
    mean_anomaly: np.ndarray | float,
    gm: float = GM_SUN,
) -> StateVector:
    """Convert Keplerian elements to a heliocentric state vector.

    Args:
        semi_major_axis: ``a`` in AU.
        eccentricity: ``e``, in ``[0, 1)``.
        inclination: ``i`` in radians.
        node: ``Omega``, longitude of ascending node, in radians.
        argument_of_perihelion: ``omega`` in radians.
        mean_anomaly: ``M`` in radians.
        gm: Gravitational parameter of the central body, in AU^3/day^2.

    Returns:
        A :class:`StateVector` whose arrays have the broadcast shape of the
        inputs, with a trailing axis of length 3.
    """
    a = np.asarray(semi_major_axis, dtype=float)
    e = np.asarray(eccentricity, dtype=float)

    eccentric = solve_kepler(mean_anomaly, e)
    cos_e, sin_e = np.cos(eccentric), np.sin(eccentric)
    sqrt_one_minus_e2 = np.sqrt(1.0 - e**2)

    # Position in the flat orbital plane.
    x_peri = a * (cos_e - e)
    y_peri = a * sqrt_one_minus_e2 * sin_e

    # Velocity in the same plane. Differentiating the two lines above with
    # respect to time introduces dE/dt, which follows from differentiating
    # Kepler's equation itself:  M = E - e sin E  =>  n = (1 - e cos E) dE/dt.
    eccentric_rate = mean_motion(a, gm) / (1.0 - e * cos_e)
    vx_peri = -a * sin_e * eccentric_rate
    vy_peri = a * sqrt_one_minus_e2 * cos_e * eccentric_rate

    rotation = _perifocal_to_ecliptic_matrix(
        np.asarray(inclination, dtype=float),
        np.asarray(node, dtype=float),
        np.asarray(argument_of_perihelion, dtype=float),
    )

    position_peri = np.stack(np.broadcast_arrays(x_peri, y_peri, np.zeros_like(x_peri)), axis=-1)
    velocity_peri = np.stack(np.broadcast_arrays(vx_peri, vy_peri, np.zeros_like(vx_peri)), axis=-1)

    # Batched matrix-vector product: (..., 3, 3) @ (..., 3) -> (..., 3)
    position = np.einsum("...ij,...j->...i", rotation, position_peri)
    velocity = np.einsum("...ij,...j->...i", rotation, velocity_peri)

    return StateVector(position, velocity)


def body_state(body: str, jd: float | np.ndarray) -> StateVector:
    """Heliocentric state of a named body at one or many Julian Dates.

    Example:
        >>> from orrery.timescales import julian_date
        >>> earth = body_state("earth", julian_date(2026, 7, 26))
        >>> bool(0.98 < float(earth.distance_au) < 1.02)
        True
    """
    key = body.lower()
    if key not in PLANETS:
        raise KeyError(f"unknown body {body!r}; known bodies: {sorted(PLANETS)}")

    elements = PLANETS[key].at(jd)
    return state_from_elements(
        elements["a"],
        elements["e"],
        elements["i"],
        elements["Omega"],
        elements["omega"],
        elements["M"],
    )


def system_state(
    jd: float | np.ndarray,
    bodies: tuple[str, ...] = PLANET_NAMES,
) -> dict[str, StateVector]:
    """States of several bodies at once, keyed by body name."""
    return {body: body_state(body, jd) for body in bodies}


def orbit_path(body: str, jd: float, samples: int = 360) -> np.ndarray:
    """Sample one full closed orbit, for drawing the trajectory line.

    The elements are frozen at ``jd`` and only the mean anomaly is swept through
    a full revolution. That yields a closed curve --- the ellipse as it stands
    *today* --- which is what you want for a rendered orbit line. Letting the
    elements drift too would produce a slowly opening spiral.

    Args:
        body: Body key, e.g. ``"saturn"``.
        jd: Julian Date at which to freeze the orbit's shape.
        samples: Number of points around the ellipse.

    Returns:
        Shape ``(samples, 3)`` array of positions in AU.
    """
    elements = PLANETS[body.lower()].at(jd)
    mean_anomalies = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=True)

    state = state_from_elements(
        elements["a"],
        elements["e"],
        elements["i"],
        elements["Omega"],
        elements["omega"],
        mean_anomalies,
    )
    return state.position


def ecliptic_to_equatorial(vectors: np.ndarray) -> np.ndarray:
    """Rotate ecliptic J2000 vectors into equatorial J2000.

    Star catalogues (Gaia included) use equatorial coordinates, while
    solar-system work uses the ecliptic. Phase 6 needs both in one scene, and
    the two frames differ by a single rotation about the shared x-axis, through
    the obliquity of the ecliptic.
    """
    cos_eps, sin_eps = np.cos(OBLIQUITY_J2000_RAD), np.sin(OBLIQUITY_J2000_RAD)
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_eps, -sin_eps],
            [0.0, sin_eps, cos_eps],
        ]
    )
    return np.einsum("ij,...j->...i", rotation, np.asarray(vectors, dtype=float))
