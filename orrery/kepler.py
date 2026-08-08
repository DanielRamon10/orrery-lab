"""Solving Kepler's equation.

The problem
-----------
Given how much *time* has passed, where is the planet on its ellipse?

Time enters through the **mean anomaly** ``M``, which grows perfectly linearly
with time (``M = n * (t - t_peri)``). But the planet does *not* move at a
constant rate: it speeds up near perihelion and slows down near aphelion
(Kepler's second law). The bridge between the two is Kepler's equation:

.. math::

    M = E - e \\sin E

where ``E`` is the **eccentric anomaly** --- an angle that does have a direct
geometric meaning on the ellipse. This equation is transcendental: there is no
closed-form solution for ``E``, so it must be solved numerically. That is what
this module does, for arrays of orbits at once.

Method
------
Newton-Raphson iteration, which converges quadratically:

.. math::

    E_{k+1} = E_k - \\frac{E_k - e \\sin E_k - M}{1 - e \\cos E_k}

Newton can stall for very eccentric orbits (comets, ``e -> 1``), so any element
that fails to converge falls back to bisection. That fallback is guaranteed to
work: rearranging the equation gives ``E = M + e sin E``, and since
``|sin E| <= 1`` the root is always trapped inside ``[M - e, M + e]``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalize_angle",
    "solve_kepler",
    "true_anomaly_from_eccentric",
    "eccentric_from_true_anomaly",
    "radius_from_eccentric",
    "mean_motion",
    "orbital_period",
]


def normalize_angle(angle: np.ndarray | float, centered: bool = True) -> np.ndarray | float:
    """Wrap an angle in radians into a single revolution.

    Args:
        angle: Angle(s) in radians.
        centered: If ``True`` wrap to ``[-pi, pi)``, otherwise to ``[0, 2*pi)``.

    Wrapping to ``[-pi, pi)`` matters for the Newton solver: it keeps the initial
    guess close to the root, which is what makes convergence fast.
    """
    two_pi = 2.0 * np.pi
    wrapped = np.mod(angle, two_pi)
    if centered:
        wrapped = np.where(wrapped >= np.pi, wrapped - two_pi, wrapped)
    return wrapped


def _starting_guess(mean_anomaly: np.ndarray, eccentricity: np.ndarray) -> np.ndarray:
    """Danby's starting value for Newton-Raphson.

    ``E0 = M + sign(sin M) * 0.85 * e`` is markedly better than the naive
    ``E0 = M`` at high eccentricity, and costs nothing extra.
    """
    return mean_anomaly + np.sign(np.sin(mean_anomaly)) * 0.85 * eccentricity


def _bisect_kepler(
    mean_anomaly: np.ndarray,
    eccentricity: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> np.ndarray:
    """Bracketed bisection fallback; always converges, just more slowly."""
    low = mean_anomaly - eccentricity
    high = mean_anomaly + eccentricity

    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        residual = mid - eccentricity * np.sin(mid) - mean_anomaly
        # Where the residual is positive the root lies below mid, and vice versa.
        high = np.where(residual > 0.0, mid, high)
        low = np.where(residual > 0.0, low, mid)
        if np.all(high - low < tolerance):
            break

    return 0.5 * (low + high)


def solve_kepler(
    mean_anomaly: np.ndarray | float,
    eccentricity: np.ndarray | float,
    tolerance: float = 1e-13,
    max_iterations: int = 64,
) -> np.ndarray:
    """Solve ``M = E - e sin E`` for the eccentric anomaly ``E``.

    Args:
        mean_anomaly: Mean anomaly in radians. Any shape; broadcast against
            ``eccentricity``.
        eccentricity: Orbital eccentricity, ``0 <= e < 1`` (elliptical orbits).
        tolerance: Absolute convergence tolerance on the residual, in radians.
        max_iterations: Iteration cap for both Newton and the bisection fallback.

    Returns:
        Eccentric anomaly in radians, wrapped to ``[-pi, pi)``, with the
        broadcast shape of the inputs.

    Raises:
        ValueError: If any eccentricity is outside ``[0, 1)``.

    Example:
        >>> float(solve_kepler(0.0, 0.5))
        0.0
        >>> E = solve_kepler(1.0, 0.2)
        >>> bool(abs(float(E - 0.2 * np.sin(E)) - 1.0) < 1e-12)
        True
    """
    mean, ecc = np.broadcast_arrays(
        np.asarray(mean_anomaly, dtype=float),
        np.asarray(eccentricity, dtype=float),
    )

    if np.any(ecc < 0.0) or np.any(ecc >= 1.0):
        raise ValueError(
            "solve_kepler handles elliptical orbits only: eccentricity must be in [0, 1). "
            "Parabolic and hyperbolic trajectories need Barker's equation or the "
            "hyperbolic Kepler equation instead."
        )

    mean = np.asarray(normalize_angle(mean), dtype=float)
    eccentric = _starting_guess(mean, ecc)

    for _ in range(max_iterations):
        residual = eccentric - ecc * np.sin(eccentric) - mean
        derivative = 1.0 - ecc * np.cos(eccentric)
        # derivative >= 1 - e > 0 for e < 1, so it never vanishes; the clip only
        # guards against catastrophic round-off as e approaches 1.
        step = residual / np.maximum(derivative, 1e-15)
        eccentric = eccentric - step
        if np.all(np.abs(residual) < tolerance):
            break

    # Repair any element Newton failed on, using the guaranteed bracket.
    final_residual = np.abs(eccentric - ecc * np.sin(eccentric) - mean)
    stalled = final_residual > tolerance
    if np.any(stalled):
        repaired = _bisect_kepler(mean, ecc, tolerance, max_iterations=200)
        eccentric = np.where(stalled, repaired, eccentric)

    return np.asarray(normalize_angle(eccentric), dtype=float)


def true_anomaly_from_eccentric(
    eccentric_anomaly: np.ndarray | float,
    eccentricity: np.ndarray | float,
) -> np.ndarray:
    """Convert eccentric anomaly ``E`` to true anomaly ``nu``.

    The true anomaly is the angle actually subtended at the focus (the Sun) ---
    the one you would measure with a telescope.

    Uses the half-angle form, which stays accurate near ``nu = pi`` where the
    ``cos``-based formula loses precision.
    """
    ecc = np.asarray(eccentricity, dtype=float)
    half = 0.5 * np.asarray(eccentric_anomaly, dtype=float)
    return 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(half),
        np.sqrt(1.0 - ecc) * np.cos(half),
    )


def eccentric_from_true_anomaly(
    true_anomaly: np.ndarray | float,
    eccentricity: np.ndarray | float,
) -> np.ndarray:
    """Inverse of :func:`true_anomaly_from_eccentric`."""
    ecc = np.asarray(eccentricity, dtype=float)
    half = 0.5 * np.asarray(true_anomaly, dtype=float)
    return 2.0 * np.arctan2(
        np.sqrt(1.0 - ecc) * np.sin(half),
        np.sqrt(1.0 + ecc) * np.cos(half),
    )


def radius_from_eccentric(
    semi_major_axis: np.ndarray | float,
    eccentricity: np.ndarray | float,
    eccentric_anomaly: np.ndarray | float,
) -> np.ndarray:
    """Heliocentric distance ``r = a (1 - e cos E)``."""
    return np.asarray(semi_major_axis) * (
        1.0 - np.asarray(eccentricity) * np.cos(np.asarray(eccentric_anomaly))
    )


def mean_motion(semi_major_axis: np.ndarray | float, gm: float) -> np.ndarray:
    """Mean angular rate ``n = sqrt(GM / a**3)``.

    This *is* Kepler's third law: with ``P = 2*pi/n`` it gives ``P**2 ~ a**3``.
    """
    return np.sqrt(gm / np.asarray(semi_major_axis, dtype=float) ** 3)


def orbital_period(semi_major_axis: np.ndarray | float, gm: float) -> np.ndarray:
    """Orbital period ``P = 2*pi*sqrt(a**3 / GM)``, in the time unit of ``gm``."""
    return 2.0 * np.pi / mean_motion(semi_major_axis, gm)
