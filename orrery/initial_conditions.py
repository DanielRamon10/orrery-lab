"""Turning the ephemeris into starting conditions for the N-body integrator.

The ephemeris of phase 1 is **heliocentric**: it places each planet relative to a
Sun that sits, by definition, at the origin and does not move. That is exactly what
a two-body solution means, and it is the wrong frame for N-body work, because in
reality the Sun is pulled about by the planets --- Jupiter alone displaces it by
roughly its own radius.

Integrating in a frame whose origin accelerates would inject a fictitious force and
quietly break every conservation law the integrator is supposed to respect. So the
states are shifted into the **barycentric** frame, where the system's centre of mass
sits at the origin with zero velocity, and stays there.

That shift is also what makes ``total_linear_momentum`` a meaningful diagnostic: it
starts at zero, and any departure from zero afterwards is a bug rather than
discretisation error.

A caveat worth stating plainly
------------------------------
JPL's approximate-element table supplies **mean** elements — a smooth fit through
1800-2050 with the short-period wobbles averaged out. The integrator needs
**osculating** elements, meaning the instantaneous state right now, wobbles included.
Using the former as the latter is an approximation, and it shows: integrate from here
and Saturn's mean semi-major axis settles near 9.505 AU against the table's 9.537, a
third of a percent out.

That is fine for what this module is for — demonstrating and validating integrators,
where the initial condition only has to be a physically sensible solar system. It
would not be fine for predicting a real close approach decades ahead, which needs
osculating elements from a full ephemeris such as DE440.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import GM_BODY, GM_SUN
from .elements import PLANET_NAMES
from .ephemeris import body_state
from .kepler import mean_motion
from .nbody import centre_of_mass, total_linear_momentum

__all__ = ["SystemState", "solar_system_state", "two_body_state"]


@dataclass(frozen=True)
class SystemState:
    """A set of bodies ready to hand to :func:`orrery.nbody.integrate`.

    Attributes:
        names: Body labels, ordered to match the arrays.
        positions: ``(N, 3)`` in AU.
        velocities: ``(N, 3)`` in AU/day.
        gms: ``(N,)`` gravitational parameters in AU^3/day^2.
        epoch_jd: Julian Date these states correspond to.
    """

    names: tuple[str, ...]
    positions: np.ndarray
    velocities: np.ndarray
    gms: np.ndarray
    epoch_jd: float

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(positions, velocities, gms)``, for splatting into the integrator."""
        return self.positions, self.velocities, self.gms

    def __len__(self) -> int:
        return len(self.names)


def _shift_to_barycentre(
    positions: np.ndarray,
    velocities: np.ndarray,
    gms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Move the centre of mass to the origin and bring it to rest."""
    total_gm = gms.sum()
    barycentre_position = np.einsum("i,ij->j", gms, positions) / total_gm
    barycentre_velocity = np.einsum("i,ij->j", gms, velocities) / total_gm
    return positions - barycentre_position, velocities - barycentre_velocity


def solar_system_state(
    jd: float,
    bodies: tuple[str, ...] = PLANET_NAMES,
    include_sun: bool = True,
) -> SystemState:
    """Barycentric state of the Sun and planets at a Julian Date.

    Args:
        jd: Julian Date of the epoch.
        bodies: Which planets to include, in the order they should appear.
        include_sun: Whether to add the Sun. Almost always yes --- without it the
            planets have nothing to orbit.

    Returns:
        A :class:`SystemState` whose centre of mass is at rest at the origin.

    Note:
        Earth's entry is really the Earth-Moon barycentre, inherited from the
        element table. For solar-system dynamics that is the correct body to use:
        the Moon's monthly wobble is irrelevant to how Earth perturbs Jupiter.
    """
    names: list[str] = []
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    gms: list[float] = []

    if include_sun:
        # In the heliocentric frame the Sun is the origin, at rest, by construction.
        names.append("sun")
        positions.append(np.zeros(3))
        velocities.append(np.zeros(3))
        gms.append(GM_SUN)

    for body in bodies:
        state = body_state(body, jd)
        names.append(body)
        positions.append(np.asarray(state.position, dtype=float))
        velocities.append(np.asarray(state.velocity, dtype=float))
        gms.append(GM_BODY[body])

    position_array = np.array(positions)
    velocity_array = np.array(velocities)
    gm_array = np.array(gms)

    position_array, velocity_array = _shift_to_barycentre(
        position_array, velocity_array, gm_array
    )

    return SystemState(
        names=tuple(names),
        positions=position_array,
        velocities=velocity_array,
        gms=gm_array,
        epoch_jd=float(jd),
    )


def two_body_state(
    semi_major_axis_au: float = 1.0,
    eccentricity: float = 0.0,
    central_gm: float = GM_SUN,
    orbiting_gm: float = 0.0,
) -> SystemState:
    """A clean two-body system, started at perihelion.

    Exists for testing. The two-body problem has an exact solution, so an
    integrator's error against it is *measurable* rather than merely plausible ---
    which is how the convergence-order tests establish that leapfrog really is
    second-order and Yoshida really is fourth.

    The bodies start at perihelion with velocity perpendicular to the separation,
    from the vis-viva equation:

    .. math::  v_{peri} = \\sqrt{\\frac{GM_{total}}{a}\\cdot\\frac{1+e}{1-e}}

    Args:
        semi_major_axis_au: Semi-major axis of the relative orbit.
        eccentricity: Orbital eccentricity, ``0 <= e < 1``.
        central_gm: Gravitational parameter of the central body.
        orbiting_gm: Gravitational parameter of the orbiting body. Zero gives the
            restricted problem, where the central body does not move.

    Returns:
        A barycentric two-body :class:`SystemState` at epoch J2000.
    """
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError(f"eccentricity must be in [0, 1); got {eccentricity}")

    total_gm = central_gm + orbiting_gm
    perihelion = semi_major_axis_au * (1.0 - eccentricity)
    perihelion_speed = np.sqrt(
        (total_gm / semi_major_axis_au) * (1.0 + eccentricity) / (1.0 - eccentricity)
    )

    positions = np.array([[0.0, 0.0, 0.0], [perihelion, 0.0, 0.0]])
    velocities = np.array([[0.0, 0.0, 0.0], [0.0, perihelion_speed, 0.0]])
    gms = np.array([central_gm, orbiting_gm])

    positions, velocities = _shift_to_barycentre(positions, velocities, gms)

    from .constants import J2000_JD

    return SystemState(
        names=("central", "orbiter"),
        positions=positions,
        velocities=velocities,
        gms=gms,
        epoch_jd=J2000_JD,
    )


def two_body_period_days(
    semi_major_axis_au: float,
    central_gm: float = GM_SUN,
    orbiting_gm: float = 0.0,
) -> float:
    """Exact period of the relative orbit, from Kepler's third law.

    Uses the *combined* gravitational parameter, which is the version of the third
    law that holds when the second body's mass is not negligible.
    """
    return float(2.0 * np.pi / mean_motion(semi_major_axis_au, central_gm + orbiting_gm))


def report_frame_quality(state: SystemState) -> dict[str, float]:
    """How well the barycentric shift worked, as numbers.

    Used by the tests and by ``scripts/plot_energy_drift.py``. Both values should be
    at round-off level; anything larger means the shift is wrong.
    """
    return {
        "centre_of_mass_offset_au": float(
            np.linalg.norm(centre_of_mass(state.positions, state.gms))
        ),
        "linear_momentum_magnitude": float(
            np.linalg.norm(total_linear_momentum(state.velocities, state.gms))
        ),
    }
