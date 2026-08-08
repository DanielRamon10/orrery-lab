r"""Gravitational N-body integration.

Phase 1 solved the *two-body* problem exactly: one planet, one Sun, a closed
ellipse. Reality has nine bodies pulling on each other, and that problem has no
closed-form solution --- it has to be stepped forward numerically.

Which stepping method you choose matters more than its stated accuracy order, and
this module exists largely to make that visible.

Symplectic versus merely accurate
---------------------------------
Classical Runge-Kutta 4 is fourth-order accurate: halve the step and the error per
step falls by sixteen. Leapfrog is only second-order. And yet leapfrog is the right
choice for orbital work, because accuracy order describes the error over *one* step
while an orbit needs millions of them.

The difference is structural. Leapfrog is **symplectic**: it exactly conserves a
slightly-wrong energy, so its energy error *oscillates within a bounded band*
forever. RK4 conserves nothing exactly, so its energy error accumulates
secularly --- it drifts in one direction without limit.

Be careful about what that does and does not claim. It is *not* "leapfrog is always
more accurate". Over a few dozen orbits RK4 is comfortably ahead, because fourth
order beats second order; the test suite asserts as much, to keep this file honest.
The claim is narrower and more useful: leapfrog's error never grows, so there is
always some integration length past which it wins, and past which RK4's planets have
quietly spiralled into or out of the Sun while leapfrog's are still on orbits.

``scripts/plot_energy_drift.py`` measures both halves of that picture.

Units and the GM convention
---------------------------
Bodies are described by their gravitational parameter ``GM``, not by mass. This is
standard in celestial mechanics because ``GM`` is measured to about ten significant
figures while ``G`` alone is known to barely five --- carrying mass would import
that uncertainty for nothing.

One consequence: the "energy" and "angular momentum" returned here are the true
quantities multiplied by ``G``. Since ``G`` is a constant, that is irrelevant to
every use they have --- conservation and *relative* drift are unaffected --- but it
is why their absolute values will not match a textbook in joules.

Positions are in AU, velocities in AU/day, and ``GM`` in AU^3/day^2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "accelerations",
    "total_energy",
    "total_angular_momentum",
    "total_linear_momentum",
    "Trajectory",
    "integrate",
    "INTEGRATORS",
]

Integrator = Literal["leapfrog", "yoshida4", "rk4", "euler"]


# ---------------------------------------------------------------------------
# Forces
# ---------------------------------------------------------------------------


def accelerations(
    positions: np.ndarray,
    gms: np.ndarray,
    softening: float = 0.0,
) -> np.ndarray:
    r"""Gravitational acceleration on every body from every other body.

    .. math::

        \mathbf{a}_i = \sum_{j \neq i} GM_j
                       \frac{\mathbf{r}_j - \mathbf{r}_i}
                            {|\mathbf{r}_j - \mathbf{r}_i|^3}

    Computed as a single vectorised pass over the full ``(N, N)`` pair matrix. For
    the ten bodies of the solar system that is trivial; it is written this way
    because the integrator calls it millions of times.

    Args:
        positions: ``(N, 3)`` array in AU.
        gms: ``(N,)`` array of gravitational parameters in AU^3/day^2.
        softening: Plummer softening length in AU. Zero for planetary work, where
            bodies never come close enough for the singularity to matter. A small
            positive value keeps the force finite through a genuine collision,
            which is only useful for star-cluster style problems.

    Returns:
        ``(N, 3)`` accelerations in AU/day^2.
    """
    positions = np.asarray(positions, dtype=float)
    gms = np.asarray(gms, dtype=float)

    # separations[i, j] = r_j - r_i
    separations = positions[None, :, :] - positions[:, None, :]
    squared_distance = np.einsum("ijk,ijk->ij", separations, separations)

    if softening:
        squared_distance = squared_distance + softening**2

    # Self-interaction: setting the diagonal to infinity makes its contribution
    # exactly zero after the inverse-cube, with no branching and no NaN.
    np.fill_diagonal(squared_distance, np.inf)

    inverse_cube = squared_distance**-1.5
    return np.einsum("j,ij,ijk->ik", gms, inverse_cube, separations)


# ---------------------------------------------------------------------------
# Conserved quantities --- the diagnostics that make the method visible
# ---------------------------------------------------------------------------


def total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    gms: np.ndarray,
) -> float:
    r"""Total energy of the system, times ``G``.

    .. math::

        G E = \tfrac{1}{2}\sum_i GM_i |\mathbf{v}_i|^2
              - \sum_{i<j} \frac{GM_i\,GM_j}{|\mathbf{r}_i - \mathbf{r}_j|}

    Negative for a gravitationally bound system. Its *drift* is the single most
    informative diagnostic of an orbital integrator.
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    gms = np.asarray(gms, dtype=float)

    kinetic = 0.5 * np.sum(gms * np.einsum("ij,ij->i", velocities, velocities))

    separations = positions[None, :, :] - positions[:, None, :]
    distance = np.sqrt(np.einsum("ijk,ijk->ij", separations, separations))

    # Upper triangle only, so each pair is counted exactly once.
    pair_i, pair_j = np.triu_indices(len(gms), k=1)
    potential = -np.sum(gms[pair_i] * gms[pair_j] / distance[pair_i, pair_j])

    return float(kinetic + potential)


def total_angular_momentum(
    positions: np.ndarray,
    velocities: np.ndarray,
    gms: np.ndarray,
) -> np.ndarray:
    r"""Total angular momentum vector, times ``G``: :math:`\sum_i GM_i (r_i \times v_i)`.

    Conserved because gravity is a central force. Unlike energy, a symplectic
    integrator conserves this to machine precision rather than approximately.
    """
    gms = np.asarray(gms, dtype=float)[:, None]
    return np.sum(gms * np.cross(positions, velocities), axis=0)


def total_linear_momentum(velocities: np.ndarray, gms: np.ndarray) -> np.ndarray:
    r"""Total linear momentum, times ``G``: :math:`\sum_i GM_i \mathbf{v}_i`.

    Zero in the barycentric frame, and it must stay zero: any drift means the
    system's centre of mass is wandering, which is unphysical and points at a bug
    rather than at discretisation error.
    """
    gms = np.asarray(gms, dtype=float)[:, None]
    return np.sum(gms * np.asarray(velocities, dtype=float), axis=0)


def centre_of_mass(positions: np.ndarray, gms: np.ndarray) -> np.ndarray:
    """Mass-weighted mean position (the barycentre)."""
    gms = np.asarray(gms, dtype=float)
    return np.einsum("i,ij->j", gms, np.asarray(positions, dtype=float)) / gms.sum()


# ---------------------------------------------------------------------------
# Single steps
# ---------------------------------------------------------------------------


def _leapfrog_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    gms: np.ndarray,
    dt: float,
    softening: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One kick-drift-kick leapfrog step.

    Symplectic, second-order, and exactly time-reversible: run it with ``-dt`` and
    you land back where you started, up to floating-point round-off. That property
    is what bounds the energy error, and it is tested directly.
    """
    half_kick = velocities + 0.5 * dt * accelerations(positions, gms, softening)
    drifted = positions + dt * half_kick
    full_kick = half_kick + 0.5 * dt * accelerations(drifted, gms, softening)
    return drifted, full_kick


# Yoshida's fourth-order composition weights.
#
# Three leapfrog steps with these lengths cancel the leading error term of a single
# one, giving fourth-order accuracy while remaining symplectic. The middle step is
# *negative* --- the integrator briefly runs time backwards, which is why the
# weights must sum to exactly 1.
_CUBE_ROOT_TWO = 2.0 ** (1.0 / 3.0)
_YOSHIDA_OUTER = 1.0 / (2.0 - _CUBE_ROOT_TWO)
_YOSHIDA_INNER = -_CUBE_ROOT_TWO / (2.0 - _CUBE_ROOT_TWO)
YOSHIDA_WEIGHTS: tuple[float, float, float] = (
    _YOSHIDA_OUTER,
    _YOSHIDA_INNER,
    _YOSHIDA_OUTER,
)


def _yoshida4_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    gms: np.ndarray,
    dt: float,
    softening: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One fourth-order symplectic step, as three weighted leapfrog steps."""
    for weight in YOSHIDA_WEIGHTS:
        positions, velocities = _leapfrog_step(
            positions, velocities, gms, weight * dt, softening
        )
    return positions, velocities


def _rk4_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    gms: np.ndarray,
    dt: float,
    softening: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One classical Runge-Kutta 4 step.

    Fourth-order accurate and **not** symplectic --- included precisely so the
    difference can be measured rather than asserted. Costs four force evaluations
    per step against leapfrog's two.
    """
    def derivative(r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return v, accelerations(r, gms, softening)

    dr1, dv1 = derivative(positions, velocities)
    dr2, dv2 = derivative(positions + 0.5 * dt * dr1, velocities + 0.5 * dt * dv1)
    dr3, dv3 = derivative(positions + 0.5 * dt * dr2, velocities + 0.5 * dt * dv2)
    dr4, dv4 = derivative(positions + dt * dr3, velocities + dt * dv3)

    new_positions = positions + (dt / 6.0) * (dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4)
    new_velocities = velocities + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)
    return new_positions, new_velocities


def _euler_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    gms: np.ndarray,
    dt: float,
    softening: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One explicit Euler step.

    First-order and badly non-symplectic. Present only as the cautionary baseline:
    it visibly spirals a circular orbit outward within a few revolutions, which
    makes the point about method choice faster than any amount of prose.
    """
    new_velocities = velocities + dt * accelerations(positions, gms, softening)
    new_positions = positions + dt * velocities
    return new_positions, new_velocities


StepFunction = Callable[
    [np.ndarray, np.ndarray, np.ndarray, float, float], tuple[np.ndarray, np.ndarray]
]

#: Available integrators, with whether each is symplectic and its accuracy order.
INTEGRATORS: dict[str, tuple[StepFunction, bool, int]] = {
    "euler": (_euler_step, False, 1),
    "leapfrog": (_leapfrog_step, True, 2),
    "yoshida4": (_yoshida4_step, True, 4),
    "rk4": (_rk4_step, False, 4),
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trajectory:
    """The result of an integration, plus its conservation diagnostics.

    Attributes:
        names: Body names, in the order of the arrays' second axis.
        times: ``(S,)`` days elapsed since the start, at each stored sample.
        positions: ``(S, N, 3)`` in AU.
        velocities: ``(S, N, 3)`` in AU/day.
        energy: ``(S,)`` total energy (times ``G``) at each sample.
        angular_momentum: ``(S, 3)``.
        linear_momentum: ``(S, 3)``.
        integrator: Which stepper produced this.
        dt: Step size in days. Note that samples may be further apart than this.
    """

    names: tuple[str, ...]
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    energy: np.ndarray
    angular_momentum: np.ndarray
    linear_momentum: np.ndarray
    integrator: str
    dt: float

    @property
    def relative_energy_error(self) -> np.ndarray:
        """``|E(t) - E(0)| / |E(0)|``, the dimensionless drift.

        This is the number to plot. For a symplectic integrator it oscillates
        inside a band set by the step size; for RK4 it grows without limit.
        """
        return np.abs((self.energy - self.energy[0]) / self.energy[0])

    @property
    def relative_angular_momentum_error(self) -> np.ndarray:
        """Relative change in the magnitude of the angular momentum vector."""
        magnitude = np.linalg.norm(self.angular_momentum, axis=-1)
        return np.abs((magnitude - magnitude[0]) / magnitude[0])

    def body(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Positions and velocities of one named body over time."""
        index = self.names.index(name)
        return self.positions[:, index, :], self.velocities[:, index, :]

    def __len__(self) -> int:
        return len(self.times)


def integrate(
    positions: np.ndarray,
    velocities: np.ndarray,
    gms: np.ndarray,
    *,
    duration_days: float,
    dt: float,
    integrator: Integrator = "leapfrog",
    names: tuple[str, ...] | None = None,
    sample_every: int = 1,
    softening: float = 0.0,
) -> Trajectory:
    """Integrate a gravitating system forward (or backward) in time.

    Args:
        positions: ``(N, 3)`` initial positions in AU.
        velocities: ``(N, 3)`` initial velocities in AU/day.
        gms: ``(N,)`` gravitational parameters in AU^3/day^2.
        duration_days: Total time to integrate. May be negative to run backwards.
        dt: Step size in days. Must be positive; the sign is taken from
            ``duration_days``.
        integrator: One of :data:`INTEGRATORS`.
        names: Optional body labels, for :meth:`Trajectory.body`.
        sample_every: Store every ``n``-th step. Integration always runs at ``dt``;
            this only controls how much is kept, which is what makes long runs
            possible without exhausting memory.
        softening: Passed through to :func:`accelerations`.

    Returns:
        A :class:`Trajectory`, always including the initial and final states.

    Raises:
        ValueError: On an unknown integrator or a non-positive ``dt``.
    """
    if integrator not in INTEGRATORS:
        raise ValueError(
            f"unknown integrator {integrator!r}; choose from {sorted(INTEGRATORS)}"
        )
    if dt <= 0:
        raise ValueError(f"dt must be positive (run backwards via duration_days); got {dt}")
    if sample_every < 1:
        raise ValueError(f"sample_every must be at least 1; got {sample_every}")

    step, _, _ = INTEGRATORS[integrator]

    current_positions = np.array(positions, dtype=float, copy=True)
    current_velocities = np.array(velocities, dtype=float, copy=True)
    gms = np.asarray(gms, dtype=float)

    signed_dt = dt if duration_days >= 0 else -dt
    total_steps = int(round(abs(duration_days) / dt))

    stored_times = [0.0]
    stored_positions = [current_positions.copy()]
    stored_velocities = [current_velocities.copy()]

    for index in range(1, total_steps + 1):
        current_positions, current_velocities = step(
            current_positions, current_velocities, gms, signed_dt, softening
        )
        # Always keep the last step, so the trajectory ends where it really ended.
        if index % sample_every == 0 or index == total_steps:
            stored_times.append(index * signed_dt)
            stored_positions.append(current_positions.copy())
            stored_velocities.append(current_velocities.copy())

    positions_array = np.array(stored_positions)
    velocities_array = np.array(stored_velocities)

    energy = np.array(
        [
            total_energy(sample_positions, sample_velocities, gms)
            for sample_positions, sample_velocities in zip(
                positions_array, velocities_array, strict=True
            )
        ]
    )
    angular_momentum = np.array(
        [
            total_angular_momentum(sample_positions, sample_velocities, gms)
            for sample_positions, sample_velocities in zip(
                positions_array, velocities_array, strict=True
            )
        ]
    )
    linear_momentum = np.array(
        [total_linear_momentum(sample_velocities, gms) for sample_velocities in velocities_array]
    )

    return Trajectory(
        names=names if names is not None else tuple(f"body{i}" for i in range(len(gms))),
        times=np.array(stored_times),
        positions=positions_array,
        velocities=velocities_array,
        energy=energy,
        angular_momentum=angular_momentum,
        linear_momentum=linear_momentum,
        integrator=integrator,
        dt=dt,
    )
