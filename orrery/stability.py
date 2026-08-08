r"""Generating an orbital-stability dataset from the project's own physics.

The idea
--------
Phase 5 needs a machine-learning problem where the labels are *earned* rather than
downloaded. This module builds one: sample thousands of two-planet systems, integrate
every one of them with the same gravity implemented in phase 3, and label each as
stable or unstable by what actually happened. The model then has to predict that
outcome from the initial conditions alone.

That makes the whole chain auditable in a way a downloaded dataset never is. If the
labels are wrong, the physics is wrong, and the physics has 44 tests on it.

The question
------------
Given two planets around a star, will they stay on their orbits or wreck each other?
The controlling quantity is their separation measured in **mutual Hill radii**:

.. math::

    R_H = \left(\frac{m_1 + m_2}{3 M_\star}\right)^{1/3} \frac{a_1 + a_2}{2},
    \qquad \Delta = \frac{a_2 - a_1}{R_H}

Gladman (1993) showed that two planets on initially circular, coplanar orbits are
guaranteed Hill-stable when :math:`\Delta > 2\sqrt{3} \approx 3.46`. That is a
*sufficient* condition and only for the circular coplanar case, so it is a genuine
baseline rather than the answer: it says nothing about eccentric systems, and being
above the threshold does not guarantee survival once eccentricity is in play.

Beating that baseline is the model's actual job, and
:func:`orrery.models.evaluate_stability_models` measures whether it does.

Why the integrator here is a separate, batched one
--------------------------------------------------
:func:`orrery.nbody.integrate` steps one system at a time, which is right for the
solar system and hopeless for three thousand of them: nearly all the cost is Python
overhead, paid per step per system. The integrator below carries a leading batch axis
and steps every system at once, turning three thousand sequential runs into one
vectorised run.

It is the same kick-drift-kick leapfrog, so it must produce the same answer, and
``tests/test_stability.py`` asserts exactly that against the phase 3 implementation.
The project already used this pattern once, for the Python/TypeScript ephemeris
parity — a fast second implementation is only safe when something pins it to the
reference.

Two planets are not enough
--------------------------
Everything above concerns a *pair* of planets, where the Hill criterion is a genuine
guide. Add a third and the mechanism of instability changes: neighbouring pairs can each
be comfortably Hill-stable while the system as a whole comes apart, because the
resonances belonging to different pairs **overlap** and open a chaotic region no
pairwise criterion can see.

:func:`build_multiplanet_dataset` measures that directly, by running two- and
three-planet systems at matched separations and comparing survival. It is the reason
this module generalised past the pair it was written for.

Mutual inclination
------------------
Tilting the orbital planes was expected to be one more way to break a system. It is the
opposite. ``max_inclination_deg`` lifts the planets out of a shared plane, and survival
climbs steeply: at 2.5-6 mutual Hill radii, three-planet systems go from 32% to 80%
survival for a median mutual inclination of only 4.5 degrees, and ejections collapse
from 24% to 2%. For scale, the solar system's own planets sit a median 2.2 degrees apart,
so this is well inside the range a real system occupies.

The reason is geometric. Resonance overlap and close encounters both need the planets to
actually meet, and orbits that do not share a plane mostly miss one another --- they
cross the same radius at different heights. Ejections, which require a close encounter,
are the first failure mode to disappear, which is what pins the cause on encounters
rather than on slow secular drift.

This puts a caveat on the paragraph above: the third planet's penalty is a **coplanar**
phenomenon. Coplanar, the two- and three-planet survival rates differ by 39 points. At a
median mutual inclination of 18 degrees, they differ by 0.6.

Honest limits
-------------
* **Inclination is drawn, not evolved.** The planes are tilted at t = 0 and the sweep
  reports what survives; the secular oscillation of inclination against eccentricity
  that a longer integration would show is not modelled separately.
* **The label is "unstable within the integration window"**, not "unstable ever".
  Systems can and do destabilise on timescales far longer than any window used here,
  so the negative class really means "survived this long".
* **The star is a point mass** with no tides and no relativity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import GM_SUN

__all__ = [
    "HILL_STABILITY_THRESHOLD",
    "SystemSample",
    "StabilityDataset",
    "mutual_hill_radius",
    "hill_separation",
    "sample_two_planet_systems",
    "batch_accelerations",
    "integrate_batch",
    "build_stability_dataset",
    "FEATURE_NAMES",
    "MultiPlanetSample",
    "MultiPlanetDataset",
    "sample_planet_systems",
    "build_multiplanet_dataset",
]

#: Gladman's analytic threshold, ``2 * sqrt(3)``.
#:
#: Above this, two planets on initially circular coplanar orbits are provably
#: Hill-stable. Below it, no guarantee either way — the criterion is sufficient, not
#: necessary, which is exactly why a learned model has room to improve on it.
HILL_STABILITY_THRESHOLD = 2.0 * np.sqrt(3.0)

#: Semi-major axis of the inner planet, in AU. Fixed, because only the *ratio* of the
#: two orbits matters dynamically; holding it constant makes the time unit uniform
#: across the whole dataset.
INNER_SEMI_MAJOR_AXIS_AU = 1.0

#: Features handed to the model. Deliberately all quantities knowable at t=0.
FEATURE_NAMES: tuple[str, ...] = (
    "log_mass_inner",
    "log_mass_outer",
    "mass_ratio",
    "semi_major_axis_ratio",
    "hill_separation",
    "eccentricity_inner",
    "eccentricity_outer",
    "max_eccentricity",
    "eccentricity_crossing",
)


@dataclass(frozen=True)
class SystemSample:
    """A batch of randomly drawn two-planet systems.

    Arrays are all shape ``(S,)`` for ``S`` systems. Masses are gravitational
    parameters in AU^3/day^2, matching the GM convention used throughout.
    """

    gm_inner: np.ndarray
    gm_outer: np.ndarray
    semi_major_axis_inner: np.ndarray
    semi_major_axis_outer: np.ndarray
    eccentricity_inner: np.ndarray
    eccentricity_outer: np.ndarray
    #: Initial true anomaly of each planet, radians.
    phase_inner: np.ndarray
    phase_outer: np.ndarray
    #: Argument of pericentre of the outer planet, radians. The inner planet's is
    #: fixed at zero, since only the relative orientation matters.
    pericentre_outer: np.ndarray
    #: Inclination of each orbit to the reference plane, radians. ``None`` for a
    #: coplanar draw, which is the default and is what the Phase 5 dataset is built
    #: from. Kept as ``None`` rather than an array of zeros so that "this sample has
    #: no inclination in it" is a fact about the object, not a value to be compared
    #: against a tolerance.
    inclinations: np.ndarray | None = None
    #: Longitude of the ascending node of each orbit, radians. Meaningless when the
    #: inclination is zero, so ``None`` travels with it.
    nodes: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.gm_inner)

    @property
    def is_coplanar(self) -> bool:
        return self.inclinations is None

    def orbit_normals(self) -> np.ndarray:
        """Unit normal of each orbit, shape ``(S, 2, 3)``.

        The normal is what carries the orientation of a plane; the angle between two
        of them is the mutual inclination, which is the quantity with dynamical
        meaning. Individual inclinations are not: two orbits each tilted 30 degrees
        can be coplanar or 60 degrees apart depending on their nodes.
        """
        if self.inclinations is None:
            normals = np.zeros((len(self), 2, 3))
            normals[..., 2] = 1.0
            return normals

        sin_i, cos_i = np.sin(self.inclinations), np.cos(self.inclinations)
        return np.stack(
            [sin_i * np.sin(self.nodes), -sin_i * np.cos(self.nodes), cos_i], axis=-1
        )

    def mutual_inclination(self) -> np.ndarray:
        """Angle between the two orbit normals, radians, shape ``(S,)``."""
        normals = self.orbit_normals()
        dot = np.clip(np.sum(normals[:, 0, :] * normals[:, 1, :], axis=-1), -1.0, 1.0)
        return np.arccos(dot)


@dataclass(frozen=True)
class StabilityDataset:
    """Features, labels and the diagnostics behind each label.

    Attributes:
        features: ``(S, F)`` design matrix, columns named by :data:`FEATURE_NAMES`.
        labels: ``(S,)`` of 1 for stable, 0 for unstable.
        hill_separation: ``(S,)`` the analytic baseline's input.
        max_axis_change: ``(S,)`` largest fractional change in either semi-major axis.
        min_separation_hill: ``(S,)`` closest approach, in mutual Hill radii.
        escaped: ``(S,)`` whether either planet became unbound.
        orbits_simulated: How long each system was integrated, in inner orbits.
        sample: The underlying draw, kept so a case can be replayed.
    """

    features: np.ndarray
    labels: np.ndarray
    hill_separation: np.ndarray
    max_axis_change: np.ndarray
    min_separation_hill: np.ndarray
    escaped: np.ndarray
    orbits_simulated: float
    sample: SystemSample

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def stable_fraction(self) -> float:
        return float(np.mean(self.labels))

    @property
    def mutual_inclination_deg(self) -> np.ndarray:
        """``(S,)`` angle between the two orbit planes, degrees. Zero if coplanar.

        Deliberately *not* a model feature. The whole point of measuring transfer is
        to ask what a model trained without this quantity does when it changes, so
        handing it to the model would answer a different question.
        """
        return np.degrees(self.sample.mutual_inclination())


def mutual_hill_radius(
    gm_inner: np.ndarray,
    gm_outer: np.ndarray,
    semi_major_axis_inner: np.ndarray,
    semi_major_axis_outer: np.ndarray,
    gm_star: float = GM_SUN,
) -> np.ndarray:
    r"""Mutual Hill radius of a planet pair.

    .. math::

        R_H = \left(\frac{m_1 + m_2}{3 M_\star}\right)^{1/3} \frac{a_1 + a_2}{2}

    Mass appears only as a ratio, so gravitational parameters can be used directly
    and ``G`` cancels.
    """
    mass_factor = ((gm_inner + gm_outer) / (3.0 * gm_star)) ** (1.0 / 3.0)
    return mass_factor * 0.5 * (semi_major_axis_inner + semi_major_axis_outer)


def hill_separation(
    gm_inner: np.ndarray,
    gm_outer: np.ndarray,
    semi_major_axis_inner: np.ndarray,
    semi_major_axis_outer: np.ndarray,
    gm_star: float = GM_SUN,
) -> np.ndarray:
    """Orbital separation expressed in mutual Hill radii --- the quantity ``Delta``."""
    radius = mutual_hill_radius(
        gm_inner, gm_outer, semi_major_axis_inner, semi_major_axis_outer, gm_star
    )
    return (semi_major_axis_outer - semi_major_axis_inner) / radius


def sample_two_planet_systems(
    count: int,
    seed: int = 20260808,
    hill_separation_range: tuple[float, float] = (1.5, 12.0),
    log_mass_range: tuple[float, float] = (-6.0, -3.0),
    max_eccentricity: float = 0.15,
    max_inclination_deg: float = 0.0,
    gm_star: float = GM_SUN,
) -> SystemSample:
    """Draw random two-planet systems, parameterised by Hill separation.

    Sampling ``Delta`` directly, rather than sampling ``a2`` and computing ``Delta``,
    is deliberate: it spreads the draws evenly across the region where the stability
    boundary actually lives. Sampling ``a2`` uniformly would pile almost every system
    far above the threshold and leave the interesting band nearly empty.

    Args:
        count: Number of systems.
        seed: Fixed for reproducibility.
        hill_separation_range: Range of ``Delta`` to sample. The lower bound sits
            below Gladman's 3.46 so both sides of the analytic boundary are covered.
        log_mass_range: ``log10`` of planet mass relative to the star. The default
            spans roughly Earth-mass to a few Jupiters.
        max_eccentricity: Eccentricities are drawn uniformly in ``[0, this]``.
        max_inclination_deg: Each orbit is tilted by an angle drawn uniformly in
            ``[0, this]`` with a random node. Zero, the default, means coplanar and
            skips the draws altogether, so the Phase 5 dataset reproduces exactly
            rather than approximately --- see :func:`sample_planet_systems` for why
            drawing and discarding would not be equivalent.
        gm_star: Gravitational parameter of the central star.

    Returns:
        A :class:`SystemSample` of the requested size.
    """
    if count < 1:
        raise ValueError(f"count must be positive; got {count}")
    if not 0.0 <= max_inclination_deg <= 180.0:
        raise ValueError(
            f"max_inclination_deg must be in [0, 180]; got {max_inclination_deg}"
        )

    generator = np.random.default_rng(seed)

    gm_inner = gm_star * 10.0 ** generator.uniform(*log_mass_range, size=count)
    gm_outer = gm_star * 10.0 ** generator.uniform(*log_mass_range, size=count)

    target_separation = generator.uniform(*hill_separation_range, size=count)
    inner_axis = np.full(count, INNER_SEMI_MAJOR_AXIS_AU)

    # Invert Delta = (a2 - a1) / R_H for a2. R_H depends on a2, so solve directly:
    #   a2 - a1 = Delta * k * (a1 + a2) / 2   with   k = ((m1+m2)/(3 M*))^(1/3)
    #   a2 (1 - Delta k / 2) = a1 (1 + Delta k / 2)
    mass_factor = ((gm_inner + gm_outer) / (3.0 * gm_star)) ** (1.0 / 3.0)
    half = 0.5 * target_separation * mass_factor
    outer_axis = inner_axis * (1.0 + half) / (1.0 - half)

    eccentricity_inner = generator.uniform(0.0, max_eccentricity, size=count)
    eccentricity_outer = generator.uniform(0.0, max_eccentricity, size=count)
    phase_inner = generator.uniform(0.0, 2.0 * np.pi, size=count)
    phase_outer = generator.uniform(0.0, 2.0 * np.pi, size=count)
    pericentre_outer = generator.uniform(0.0, 2.0 * np.pi, size=count)

    # Drawn last, and only when asked for, so that every draw above is untouched by
    # the existence of this parameter and the coplanar dataset stays bit-identical.
    inclinations = nodes = None
    if max_inclination_deg > 0.0:
        inclinations = np.radians(
            generator.uniform(0.0, max_inclination_deg, size=(count, 2))
        )
        nodes = generator.uniform(0.0, 2.0 * np.pi, size=(count, 2))

    return SystemSample(
        gm_inner=gm_inner,
        gm_outer=gm_outer,
        semi_major_axis_inner=inner_axis,
        semi_major_axis_outer=outer_axis,
        eccentricity_inner=eccentricity_inner,
        eccentricity_outer=eccentricity_outer,
        phase_inner=phase_inner,
        phase_outer=phase_outer,
        pericentre_outer=pericentre_outer,
        inclinations=inclinations,
        nodes=nodes,
    )


def _initial_state(
    sample: SystemSample, gm_star: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build barycentric positions, velocities and masses for the whole batch.

    Returns arrays of shape ``(S, 3, 3)``, ``(S, 3, 3)`` and ``(S, 3)``, with body 0
    the star. Planets are placed on their ellipses at the sampled true anomaly, using
    the same perifocal construction as :mod:`orrery.ephemeris`.

    The full ``Rz(node) Rx(inclination) Rz(pericentre)`` rotation is applied
    unconditionally. With zero inclination and node it collapses to the plane rotation
    this function used to do, and does so *exactly*: ``sin(0.0)`` is 0.0 and
    ``cos(0.0)`` is 1.0 with no rounding, so every inclined term multiplies out to a
    hard zero rather than to something small. That is what lets the coplanar dataset
    stay bit-identical while the general case is available.
    """
    count = len(sample)
    positions = np.zeros((count, 3, 3))
    velocities = np.zeros((count, 3, 3))
    gms = np.stack(
        [np.full(count, gm_star), sample.gm_inner, sample.gm_outer], axis=1
    )

    zero = np.zeros(count)
    inclinations, nodes = sample.inclinations, sample.nodes

    for index, (axis, eccentricity, phase, pericentre) in enumerate(
        (
            (sample.semi_major_axis_inner, sample.eccentricity_inner, sample.phase_inner, 0.0),
            (
                sample.semi_major_axis_outer,
                sample.eccentricity_outer,
                sample.phase_outer,
                sample.pericentre_outer,
            ),
        ),
        start=1,
    ):
        # Radius and in-plane velocity from the true anomaly.
        semi_latus_rectum = axis * (1.0 - eccentricity**2)
        radius = semi_latus_rectum / (1.0 + eccentricity * np.cos(phase))
        speed_scale = np.sqrt(gm_star / semi_latus_rectum)

        cos_phase, sin_phase = np.cos(phase), np.sin(phase)
        x_perifocal = radius * cos_phase
        y_perifocal = radius * sin_phase
        vx_perifocal = -speed_scale * sin_phase
        vy_perifocal = speed_scale * (eccentricity + cos_phase)

        inclination = zero if inclinations is None else inclinations[:, index - 1]
        node = zero if nodes is None else nodes[:, index - 1]

        cos_peri, sin_peri = np.cos(pericentre), np.sin(pericentre)
        cos_inc, sin_inc = np.cos(inclination), np.sin(inclination)
        cos_node, sin_node = np.cos(node), np.sin(node)

        # Columns of Rz(node) Rx(inclination) Rz(pericentre), the first two of which
        # are all a perifocal vector with no z component needs.
        m00 = cos_node * cos_peri - sin_node * sin_peri * cos_inc
        m01 = -cos_node * sin_peri - sin_node * cos_peri * cos_inc
        m10 = sin_node * cos_peri + cos_node * sin_peri * cos_inc
        m11 = -sin_node * sin_peri + cos_node * cos_peri * cos_inc
        m20 = sin_peri * sin_inc
        m21 = cos_peri * sin_inc

        positions[:, index, 0] = m00 * x_perifocal + m01 * y_perifocal
        positions[:, index, 1] = m10 * x_perifocal + m11 * y_perifocal
        positions[:, index, 2] = m20 * x_perifocal + m21 * y_perifocal
        velocities[:, index, 0] = m00 * vx_perifocal + m01 * vy_perifocal
        velocities[:, index, 1] = m10 * vx_perifocal + m11 * vy_perifocal
        velocities[:, index, 2] = m20 * vx_perifocal + m21 * vy_perifocal

    # Shift to the barycentre so the systems do not drift while being integrated.
    total_gm = gms.sum(axis=1, keepdims=True)
    positions -= np.einsum("sn,snk->sk", gms, positions)[:, None, :] / total_gm[:, :, None]
    velocities -= np.einsum("sn,snk->sk", gms, velocities)[:, None, :] / total_gm[:, :, None]

    return positions, velocities, gms


def batch_accelerations(positions: np.ndarray, gms: np.ndarray) -> np.ndarray:
    """Gravitational acceleration for a whole batch of systems at once.

    Args:
        positions: ``(S, N, 3)``.
        gms: ``(S, N)``.

    Returns:
        ``(S, N, 3)`` accelerations.
    """
    separations = positions[:, None, :, :] - positions[:, :, None, :]
    squared = np.einsum("sijk,sijk->sij", separations, separations)

    # Zero the self-interaction without branching, exactly as the scalar version does.
    diagonal = np.arange(positions.shape[1])
    squared[:, diagonal, diagonal] = np.inf

    return np.einsum("sj,sij,sijk->sik", gms, squared**-1.5, separations)


def integrate_batch(
    positions: np.ndarray,
    velocities: np.ndarray,
    gms: np.ndarray,
    steps: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance every system by ``steps`` kick-drift-kick leapfrog steps.

    Identical arithmetic to :func:`orrery.nbody.integrate` with ``integrator=
    "leapfrog"``, carrying a leading batch axis. Held to that implementation by
    ``tests/test_stability.py``.
    """
    half_dt = 0.5 * dt
    acceleration = batch_accelerations(positions, gms)

    for _ in range(steps):
        velocities = velocities + half_dt * acceleration
        positions = positions + dt * velocities
        acceleration = batch_accelerations(positions, gms)
        velocities = velocities + half_dt * acceleration

    return positions, velocities


def _osculating_axes(
    positions: np.ndarray, velocities: np.ndarray, gms: np.ndarray, gm_star: float
) -> np.ndarray:
    """Semi-major axis of each planet relative to the star, shape ``(S, 2)``.

    Negative values mean the orbit is unbound, which is how escapes are detected.
    """
    relative_position = positions[:, 1:, :] - positions[:, 0:1, :]
    relative_velocity = velocities[:, 1:, :] - velocities[:, 0:1, :]

    distance = np.linalg.norm(relative_position, axis=-1)
    speed_squared = np.einsum("snk,snk->sn", relative_velocity, relative_velocity)
    mu = gm_star + gms[:, 1:]

    # Vis-viva rearranged. The denominator vanishes exactly at escape velocity; the
    # tiny epsilon keeps that from raising rather than producing a huge axis.
    denominator = 2.0 / distance - speed_squared / mu
    return 1.0 / np.where(np.abs(denominator) < 1e-30, 1e-30, denominator)


def build_stability_dataset(
    count: int = 3000,
    orbits: float = 1500.0,
    steps_per_orbit: int = 50,
    checks_per_run: int = 40,
    axis_change_threshold: float = 0.20,
    close_encounter_hill_radii: float = 1.0,
    seed: int = 20260808,
    max_inclination_deg: float = 0.0,
    gm_star: float = GM_SUN,
    progress: bool = False,
) -> StabilityDataset:
    """Simulate a batch of systems and label each stable or unstable.

    A system is called **unstable** if any of the following happened during the run:

    * either semi-major axis changed by more than ``axis_change_threshold``,
    * the planets came within ``close_encounter_hill_radii`` mutual Hill radii,
    * either planet became unbound.

    Args:
        count: Number of systems.
        orbits: Length of the run, in orbits of the inner planet.
        steps_per_orbit: Integration steps per inner orbit. Fifty gives a relative
            energy error of roughly ``(1/50)**2 = 4e-4``, second order in the step as
            expected, rising to a few percent for the tightest pairs in the sample.

            That is comfortably safe for the labels even so. Semi-major axis follows
            energy directly, so a 4e-4 energy error is a 4e-4 axis error — three
            orders of magnitude below the 20% change that defines disruption. Even
            the worst systems, at a few percent, stay well clear of the threshold.
        checks_per_run: How many times to pause and record diagnostics.
        axis_change_threshold: Fractional change counting as disruption.
        close_encounter_hill_radii: Approach distance counting as disruption.
        seed: Passed to :func:`sample_two_planet_systems`.
        max_inclination_deg: Tilt between the orbital planes. Zero, the default, is
            the coplanar dataset the models are trained on; a non-zero value produces
            a *shifted* dataset with the same features and a different truth, which
            is what :func:`orrery.models.evaluate_inclination_transfer` uses.
        gm_star: Central mass.
        progress: Print progress, since a full run takes a minute or two.

    Returns:
        A :class:`StabilityDataset` ready for :mod:`orrery.models`.
    """
    sample = sample_two_planet_systems(
        count, seed=seed, max_inclination_deg=max_inclination_deg, gm_star=gm_star
    )
    positions, velocities, gms = _initial_state(sample, gm_star)

    inner_period = 2.0 * np.pi * np.sqrt(INNER_SEMI_MAJOR_AXIS_AU**3 / gm_star)
    dt = inner_period / steps_per_orbit
    total_steps = int(orbits * steps_per_orbit)
    steps_per_check = max(1, total_steps // checks_per_run)

    initial_axes = _osculating_axes(positions, velocities, gms, gm_star)
    max_axis_change = np.zeros(count)
    min_separation = np.full(count, np.inf)
    escaped = np.zeros(count, dtype=bool)

    hill_radius = mutual_hill_radius(
        sample.gm_inner,
        sample.gm_outer,
        sample.semi_major_axis_inner,
        sample.semi_major_axis_outer,
        gm_star,
    )

    completed = 0
    while completed < total_steps:
        chunk = min(steps_per_check, total_steps - completed)
        positions, velocities = integrate_batch(positions, velocities, gms, chunk, dt)
        completed += chunk

        axes = _osculating_axes(positions, velocities, gms, gm_star)
        change = np.max(np.abs(axes - initial_axes) / np.abs(initial_axes), axis=1)
        max_axis_change = np.maximum(max_axis_change, change)
        escaped |= np.any(axes < 0.0, axis=1)

        pair_distance = np.linalg.norm(positions[:, 2, :] - positions[:, 1, :], axis=-1)
        min_separation = np.minimum(min_separation, pair_distance / hill_radius)

        if progress:
            print(
                f"  {completed / total_steps:5.0%}  "
                f"unstable so far: {np.mean(max_axis_change > axis_change_threshold):.1%}"
            )

    unstable = (
        (max_axis_change > axis_change_threshold)
        | (min_separation < close_encounter_hill_radii)
        | escaped
    )

    separation = hill_separation(
        sample.gm_inner,
        sample.gm_outer,
        sample.semi_major_axis_inner,
        sample.semi_major_axis_outer,
        gm_star,
    )
    axis_ratio = sample.semi_major_axis_outer / sample.semi_major_axis_inner
    log_inner = np.log10(sample.gm_inner / gm_star)
    log_outer = np.log10(sample.gm_outer / gm_star)

    # Do the orbits overlap once eccentricity is accounted for? A geometric red flag
    # that Delta alone, being a comparison of *mean* distances, cannot express.
    inner_apocentre = sample.semi_major_axis_inner * (1.0 + sample.eccentricity_inner)
    outer_pericentre = sample.semi_major_axis_outer * (1.0 - sample.eccentricity_outer)
    crossing = (inner_apocentre > outer_pericentre).astype(float)

    features = np.column_stack(
        [
            log_inner,
            log_outer,
            sample.gm_outer / sample.gm_inner,
            axis_ratio,
            separation,
            sample.eccentricity_inner,
            sample.eccentricity_outer,
            np.maximum(sample.eccentricity_inner, sample.eccentricity_outer),
            crossing,
        ]
    )

    return StabilityDataset(
        features=features,
        labels=(~unstable).astype(int),
        hill_separation=separation,
        max_axis_change=max_axis_change,
        min_separation_hill=min_separation,
        escaped=escaped,
        orbits_simulated=orbits,
        sample=sample,
    )


# ---------------------------------------------------------------------------
# Three or more planets: where pairwise reasoning stops working
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiPlanetSample:
    """A batch of systems with an arbitrary number of coplanar planets.

    Arrays are shape ``(S, P)`` for ``S`` systems of ``P`` planets, ordered outward.
    Kept separate from :class:`SystemSample` rather than replacing it, because the
    two-planet path carries the phase 5 model's feature definitions and there is no
    reason to disturb numbers that are already reported.
    """

    gms: np.ndarray
    semi_major_axes: np.ndarray
    eccentricities: np.ndarray
    phases: np.ndarray
    pericentres: np.ndarray
    #: Orbital inclination of each planet to the reference plane, radians. All zero
    #: for a coplanar draw, which is the default and reproduces the earlier study
    #: exactly.
    inclinations: np.ndarray
    #: Longitude of the ascending node, radians. Meaningless when the inclination is
    #: zero, and drawn anyway so the coplanar case stays a special case of the general
    #: one rather than a separate code path.
    nodes: np.ndarray

    def __len__(self) -> int:
        return self.gms.shape[0]

    @property
    def planet_count(self) -> int:
        return self.gms.shape[1]

    def adjacent_hill_separations(self, gm_star: float = GM_SUN) -> np.ndarray:
        """``(S, P-1)`` separation of each neighbouring pair, in mutual Hill radii."""
        return hill_separation(
            self.gms[:, :-1],
            self.gms[:, 1:],
            self.semi_major_axes[:, :-1],
            self.semi_major_axes[:, 1:],
            gm_star,
        )

    def orbit_normals(self) -> np.ndarray:
        r"""``(S, P, 3)`` unit vector perpendicular to each orbital plane.

        Follows from the inclination and node alone --- where the planet sits on its
        ellipse does not tilt the plane:

        .. math::

            \hat{n} = (\sin i \sin \Omega,\; -\sin i \cos \Omega,\; \cos i)
        """
        sin_i, cos_i = np.sin(self.inclinations), np.cos(self.inclinations)
        return np.stack(
            [sin_i * np.sin(self.nodes), -sin_i * np.cos(self.nodes), cos_i], axis=-1
        )

    def mutual_inclinations(self) -> np.ndarray:
        """``(S, P-1)`` angle between neighbouring orbital planes, radians.

        This is the quantity that matters dynamically, not each planet's inclination
        to some arbitrary reference plane. Two orbits both tilted 20 degrees can be
        mutually coplanar or mutually inclined by 40, depending on their nodes.
        """
        normals = self.orbit_normals()
        dot = np.einsum("spk,spk->sp", normals[:, :-1, :], normals[:, 1:, :])
        return np.arccos(np.clip(dot, -1.0, 1.0))


@dataclass(frozen=True)
class MultiPlanetDataset:
    """Survival of ``P``-planet systems, indexed by their tightest pair.

    Attributes:
        planet_count: How many planets each system had.
        min_hill_separation: ``(S,)`` the smallest adjacent separation --- the quantity
            a pairwise criterion would judge the system on.
        max_mutual_inclination: ``(S,)`` the largest angle between neighbouring
            orbital planes, in **degrees**. Zero for a coplanar run.
        labels: ``(S,)`` 1 for stable, 0 for disrupted.
        max_axis_change: ``(S,)`` largest fractional change in any semi-major axis.
        escaped: ``(S,)`` whether any planet became unbound.
        orbits_simulated: Length of the run, in orbits of the innermost planet.
        sample: The underlying draw.
    """

    planet_count: int
    min_hill_separation: np.ndarray
    max_mutual_inclination: np.ndarray
    labels: np.ndarray
    max_axis_change: np.ndarray
    escaped: np.ndarray
    orbits_simulated: float
    sample: MultiPlanetSample

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def stable_fraction(self) -> float:
        return float(np.mean(self.labels))

    def survival_by_separation(
        self, edges: np.ndarray, minimum_per_bin: int = 10
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Survival rate binned by the tightest adjacent separation.

        Returns:
            ``(centres, rate, count)``, keeping only bins holding enough systems to
            mean anything.
        """
        return self._survival_binned(self.min_hill_separation, edges, minimum_per_bin)

    def survival_by_inclination(
        self, edges: np.ndarray, minimum_per_bin: int = 10
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Survival rate binned by the largest mutual inclination, in degrees."""
        return self._survival_binned(self.max_mutual_inclination, edges, minimum_per_bin)

    def _survival_binned(
        self, quantity: np.ndarray, edges: np.ndarray, minimum_per_bin: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centres, rates, counts = [], [], []
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            inside = (quantity >= low) & (quantity < high)
            if inside.sum() >= minimum_per_bin:
                centres.append(0.5 * (low + high))
                rates.append(float(self.labels[inside].mean()))
                counts.append(int(inside.sum()))
        return np.array(centres), np.array(rates), np.array(counts)


def sample_planet_systems(
    count: int,
    planets: int = 2,
    seed: int = 20260808,
    hill_separation_range: tuple[float, float] = (2.0, 14.0),
    log_mass_range: tuple[float, float] = (-6.0, -4.0),
    max_eccentricity: float = 0.05,
    max_inclination_deg: float = 0.0,
    gm_star: float = GM_SUN,
) -> MultiPlanetSample:
    """Draw random systems of ``planets`` coplanar planets.

    Each **adjacent** pair gets its own separation drawn from
    ``hill_separation_range``, and the semi-major axes are built outward from the
    innermost. Sampling separations rather than axes keeps the draws concentrated where
    the stability boundary lives, exactly as in the two-planet case.

    The default mass and eccentricity ranges are narrower than
    :func:`sample_two_planet_systems` uses. That is deliberate for the comparison in
    :func:`build_multiplanet_dataset`: with heavy or eccentric planets a three-planet
    system is unstable almost everywhere, and the effect worth seeing --- pairs that are
    individually fine failing collectively --- would be buried under it.

    Args:
        count: Number of systems.
        planets: Planets per system, at least two.
        seed: Fixed for reproducibility.
        hill_separation_range: Range for each adjacent separation.
        log_mass_range: ``log10`` of planet mass relative to the star.
        max_eccentricity: Eccentricities drawn uniformly in ``[0, this]``.
        max_inclination_deg: Each planet's inclination to the reference plane is drawn
            uniformly in ``[0, this]``, with a random node. **Zero by default**, which
            reproduces the coplanar study exactly rather than approximately --- the
            rotation below collapses to the old in-plane one when the inclination and
            node vanish.

            Note that this is each planet's inclination to a shared reference plane,
            not the mutual angle between orbits. Two planets tilted by 20 degrees each
            can be mutually coplanar or mutually inclined by 40, depending on their
            nodes; :meth:`MultiPlanetSample.mutual_inclinations` reports what actually
            came out.
        gm_star: Central mass.

    Raises:
        ValueError: If fewer than two planets, a non-positive count, or an inclination
            outside ``[0, 180]`` is requested.
    """
    if planets < 2:
        raise ValueError(f"need at least two planets to have a separation; got {planets}")
    if count < 1:
        raise ValueError(f"count must be positive; got {count}")
    if not 0.0 <= max_inclination_deg <= 180.0:
        raise ValueError(
            f"max_inclination_deg must be in [0, 180]; got {max_inclination_deg}"
        )

    generator = np.random.default_rng(seed)

    gms = gm_star * 10.0 ** generator.uniform(*log_mass_range, size=(count, planets))
    separations = generator.uniform(*hill_separation_range, size=(count, planets - 1))

    axes = np.empty((count, planets))
    axes[:, 0] = INNER_SEMI_MAJOR_AXIS_AU

    # The same inversion as the two-planet sampler, applied outward pair by pair:
    #   a_{k+1} (1 - D k / 2) = a_k (1 + D k / 2),  k = ((m_k + m_{k+1}) / 3 M*)^(1/3)
    for index in range(planets - 1):
        mass_factor = ((gms[:, index] + gms[:, index + 1]) / (3.0 * gm_star)) ** (1.0 / 3.0)
        half = 0.5 * separations[:, index] * mass_factor
        axes[:, index + 1] = axes[:, index] * (1.0 + half) / (1.0 - half)

    # Both angles are skipped entirely, not drawn-then-zeroed, when the run is coplanar.
    # ``uniform(0, 0)`` returns zeros but still advances the generator, which would shift
    # every draw below it and silently resample the coplanar study --- the earlier
    # published three-planet numbers came from a stream that had no inclination in it.
    # Skipping keeps the coplanar case a byte-for-byte special case of the general one.
    # A zero inclination also makes the node meaningless, so pinning it avoids suggesting
    # the coplanar draws differ from one another when they do not.
    if max_inclination_deg > 0.0:
        inclinations = np.radians(
            generator.uniform(0.0, max_inclination_deg, size=(count, planets))
        )
        nodes = generator.uniform(0.0, 2.0 * np.pi, size=(count, planets))
    else:
        inclinations = np.zeros((count, planets))
        nodes = np.zeros((count, planets))

    return MultiPlanetSample(
        gms=gms,
        semi_major_axes=axes,
        eccentricities=generator.uniform(0.0, max_eccentricity, size=(count, planets)),
        phases=generator.uniform(0.0, 2.0 * np.pi, size=(count, planets)),
        pericentres=generator.uniform(0.0, 2.0 * np.pi, size=(count, planets)),
        inclinations=inclinations,
        nodes=nodes,
    )


def _multiplanet_initial_state(
    sample: MultiPlanetSample, gm_star: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Barycentric state for a multi-planet batch.

    Returns shapes ``(S, P+1, 3)``, ``(S, P+1, 3)`` and ``(S, P+1)``, with body 0 the
    star.
    """
    count, planets = sample.gms.shape
    bodies = planets + 1

    positions = np.zeros((count, bodies, 3))
    velocities = np.zeros((count, bodies, 3))
    gms = np.concatenate([np.full((count, 1), gm_star), sample.gms], axis=1)

    for index in range(planets):
        axis = sample.semi_major_axes[:, index]
        eccentricity = sample.eccentricities[:, index]
        phase = sample.phases[:, index]
        pericentre = sample.pericentres[:, index]
        inclination = sample.inclinations[:, index]
        node = sample.nodes[:, index]

        semi_latus_rectum = axis * (1.0 - eccentricity**2)
        radius = semi_latus_rectum / (1.0 + eccentricity * np.cos(phase))
        speed_scale = np.sqrt(gm_star / semi_latus_rectum)

        cos_phase, sin_phase = np.cos(phase), np.sin(phase)
        x_perifocal = radius * cos_phase
        y_perifocal = radius * sin_phase
        vx_perifocal = -speed_scale * sin_phase
        vy_perifocal = speed_scale * (eccentricity + cos_phase)

        # Rz(node) . Rx(inclination) . Rz(pericentre), written out --- the same
        # rotation orrery/ephemeris.py uses for the planets. With inclination and node
        # both zero it collapses to a plain in-plane rotation by the pericentre, so
        # the coplanar case is reproduced exactly rather than approximately.
        cos_w, sin_w = np.cos(pericentre), np.sin(pericentre)
        cos_i, sin_i = np.cos(inclination), np.sin(inclination)
        cos_n, sin_n = np.cos(node), np.sin(node)

        m00 = cos_w * cos_n - sin_w * sin_n * cos_i
        m01 = -sin_w * cos_n - cos_w * sin_n * cos_i
        m10 = cos_w * sin_n + sin_w * cos_n * cos_i
        m11 = -sin_w * sin_n + cos_w * cos_n * cos_i
        m20 = sin_w * sin_i
        m21 = cos_w * sin_i

        body = index + 1
        positions[:, body, 0] = m00 * x_perifocal + m01 * y_perifocal
        positions[:, body, 1] = m10 * x_perifocal + m11 * y_perifocal
        positions[:, body, 2] = m20 * x_perifocal + m21 * y_perifocal
        velocities[:, body, 0] = m00 * vx_perifocal + m01 * vy_perifocal
        velocities[:, body, 1] = m10 * vx_perifocal + m11 * vy_perifocal
        velocities[:, body, 2] = m20 * vx_perifocal + m21 * vy_perifocal

    total_gm = gms.sum(axis=1)
    positions -= (np.einsum("sn,snk->sk", gms, positions) / total_gm[:, None])[:, None, :]
    velocities -= (np.einsum("sn,snk->sk", gms, velocities) / total_gm[:, None])[:, None, :]

    return positions, velocities, gms


def build_multiplanet_dataset(
    count: int = 1200,
    planets: int = 2,
    orbits: float = 2000.0,
    steps_per_orbit: int = 40,
    checks_per_run: int = 30,
    axis_change_threshold: float = 0.20,
    seed: int = 20260808,
    gm_star: float = GM_SUN,
    **sample_options,
) -> MultiPlanetDataset:
    """Simulate systems of ``planets`` planets and label each stable or disrupted.

    The point of allowing ``planets > 2`` is that pairwise stability criteria stop
    working there. Two planets destabilise by approaching each other, which the Hill
    separation predicts well. Three destabilise mainly through **resonance overlap**:
    each pair can sit comfortably outside its own Hill limit while resonances belonging
    to *different* pairs overlap in between, opening a chaotic band that no pairwise
    number can see.

    Running this at ``planets=2`` and ``planets=3`` with the same separation range and
    the same integration length isolates that effect, because the only difference
    between the two runs is the presence of a third body.

    Args:
        count: Number of systems.
        planets: Planets per system.
        orbits: Length of the run, in orbits of the innermost planet.
        steps_per_orbit: Integration steps per innermost orbit.
        checks_per_run: How often to pause and record diagnostics.
        axis_change_threshold: Fractional change counting as disruption.
        seed: Passed to :func:`sample_planet_systems`.
        gm_star: Central mass.
        **sample_options: Forwarded to :func:`sample_planet_systems`.
    """
    sample = sample_planet_systems(
        count, planets=planets, seed=seed, gm_star=gm_star, **sample_options
    )
    positions, velocities, gms = _multiplanet_initial_state(sample, gm_star)

    inner_period = 2.0 * np.pi * np.sqrt(INNER_SEMI_MAJOR_AXIS_AU**3 / gm_star)
    dt = inner_period / steps_per_orbit
    total_steps = int(orbits * steps_per_orbit)
    steps_per_check = max(1, total_steps // checks_per_run)

    initial_axes = _osculating_axes(positions, velocities, gms, gm_star)
    max_axis_change = np.zeros(count)
    escaped = np.zeros(count, dtype=bool)

    completed = 0
    while completed < total_steps:
        chunk = min(steps_per_check, total_steps - completed)
        positions, velocities = integrate_batch(positions, velocities, gms, chunk, dt)
        completed += chunk

        axes = _osculating_axes(positions, velocities, gms, gm_star)
        change = np.max(np.abs(axes - initial_axes) / np.abs(initial_axes), axis=1)
        max_axis_change = np.maximum(max_axis_change, change)
        escaped |= np.any(axes < 0.0, axis=1)

    unstable = (max_axis_change > axis_change_threshold) | escaped

    return MultiPlanetDataset(
        planet_count=planets,
        min_hill_separation=sample.adjacent_hill_separations(gm_star).min(axis=1),
        max_mutual_inclination=np.degrees(sample.mutual_inclinations().max(axis=1)),
        labels=(~unstable).astype(int),
        max_axis_change=max_axis_change,
        escaped=escaped,
        orbits_simulated=orbits,
        sample=sample,
    )
