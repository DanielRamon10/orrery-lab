"""Tests for the stability dataset generator and the batched integrator.

The load-bearing test here is :class:`TestBatchIntegratorParity`. The stability study
needs to integrate thousands of systems, which the phase 3 integrator cannot do at a
usable speed, so :mod:`orrery.stability` carries a second, batched leapfrog. A fast
reimplementation of physics that already exists is only safe if something holds it to
the original — the same reasoning that produced the Python/TypeScript ephemeris parity
fixture in phase 2.

The rest check that the generated labels behave like orbital dynamics rather than like
a random number generator: instability must increase as separation shrinks, the
analytic Hill criterion must show up in the labels, and the physics must not depend on
irrelevant choices such as where the planets happen to start on their orbits.
"""

from __future__ import annotations

import numpy as np
import pytest

from orrery.constants import GM_BODY, GM_SUN
from orrery.nbody import integrate, total_energy
from orrery.stability import (
    FEATURE_NAMES,
    HILL_STABILITY_THRESHOLD,
    _initial_state,
    batch_accelerations,
    build_stability_dataset,
    hill_separation,
    integrate_batch,
    mutual_hill_radius,
    sample_two_planet_systems,
)

# Small and short: these tests check correctness, not statistical power.
SMALL = {"count": 240, "orbits": 120.0}


@pytest.fixture(scope="module")
def dataset():
    """One modest dataset, reused across the statistical checks."""
    return build_stability_dataset(**SMALL, seed=7)


class TestHillGeometry:
    def test_mutual_hill_radius_matches_the_formula(self):
        gm_inner = np.array([GM_BODY["earth"]])
        gm_outer = np.array([GM_BODY["earth"]])
        a_inner, a_outer = np.array([1.0]), np.array([1.5])

        expected = ((2 * GM_BODY["earth"]) / (3 * GM_SUN)) ** (1 / 3) * 1.25
        result = mutual_hill_radius(gm_inner, gm_outer, a_inner, a_outer)

        assert result[0] == pytest.approx(expected)

    def test_heavier_planets_have_a_wider_hill_radius(self):
        light = mutual_hill_radius(
            np.array([1e-9]), np.array([1e-9]), np.array([1.0]), np.array([2.0])
        )
        heavy = mutual_hill_radius(
            np.array([1e-6]), np.array([1e-6]), np.array([1.0]), np.array([2.0])
        )
        assert heavy[0] > light[0]

    def test_sampling_hits_the_requested_separations(self):
        """The sampler inverts Delta for a2; the inversion has to round-trip."""
        wanted = (2.0, 10.0)
        sample = sample_two_planet_systems(500, seed=3, hill_separation_range=wanted)

        actual = hill_separation(
            sample.gm_inner,
            sample.gm_outer,
            sample.semi_major_axis_inner,
            sample.semi_major_axis_outer,
        )
        assert actual.min() >= wanted[0] - 1e-9
        assert actual.max() <= wanted[1] + 1e-9
        # And the range is actually covered, not bunched at one end.
        assert actual.min() < 3.0 and actual.max() > 9.0

    def test_outer_planet_is_always_outside(self):
        sample = sample_two_planet_systems(400, seed=11)
        assert np.all(sample.semi_major_axis_outer > sample.semi_major_axis_inner)


class TestBatchIntegratorParity:
    """Hold the batched leapfrog to the phase 3 implementation."""

    def test_accelerations_match_the_scalar_version(self):
        from orrery.nbody import accelerations

        sample = sample_two_planet_systems(6, seed=5)
        positions, _, gms = _initial_state(sample, GM_SUN)

        batched = batch_accelerations(positions, gms)
        for index in range(len(sample)):
            single = accelerations(positions[index], gms[index])
            np.testing.assert_allclose(batched[index], single, rtol=1e-13, atol=1e-20)

    def test_trajectories_match_the_scalar_integrator(self):
        """Same arithmetic, so the two must agree to round-off over many steps."""
        sample = sample_two_planet_systems(4, seed=13, hill_separation_range=(6.0, 10.0))
        positions, velocities, gms = _initial_state(sample, GM_SUN)

        steps, dt = 400, 3.0
        batch_positions, batch_velocities = integrate_batch(
            positions.copy(), velocities.copy(), gms, steps, dt
        )

        for index in range(len(sample)):
            reference = integrate(
                positions[index],
                velocities[index],
                gms[index],
                duration_days=steps * dt,
                dt=dt,
                integrator="leapfrog",
                sample_every=steps,
            )
            np.testing.assert_allclose(
                batch_positions[index], reference.positions[-1], rtol=1e-11, atol=1e-14
            )
            np.testing.assert_allclose(
                batch_velocities[index], reference.velocities[-1], rtol=1e-11, atol=1e-14
            )

    @staticmethod
    def _energy_drift(dt: float, steps: int) -> np.ndarray:
        sample = sample_two_planet_systems(30, seed=17, hill_separation_range=(8.0, 12.0))
        positions, velocities, gms = _initial_state(sample, GM_SUN)

        before = np.array(
            [total_energy(positions[i], velocities[i], gms[i]) for i in range(len(sample))]
        )
        positions, velocities = integrate_batch(positions, velocities, gms, steps, dt)
        after = np.array(
            [total_energy(positions[i], velocities[i], gms[i]) for i in range(len(sample))]
        )
        return np.abs((after - before) / before)

    def test_energy_drift_is_second_order_in_the_step(self):
        """Halve the step, quarter the error — the property, not a guessed number.

        An absolute tolerance here would be a guess, and the first one written was
        wrong by two orders of magnitude: at 73 steps per orbit the median drift is
        about 2.4e-4, which is simply ``(dt/P)**2`` for a second-order method. The
        meaningful assertion is that the error *scales* the way leapfrog must.
        """
        coarse = np.median(self._energy_drift(dt=5.0, steps=2000))
        fine = np.median(self._energy_drift(dt=2.5, steps=4000))

        assert coarse / fine == pytest.approx(4.0, rel=0.4), (
            f"expected fourfold improvement; got {coarse:.2e} -> {fine:.2e}"
        )

    def test_energy_drift_is_far_below_the_label_threshold(self):
        """What actually matters: integration error cannot flip a stability label.

        Semi-major axis follows energy directly, so a relative energy error is a
        relative axis error. Disruption is defined as a 20% axis change, so the drift
        has to stay far below that even for the worst system in the sample.
        """
        drift = self._energy_drift(dt=5.0, steps=2000)

        assert np.median(drift) < 1e-3
        # The tightest pairs are the noisiest, but still an order below the threshold.
        assert np.max(drift) < 0.05


class TestInitialConditions:
    def test_barycentre_starts_at_rest_at_the_origin(self):
        sample = sample_two_planet_systems(50, seed=19)
        positions, velocities, gms = _initial_state(sample, GM_SUN)

        centre = np.einsum("sn,snk->sk", gms, positions) / gms.sum(axis=1)[:, None]
        momentum = np.einsum("sn,snk->sk", gms, velocities)

        assert np.max(np.abs(centre)) < 1e-15
        assert np.max(np.abs(momentum)) < 1e-18

    def test_planets_start_on_their_intended_ellipses(self):
        """Distance from the star must lie between pericentre and apocentre."""
        sample = sample_two_planet_systems(200, seed=23)
        positions, _, _ = _initial_state(sample, GM_SUN)

        for index, (axis, eccentricity) in enumerate(
            (
                (sample.semi_major_axis_inner, sample.eccentricity_inner),
                (sample.semi_major_axis_outer, sample.eccentricity_outer),
            ),
            start=1,
        ):
            distance = np.linalg.norm(positions[:, index, :] - positions[:, 0, :], axis=-1)
            assert np.all(distance >= axis * (1 - eccentricity) - 1e-9)
            assert np.all(distance <= axis * (1 + eccentricity) + 1e-9)

    def test_motion_is_confined_to_a_plane(self):
        sample = sample_two_planet_systems(50, seed=29)
        positions, velocities, _ = _initial_state(sample, GM_SUN)
        assert np.max(np.abs(positions[..., 2])) == 0.0
        assert np.max(np.abs(velocities[..., 2])) == 0.0


class TestLabelsBehaveLikeDynamics:
    def test_stability_increases_with_separation(self, dataset):
        """The single most basic sanity check on the labels."""
        wide = dataset.hill_separation > 8.0
        narrow = dataset.hill_separation < 3.0

        assert dataset.labels[wide].mean() > dataset.labels[narrow].mean() + 0.4

    def test_the_analytic_criterion_shows_up_in_the_labels(self, dataset):
        below = dataset.hill_separation < HILL_STABILITY_THRESHOLD
        assert dataset.labels[below].mean() < 0.5
        assert dataset.labels[~below].mean() > 0.7

    def test_gladman_is_sufficient_not_necessary(self, dataset):
        """Above the threshold is not a guarantee, because these orbits are eccentric.

        Gladman's criterion is derived for circular coplanar orbits. This dataset has
        eccentricities up to 0.15, so some systems above the threshold must still fail
        — if none did, the sampler would not be exploring the interesting regime.
        """
        above = dataset.hill_separation > HILL_STABILITY_THRESHOLD
        assert dataset.labels[above].mean() < 1.0

    def test_both_classes_are_well_represented(self, dataset):
        assert 0.2 < dataset.stable_fraction < 0.9

    def test_unstable_systems_show_a_physical_cause(self, dataset):
        """Every unstable label must be traceable to a diagnostic, not to noise."""
        unstable = dataset.labels == 0
        cause = (
            (dataset.max_axis_change > 0.20)
            | (dataset.min_separation_hill < 1.0)
            | dataset.escaped
        )
        assert np.all(cause[unstable])

    def test_stable_systems_kept_their_orbits(self, dataset):
        stable = dataset.labels == 1
        assert np.all(dataset.max_axis_change[stable] <= 0.20)
        assert not np.any(dataset.escaped[stable])

    def test_starting_phase_does_not_decide_the_outcome(self, dataset):
        """A well-posed dataset should not be predictable from an arbitrary choice.

        The initial true anomalies are a free choice of when the clock starts. They
        can matter for an individual marginal system, but if they predicted stability
        overall the setup would be wrong.
        """
        phase = dataset.sample.phase_inner
        first_half = dataset.labels[phase < np.pi].mean()
        second_half = dataset.labels[phase >= np.pi].mean()

        assert abs(first_half - second_half) < 0.12


class TestDatasetShape:
    def test_feature_matrix_matches_the_declared_names(self, dataset):
        assert dataset.features.shape == (len(dataset), len(FEATURE_NAMES))
        assert np.all(np.isfinite(dataset.features))

    def test_hill_separation_column_matches_the_standalone_function(self, dataset):
        column = dataset.features[:, FEATURE_NAMES.index("hill_separation")]
        np.testing.assert_allclose(column, dataset.hill_separation, rtol=1e-12)

    def test_labels_are_binary(self, dataset):
        assert set(np.unique(dataset.labels)).issubset({0, 1})

    def test_reproducible_for_a_fixed_seed(self):
        first = build_stability_dataset(count=60, orbits=60.0, seed=99)
        second = build_stability_dataset(count=60, orbits=60.0, seed=99)
        np.testing.assert_array_equal(first.labels, second.labels)

    def test_rejects_empty_request(self):
        with pytest.raises(ValueError, match="count must be positive"):
            sample_two_planet_systems(0)
