"""Tests for the N-body integrators.

The interesting claims in :mod:`orrery.nbody` are all quantitative, so they are all
measured here rather than asserted in a docstring:

* the forces obey Newton's third law,
* each integrator really achieves its stated order of accuracy, checked by halving
  the step and watching the error fall by the predicted factor,
* symplectic integrators keep energy in a **bounded band** while RK4 drifts
  secularly --- the central claim of the module,
* leapfrog is exactly time-reversible,
* momentum is conserved to round-off, not merely approximately.

Ground truth comes from phase 1: the two-body problem has an exact solution, and
:func:`orrery.kepler.solve_kepler` provides it. So the exact solver validates the
approximate one, and the two halves of the project check each other.
"""

from __future__ import annotations

import numpy as np
import pytest

from orrery.constants import GM_BODY, GM_SUN, J2000_JD
from orrery.initial_conditions import (
    report_frame_quality,
    solar_system_state,
    two_body_period_days,
    two_body_state,
)
from orrery.kepler import mean_motion, solve_kepler
from orrery.nbody import (
    INTEGRATORS,
    accelerations,
    centre_of_mass,
    integrate,
    total_angular_momentum,
    total_energy,
    total_linear_momentum,
)

# A companion heavy enough that the system's energy is not identically zero, so
# relative drift is well defined. Roughly Jupiter's mass.
COMPANION_GM = GM_BODY["jupiter"]


def exact_relative_position(
    semi_major_axis: float,
    eccentricity: float,
    gm_total: float,
    elapsed_days: float,
) -> np.ndarray:
    """Exact two-body separation vector, from the Kepler solver.

    Valid because :func:`two_body_state` starts the pair at perihelion, so the mean
    anomaly is simply ``n * t`` with no phase offset.
    """
    anomaly = mean_motion(semi_major_axis, gm_total) * elapsed_days
    eccentric = float(solve_kepler(anomaly, eccentricity))
    return np.array(
        [
            semi_major_axis * (np.cos(eccentric) - eccentricity),
            semi_major_axis * np.sqrt(1.0 - eccentricity**2) * np.sin(eccentric),
            0.0,
        ]
    )


def separation_error(trajectory, semi_major_axis, eccentricity, gm_total) -> float:
    """Distance between the integrated and exact separation at the final sample."""
    central = trajectory.positions[-1, 0, :]
    orbiter = trajectory.positions[-1, 1, :]
    exact = exact_relative_position(
        semi_major_axis, eccentricity, gm_total, float(trajectory.times[-1])
    )
    return float(np.linalg.norm((orbiter - central) - exact))


class TestAccelerations:
    def test_two_body_matches_newtons_law(self):
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        gms = np.array([GM_SUN, COMPANION_GM])

        acceleration = accelerations(positions, gms)

        # Each body is pulled toward the other, with magnitude GM_other / r^2.
        assert acceleration[0, 0] == pytest.approx(COMPANION_GM / 4.0)
        assert acceleration[1, 0] == pytest.approx(-GM_SUN / 4.0)
        # Nothing off-axis for a pair on the x-axis.
        np.testing.assert_allclose(acceleration[:, 1:], 0.0, atol=1e-18)

    def test_newtons_third_law(self):
        """Momentum-weighted accelerations must cancel: sum of GM_i * a_i = 0.

        This is the internal-forces-cancel identity, and it is why the barycentre
        cannot accelerate. If this failed, no integrator could conserve momentum.
        """
        rng = np.random.default_rng(20260726)
        positions = rng.normal(scale=5.0, size=(6, 3))
        gms = np.array([GM_SUN, *list(GM_BODY.values())[:5]])

        acceleration = accelerations(positions, gms)
        np.testing.assert_allclose(
            np.einsum("i,ij->j", gms, acceleration), 0.0, atol=1e-20
        )

    def test_no_self_interaction(self):
        """A lone body must feel nothing, rather than dividing by zero."""
        acceleration = accelerations(np.array([[1.0, 2.0, 3.0]]), np.array([GM_SUN]))
        np.testing.assert_allclose(acceleration, 0.0)

    def test_softening_reduces_force_at_close_range(self):
        positions = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
        gms = np.array([GM_SUN, COMPANION_GM])

        hard = accelerations(positions, gms)
        soft = accelerations(positions, gms, softening=0.05)

        assert abs(soft[0, 0]) < abs(hard[0, 0])


class TestConvergenceOrder:
    """Halve the step; the error should fall by 2**order.

    This is the sharpest available check that each stepper is implemented correctly:
    a scheme with a subtly wrong coefficient still converges, but converges at the
    wrong rate, and no amount of eyeballing a trajectory would reveal it.
    """

    SEMI_MAJOR_AXIS = 1.0
    ECCENTRICITY = 0.2

    @pytest.mark.parametrize(
        ("integrator", "expected_order"),
        [(name, order) for name, (_, _, order) in INTEGRATORS.items()],
    )
    def test_observed_order_matches_the_claim(self, integrator, expected_order):
        gm_total = GM_SUN + COMPANION_GM
        period = two_body_period_days(self.SEMI_MAJOR_AXIS, GM_SUN, COMPANION_GM)

        # Euler is so inaccurate that a full orbit at a usable step size leaves the
        # asymptotic regime entirely, so it is measured over a shorter arc.
        span = period * (0.1 if integrator == "euler" else 1.0)
        coarse_dt = 0.4 if integrator == "euler" else 2.0

        errors = []
        for dt in (coarse_dt, coarse_dt / 2.0):
            state = two_body_state(
                self.SEMI_MAJOR_AXIS, self.ECCENTRICITY, GM_SUN, COMPANION_GM
            )
            trajectory = integrate(
                *state.as_arrays(),
                duration_days=span,
                dt=dt,
                integrator=integrator,
                sample_every=10_000_000,  # only the endpoints matter here
            )
            errors.append(
                separation_error(
                    trajectory, self.SEMI_MAJOR_AXIS, self.ECCENTRICITY, gm_total
                )
            )

        observed_ratio = errors[0] / errors[1]
        expected_ratio = 2.0**expected_order

        # A wide band: the asymptotic rate is only approached, not attained, and the
        # neighbouring orders differ by a factor of two, so this still discriminates.
        assert 0.55 * expected_ratio < observed_ratio < 1.9 * expected_ratio, (
            f"{integrator}: expected error ratio near {expected_ratio:.1f}, "
            f"observed {observed_ratio:.2f} (errors {errors[0]:.3e} -> {errors[1]:.3e})"
        )


class TestSymplecticVersusNot:
    """The central claim: bounded oscillation versus unbounded drift.

    A step size deliberately on the large side, over many orbits, so the secular
    behaviour dominates round-off and the distinction is unmistakable.
    """

    SEMI_MAJOR_AXIS = 1.0
    ECCENTRICITY = 0.3
    ORBITS = 60
    DT_DAYS = 3.0

    def _run(self, integrator: str):
        state = two_body_state(
            self.SEMI_MAJOR_AXIS, self.ECCENTRICITY, GM_SUN, COMPANION_GM
        )
        period = two_body_period_days(self.SEMI_MAJOR_AXIS, GM_SUN, COMPANION_GM)
        return integrate(
            *state.as_arrays(),
            duration_days=period * self.ORBITS,
            dt=self.DT_DAYS,
            integrator=integrator,
            sample_every=5,
        )

    @pytest.mark.parametrize("integrator", ["leapfrog", "yoshida4"])
    def test_symplectic_energy_error_stays_bounded(self, integrator):
        drift = self._run(integrator).relative_energy_error
        first_half = drift[: len(drift) // 2].max()
        second_half = drift[len(drift) // 2 :].max()

        # Bounded means the late maximum is no worse than the early one, give or
        # take: the error oscillates rather than accumulating.
        assert second_half < 2.0 * max(first_half, 1e-16), (
            f"{integrator}: energy error grew from {first_half:.3e} to {second_half:.3e}"
        )

    def test_rk4_energy_error_grows_secularly(self):
        """RK4 conserves nothing exactly, so its energy error accumulates.

        Stated as an assertion so the comparison in the documentation is a measured
        fact and would fail loudly if it ever stopped being true.
        """
        drift = self._run("rk4").relative_energy_error
        first_half = drift[: len(drift) // 2].max()
        second_half = drift[len(drift) // 2 :].max()

        assert second_half > 1.5 * first_half, (
            f"expected secular growth; got {first_half:.3e} -> {second_half:.3e}"
        )

    def test_rk4_is_more_accurate_than_leapfrog_at_this_span(self):
        """The nuance that makes the symplectic argument honest.

        Over sixty orbits RK4 is *more* accurate in absolute terms than leapfrog ---
        fourth order beats second order, as it should. The symplectic advantage is
        not "always smaller error"; it is "error that never grows". So leapfrog wins
        eventually, and this test pins down that at this span it has not yet won,
        which keeps the documentation from overclaiming.
        """
        leapfrog = self._run("leapfrog").relative_energy_error.max()
        rk4 = self._run("rk4").relative_energy_error.max()

        assert rk4 < leapfrog, (
            "expected RK4 to still be ahead on absolute error at this span; "
            f"leapfrog {leapfrog:.3e}, rk4 {rk4:.3e}"
        )

    def test_rk4_growth_is_steady_enough_to_extrapolate_a_crossover(self):
        """RK4's drift grows roughly linearly, so the crossover is predictable.

        This is the mechanism behind the previous test's caveat: leapfrog's band is
        flat while RK4's grows, so a finite integration length exists past which
        leapfrog is the more accurate choice. Measuring the growth rate lets the
        README state that length instead of hand-waving about "long enough".
        """
        rk4 = self._run("rk4").relative_energy_error
        quarter = len(rk4) // 4

        early = rk4[quarter] - rk4[0]
        late = rk4[-1] - rk4[3 * quarter]

        # Same elapsed time in each window, so equal increments mean linear growth.
        assert early > 0 and late > 0
        assert 0.5 < late / early < 2.0, f"growth not steady: {early:.3e} vs {late:.3e}"

    def test_euler_is_visibly_unstable(self):
        drift = self._run("euler").relative_energy_error
        assert drift[-1] > 1e-3, "explicit Euler should degrade obviously"


class TestConservation:
    def _state(self):
        return two_body_state(1.0, 0.4, GM_SUN, COMPANION_GM)

    @pytest.mark.parametrize("integrator", sorted(INTEGRATORS))
    def test_linear_momentum_is_conserved_to_round_off(self, integrator):
        """Every scheme here conserves momentum exactly, by construction.

        Both the kick and the drift add the *same* increment to every body, and the
        internal forces cancel, so momentum error can only come from floating-point
        summation --- never from the discretisation.
        """
        state = self._state()
        trajectory = integrate(
            *state.as_arrays(), duration_days=400.0, dt=1.0, integrator=integrator
        )

        magnitudes = np.linalg.norm(trajectory.linear_momentum, axis=-1)
        scale = float(np.sum(state.gms * np.linalg.norm(state.velocities, axis=-1)))
        assert magnitudes.max() < 1e-12 * scale

    @pytest.mark.parametrize(
        ("integrator", "tolerance"),
        [
            # The symplectic pair hold angular momentum at round-off level.
            ("leapfrog", 1e-13),
            ("yoshida4", 1e-13),
            # RK4 conserves it only approximately — it has no structural reason to.
            ("rk4", 1e-8),
        ],
    )
    def test_angular_momentum_is_conserved(self, integrator, tolerance):
        state = self._state()
        trajectory = integrate(
            *state.as_arrays(), duration_days=400.0, dt=1.0, integrator=integrator
        )
        assert trajectory.relative_angular_momentum_error.max() < tolerance

    def test_euler_does_not_conserve_angular_momentum(self):
        """Explicit Euler fails even this, and the failure is not subtle.

        Its kick and drift use mismatched times — the position update uses the old
        velocity while the velocity update uses the old position — so the scheme is
        not a symmetric splitting and preserves nothing. Asserted rather than
        omitted, so the contrast with the symplectic pair above is on the record.
        """
        state = self._state()
        trajectory = integrate(
            *state.as_arrays(), duration_days=400.0, dt=1.0, integrator="euler"
        )
        assert trajectory.relative_angular_momentum_error.max() > 1e-3

    def test_barycentre_stays_at_the_origin(self):
        state = self._state()
        trajectory = integrate(
            *state.as_arrays(), duration_days=400.0, dt=1.0, integrator="leapfrog"
        )

        offsets = [
            np.linalg.norm(centre_of_mass(sample, state.gms))
            for sample in trajectory.positions
        ]
        assert max(offsets) < 1e-12


class TestTimeReversibility:
    def test_leapfrog_returns_to_its_starting_point(self):
        """Integrate forward, then back, and land where you began.

        Exact time-reversibility is the structural property that bounds leapfrog's
        energy error. RK4 has no such symmetry, which is checked below so the
        contrast is measured rather than claimed.
        """
        state = two_body_state(1.0, 0.35, GM_SUN, COMPANION_GM)

        forward = integrate(
            *state.as_arrays(), duration_days=500.0, dt=1.0, integrator="leapfrog"
        )
        backward = integrate(
            forward.positions[-1],
            forward.velocities[-1],
            state.gms,
            duration_days=-500.0,
            dt=1.0,
            integrator="leapfrog",
        )

        np.testing.assert_allclose(backward.positions[-1], state.positions, atol=1e-11)
        np.testing.assert_allclose(backward.velocities[-1], state.velocities, atol=1e-13)

    def test_rk4_is_not_time_reversible(self):
        state = two_body_state(1.0, 0.35, GM_SUN, COMPANION_GM)

        forward = integrate(
            *state.as_arrays(), duration_days=500.0, dt=1.0, integrator="rk4"
        )
        backward = integrate(
            forward.positions[-1],
            forward.velocities[-1],
            state.gms,
            duration_days=-500.0,
            dt=1.0,
            integrator="rk4",
        )

        residual = np.linalg.norm(backward.positions[-1] - state.positions)
        assert residual > 1e-13, "RK4 should not close the loop exactly"


class TestSolarSystemInitialConditions:
    def test_barycentric_frame_is_clean(self):
        state = solar_system_state(J2000_JD)
        quality = report_frame_quality(state)

        assert quality["centre_of_mass_offset_au"] < 1e-15
        assert quality["linear_momentum_magnitude"] < 1e-18

    def test_includes_sun_and_all_planets(self):
        state = solar_system_state(J2000_JD)
        assert state.names[0] == "sun"
        assert len(state) == 9

    def test_sun_is_displaced_from_the_origin_by_the_planets(self):
        """The Sun does not sit at the barycentre --- Jupiter alone shifts it.

        The offset is around one solar radius, which is 0.0047 AU. Getting this
        wrong would mean the heliocentric-to-barycentric shift never happened.
        """
        state = solar_system_state(J2000_JD)
        sun_offset = float(np.linalg.norm(state.positions[0]))

        assert 0.0005 < sun_offset < 0.01, f"Sun sits {sun_offset:.5f} AU from barycentre"

    def test_sun_carries_momentum(self):
        state = solar_system_state(J2000_JD)
        assert float(np.linalg.norm(state.velocities[0])) > 0.0


class TestSolarSystemIntegration:
    def test_earth_returns_after_one_year(self):
        state = solar_system_state(J2000_JD)
        period_days = 365.256

        trajectory = integrate(
            *state.as_arrays(),
            duration_days=period_days,
            dt=0.5,
            integrator="leapfrog",
            names=state.names,
            sample_every=20,
        )

        start, _ = trajectory.positions[0, state.names.index("earth")], None
        end = trajectory.positions[-1, state.names.index("earth")]

        # Not exact: Earth's sidereal year is not precisely 365.256 days at this
        # epoch, and the other planets perturb it. A few thousandths of an AU is
        # the right order --- returning to within a millimetre would be suspicious.
        assert float(np.linalg.norm(end - start)) < 0.02

    def test_energy_is_conserved_over_a_century(self):
        """A century of the full system, with the step size Mercury demands.

        Mercury sets the requirement: its 88-day year means a 2-day step resolves
        the orbit only about forty times, and the resulting energy error is a few
        parts in a million. Every other planet is far better resolved, so tightening
        this bound means shortening the step for Mercury's sake alone.
        """
        state = solar_system_state(J2000_JD)

        trajectory = integrate(
            *state.as_arrays(),
            duration_days=365.25 * 100,
            dt=2.0,
            integrator="leapfrog",
            names=state.names,
            sample_every=500,
        )

        assert trajectory.relative_energy_error.max() < 1e-4

    def test_shorter_steps_conserve_energy_better(self):
        """The error is discretisation, not a bug: it shrinks when the step does.

        Distinguishes "acceptable numerical error" from "something is wrong", which
        a single tolerance never can.
        """
        drifts = {}
        for dt in (4.0, 1.0):
            state = solar_system_state(J2000_JD)
            trajectory = integrate(
                *state.as_arrays(),
                duration_days=365.25 * 5,
                dt=dt,
                integrator="leapfrog",
                names=state.names,
                sample_every=100,
            )
            drifts[dt] = trajectory.relative_energy_error.max()

        assert drifts[1.0] < drifts[4.0] / 4.0, (
            f"expected second-order improvement; got {drifts[4.0]:.3e} -> {drifts[1.0]:.3e}"
        )

    @staticmethod
    def _osculating_axis_spread(trajectory, names: tuple[str, ...], body: str) -> float:
        """Peak-to-peak variation of a body's osculating semi-major axis.

        Measured via the **semi-major axis**, not the instantaneous distance. On an
        eccentric orbit the distance swings by tens of percent within a single year,
        so comparing distances at two arbitrary moments measures orbital phase, not
        drift. The semi-major axis follows from the energy and is phase-independent,
        by the vis-viva equation rearranged:

            a = 1 / (2/r - v^2/mu)
        """
        sun_index = names.index("sun")
        index = names.index(body)

        relative_position = (
            trajectory.positions[:, index, :] - trajectory.positions[:, sun_index, :]
        )
        relative_velocity = (
            trajectory.velocities[:, index, :] - trajectory.velocities[:, sun_index, :]
        )
        distance = np.linalg.norm(relative_position, axis=-1)
        speed_squared = np.einsum("ij,ij->i", relative_velocity, relative_velocity)

        semi_major_axis = 1.0 / (2.0 / distance - speed_squared / (GM_SUN + GM_BODY[body]))
        return float(
            (semi_major_axis.max() - semi_major_axis.min()) / semi_major_axis.mean()
        )

    @staticmethod
    def _century(dt: float):
        state = solar_system_state(J2000_JD)
        trajectory = integrate(
            *state.as_arrays(),
            duration_days=365.25 * 100,
            dt=dt,
            integrator="leapfrog",
            names=state.names,
            sample_every=max(1, int(200 * 2.0 / dt)),
        )
        return state, trajectory

    def test_no_planet_escapes_or_falls_in(self):
        """Every planet stays recognisably on its own orbit for a century."""
        state, trajectory = self._century(2.0)

        for name in ("mercury", "earth", "jupiter", "neptune"):
            spread = self._osculating_axis_spread(trajectory, state.names, name)
            assert spread < 0.02, f"{name} semi-major axis varied by {spread:.3%}"

    def test_outer_planet_variation_is_physical_not_numerical(self):
        """Halving the step must not change a *real* perturbation.

        Saturn, Uranus and Neptune show osculating semi-major axes wandering by
        0.7-1.2% over a century, which looks alarming until you vary the step size:
        the figures are identical to four significant digits at dt = 2 days and
        dt = 0.5 days. Step-independent means it is the mutual gravity of the giant
        planets, not discretisation — which is exactly the perturbation the two-body
        ephemeris of phase 1 cannot represent, and the reason this module exists.

        This is the diagnostic that separates "the integrator is wrong" from "the
        physics is richer than the model it replaced". A fixed tolerance can never
        make that distinction.
        """
        coarse_state, coarse = self._century(2.0)
        fine_state, fine = self._century(0.5)

        for name in ("saturn", "uranus", "neptune"):
            coarse_spread = self._osculating_axis_spread(coarse, coarse_state.names, name)
            fine_spread = self._osculating_axis_spread(fine, fine_state.names, name)

            assert coarse_spread > 0.005, f"{name}: expected a real perturbation signal"
            assert abs(coarse_spread - fine_spread) / coarse_spread < 0.02, (
                f"{name}: spread moved with the step size "
                f"({coarse_spread:.5%} -> {fine_spread:.5%}), so it is numerical"
            )

    def test_mercury_variation_is_numerical_and_converges(self):
        """The counterpart: Mercury's wander *does* shrink with the step.

        Mercury's 88-day year is the worst-resolved orbit in the system, so at a
        2-day step its osculating semi-major axis wobbles by half a percent — larger
        than Neptune's genuine perturbation, and entirely an artefact. Quartering the
        step should cut it by roughly sixteen, second order in dt.
        """
        coarse_state, coarse = self._century(2.0)
        fine_state, fine = self._century(0.5)

        coarse_spread = self._osculating_axis_spread(coarse, coarse_state.names, "mercury")
        fine_spread = self._osculating_axis_spread(fine, fine_state.names, "mercury")

        assert fine_spread < coarse_spread / 4.0, (
            f"expected second-order convergence; got {coarse_spread:.5%} -> {fine_spread:.5%}"
        )


class TestTrajectoryContainer:
    def test_records_endpoints_even_when_sampling_sparsely(self):
        state = two_body_state(1.0, 0.1, GM_SUN, COMPANION_GM)
        trajectory = integrate(
            *state.as_arrays(),
            duration_days=100.0,
            dt=1.0,
            integrator="leapfrog",
            sample_every=7,  # 100 is not a multiple of 7
        )

        assert trajectory.times[0] == 0.0
        assert trajectory.times[-1] == pytest.approx(100.0)

    def test_body_lookup_by_name(self):
        state = solar_system_state(J2000_JD, bodies=("earth", "mars"))
        trajectory = integrate(
            *state.as_arrays(),
            duration_days=10.0,
            dt=1.0,
            integrator="leapfrog",
            names=state.names,
        )

        positions, velocities = trajectory.body("mars")
        assert positions.shape == (11, 3)
        assert velocities.shape == (11, 3)

    def test_diagnostics_match_direct_computation(self):
        state = two_body_state(1.0, 0.2, GM_SUN, COMPANION_GM)
        trajectory = integrate(
            *state.as_arrays(), duration_days=50.0, dt=1.0, integrator="leapfrog"
        )

        assert trajectory.energy[0] == pytest.approx(
            total_energy(state.positions, state.velocities, state.gms)
        )
        np.testing.assert_allclose(
            trajectory.angular_momentum[0],
            total_angular_momentum(state.positions, state.velocities, state.gms),
        )
        np.testing.assert_allclose(
            trajectory.linear_momentum[0],
            total_linear_momentum(state.velocities, state.gms),
        )

    def test_energy_is_negative_for_a_bound_system(self):
        state = solar_system_state(J2000_JD)
        assert total_energy(state.positions, state.velocities, state.gms) < 0.0


class TestInputValidation:
    def test_rejects_unknown_integrator(self):
        state = two_body_state()
        with pytest.raises(ValueError, match="unknown integrator"):
            integrate(*state.as_arrays(), duration_days=1.0, dt=1.0, integrator="verlet")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_dt", [0.0, -1.0])
    def test_rejects_non_positive_dt(self, bad_dt):
        state = two_body_state()
        with pytest.raises(ValueError, match="dt must be positive"):
            integrate(*state.as_arrays(), duration_days=10.0, dt=bad_dt)

    def test_rejects_bad_sample_every(self):
        state = two_body_state()
        with pytest.raises(ValueError, match="sample_every"):
            integrate(*state.as_arrays(), duration_days=10.0, dt=1.0, sample_every=0)

    def test_rejects_non_elliptical_two_body(self):
        with pytest.raises(ValueError, match="eccentricity"):
            two_body_state(1.0, 1.0)
