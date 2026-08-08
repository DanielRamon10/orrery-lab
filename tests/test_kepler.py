"""Tests for the Kepler equation solver.

The strategy here is to test against *physical laws* rather than against a table
of expected numbers. A residual check ("does the answer satisfy the equation?")
is stronger than a golden-value check, because it cannot be satisfied by a
plausible-looking wrong answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from orrery.constants import GM_SUN
from orrery.kepler import (
    eccentric_from_true_anomaly,
    mean_motion,
    normalize_angle,
    orbital_period,
    radius_from_eccentric,
    solve_kepler,
    true_anomaly_from_eccentric,
)

# A grid that covers circular through very eccentric orbits, and a full
# revolution of mean anomaly including the awkward points at 0 and pi.
ECCENTRICITIES = np.array([0.0, 0.0067, 0.0934, 0.2488, 0.5, 0.8, 0.95, 0.99])
MEAN_ANOMALIES = np.linspace(-np.pi, np.pi, 73)


class TestSolveKepler:
    def test_residual_is_zero_across_the_full_grid(self):
        """E - e sin E must reproduce M to near machine precision, everywhere.

        The residual is wrapped before comparison because angles that differ by a
        whole revolution are the same angle: the solver returns E in [-pi, pi),
        so an input of M = +pi legitimately comes back as E = -pi.
        """
        mean = MEAN_ANOMALIES[:, None]
        ecc = ECCENTRICITIES[None, :]

        eccentric = solve_kepler(mean, ecc)
        residual = normalize_angle(eccentric - ecc * np.sin(eccentric) - mean)

        assert np.max(np.abs(residual)) < 1e-11

    def test_circular_orbit_is_the_identity(self):
        """With e = 0 the equation collapses to E = M (up to a full revolution)."""
        eccentric = solve_kepler(MEAN_ANOMALIES, 0.0)
        np.testing.assert_allclose(
            normalize_angle(eccentric - MEAN_ANOMALIES), 0.0, atol=1e-14
        )

    @pytest.mark.parametrize("eccentricity", [0.0, 0.3, 0.9])
    def test_apsides_are_fixed_points(self, eccentricity):
        """At perihelion (M=0) and aphelion (M=pi), E equals M exactly."""
        assert abs(float(solve_kepler(0.0, eccentricity))) < 1e-14
        assert abs(abs(float(solve_kepler(np.pi, eccentricity))) - np.pi) < 1e-12

    def test_odd_symmetry(self):
        """Kepler's equation is odd in M, so E(-M) = -E(M)."""
        forward = MEAN_ANOMALIES[MEAN_ANOMALIES > 0]
        positive = solve_kepler(forward, 0.4)
        negative = solve_kepler(-forward, 0.4)
        np.testing.assert_allclose(normalize_angle(positive + negative), 0.0, atol=1e-12)

    def test_revolutions_wrap(self):
        """Adding whole revolutions to M must not change the answer."""
        base = solve_kepler(0.7, 0.3)
        wrapped = solve_kepler(0.7 + 6.0 * np.pi, 0.3)
        assert abs(float(base - wrapped)) < 1e-12

    def test_broadcasting_shape(self):
        result = solve_kepler(np.zeros((4, 1)), np.zeros((1, 3)))
        assert result.shape == (4, 3)

    @pytest.mark.parametrize("bad_eccentricity", [-0.1, 1.0, 1.5])
    def test_rejects_non_elliptical_orbits(self, bad_eccentricity):
        with pytest.raises(ValueError, match="elliptical"):
            solve_kepler(0.5, bad_eccentricity)


class TestAnomalyConversions:
    def test_true_anomaly_round_trip(self):
        mean = MEAN_ANOMALIES[:, None]
        ecc = ECCENTRICITIES[None, :]

        eccentric = solve_kepler(mean, ecc)
        true = true_anomaly_from_eccentric(eccentric, ecc)
        recovered = eccentric_from_true_anomaly(true, ecc)

        np.testing.assert_allclose(
            normalize_angle(recovered), normalize_angle(eccentric), atol=1e-11
        )

    def test_true_anomaly_leads_eccentric_after_perihelion(self):
        """For 0 < M < pi the true anomaly runs ahead of the eccentric anomaly.

        This is Kepler's second law showing up as an inequality: the planet has
        already swept more angle than a uniform-rate body would have.
        """
        mean = np.linspace(0.05, np.pi - 0.05, 40)
        eccentric = solve_kepler(mean, 0.5)
        true = true_anomaly_from_eccentric(eccentric, 0.5)

        assert np.all(true > eccentric)
        assert np.all(eccentric > mean)

    def test_conic_equation_agrees_with_eccentric_form(self):
        """r = a(1 - e cos E) must equal the focal conic r = a(1-e^2)/(1+e cos nu)."""
        a, e = 1.5, 0.35
        eccentric = solve_kepler(MEAN_ANOMALIES, e)
        true = true_anomaly_from_eccentric(eccentric, e)

        from_eccentric = radius_from_eccentric(a, e, eccentric)
        from_conic = a * (1.0 - e**2) / (1.0 + e * np.cos(true))

        np.testing.assert_allclose(from_eccentric, from_conic, rtol=1e-12)


class TestKeplersThirdLaw:
    def test_earth_period_is_one_year(self):
        """a = 1 AU must give a period of one year, by construction of the units."""
        period = float(orbital_period(1.0, GM_SUN))
        assert abs(period - 365.256) < 0.01

    def test_period_squared_scales_as_axis_cubed(self):
        axes = np.array([0.387, 0.723, 1.0, 1.524, 5.203, 9.537, 19.19, 30.07])
        periods = orbital_period(axes, GM_SUN)

        # log P = 1.5 log a + const  =>  the fitted slope must be exactly 3/2.
        slope, _ = np.polyfit(np.log10(axes), np.log10(periods), deg=1)
        assert abs(slope - 1.5) < 1e-12

    def test_mean_motion_is_inverse_of_period(self):
        axes = np.array([0.5, 1.0, 10.0])
        np.testing.assert_allclose(
            mean_motion(axes, GM_SUN) * orbital_period(axes, GM_SUN),
            2.0 * np.pi,
            rtol=1e-14,
        )


class TestNormalizeAngle:
    def test_centered_range(self):
        wrapped = normalize_angle(np.array([0.0, np.pi + 0.1, 3.0 * np.pi]))
        assert np.all(wrapped >= -np.pi) and np.all(wrapped < np.pi)

    def test_positive_range(self):
        wrapped = normalize_angle(np.array([-0.5, -7.0, 9.0]), centered=False)
        assert np.all(wrapped >= 0.0) and np.all(wrapped < 2.0 * np.pi)
