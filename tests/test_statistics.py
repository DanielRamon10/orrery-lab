"""Tests for the statistical layer.

Two of these test classes exist mainly to stop the project from overclaiming.

:class:`TestTitiusBodeIsNotAsGoodAsRSquaredSuggests` asserts *both* that the fitted
geometric progression has an R-squared above 0.99 *and* that its worst prediction is
off by more than 15%. Either number alone tells a misleading story; the pair is the
finding.

:class:`TestResonanceClustering` asserts the Monte Carlo p-value is large — that the
solar system's period ratios are **not** unusually commensurable under this null. It
is a negative result, and pinning it down means nobody can later quietly present the
resonance list as evidence of something it is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from orrery.elements import PLANET_NAMES, PLANETS
from orrery.statistics import (
    GEOMETRIC_SLOTS,
    TITIUS_BODE_SLOTS,
    angular_momentum_budget,
    best_rational_approximation,
    bootstrap_slope,
    find_resonances,
    fit_line,
    fit_power_law,
    keplers_third_law_fit,
    monte_carlo_resonance_test,
    observed_distances,
    resonance_statistic,
    titius_bode_distance,
    titius_bode_fit,
    titius_bode_prediction,
)


class TestFitLine:
    def test_recovers_an_exact_line(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        fit = fit_line(x, 3.0 + 2.5 * x)

        assert fit.slope == pytest.approx(2.5)
        assert fit.intercept == pytest.approx(3.0)
        assert fit.r_squared == pytest.approx(1.0)
        # A perfect fit has no residual scatter, so the standard error vanishes.
        assert fit.slope_stderr == pytest.approx(0.0, abs=1e-14)

    def test_recovers_a_noisy_line_within_its_own_error_bars(self):
        generator = np.random.default_rng(20260726)
        x = np.linspace(0.0, 10.0, 40)
        y = 1.0 + 0.75 * x + generator.normal(scale=0.4, size=x.size)

        fit = fit_line(x, y)
        low, high = fit.slope_confidence_interval(0.95)

        assert low < 0.75 < high
        assert fit.slope_stderr > 0.0
        assert fit.degrees_of_freedom == 38

    def test_residuals_sum_to_zero(self):
        """A property of least squares, and a cheap check that the fit is a fit."""
        generator = np.random.default_rng(1)
        x = np.linspace(0, 5, 20)
        fit = fit_line(x, 2 * x + generator.normal(size=20))
        assert float(np.sum(fit.residuals)) == pytest.approx(0.0, abs=1e-12)

    def test_power_law_recovers_the_exponent(self):
        x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        fit = fit_power_law(x, 3.0 * x**2.5)

        assert fit.slope == pytest.approx(2.5)
        assert 10.0**fit.intercept == pytest.approx(3.0)

    @pytest.mark.parametrize(
        ("x", "y", "message"),
        [
            (np.array([1.0, 2.0]), np.array([1.0, 2.0]), "at least 3"),
            (np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]), "identical"),
            (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]), "same shape"),
        ],
    )
    def test_rejects_degenerate_input(self, x, y, message):
        with pytest.raises(ValueError, match=message):
            fit_line(x, y)

    def test_power_law_rejects_non_positive_data(self):
        with pytest.raises(ValueError, match="strictly positive"):
            fit_power_law(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 3.0]))


class TestKeplersThirdLaw:
    def test_exponent_is_three_halves(self):
        fit = keplers_third_law_fit()
        assert fit.slope == pytest.approx(1.5, abs=1e-12)
        assert fit.r_squared == pytest.approx(1.0)

    def test_consistent_with_the_theoretical_value(self):
        """A t-test that should *fail to reject*, which is the expected outcome."""
        _, p_value = keplers_third_law_fit().test_slope(1.5)
        assert p_value > 0.05

    def test_would_reject_a_wrong_hypothesis(self):
        """The test has power: it does not accept everything put to it."""
        _, p_value = keplers_third_law_fit().test_slope(1.4)
        assert p_value < 1e-6

    def test_confidence_interval_brackets_three_halves(self):
        low, high = keplers_third_law_fit().slope_confidence_interval()
        assert low <= 1.5 <= high


class TestClassicalTitiusBode:
    """The parameter-free rule ``a = 0.4 + 0.3 * 2**n``."""

    def test_slot_formula(self):
        assert titius_bode_distance(None) == pytest.approx(0.4)  # Mercury
        assert titius_bode_distance(0) == pytest.approx(0.7)  # Venus
        assert titius_bode_distance(1) == pytest.approx(1.0)  # Earth
        assert titius_bode_distance(4) == pytest.approx(5.2)  # Jupiter
        assert titius_bode_distance(7) == pytest.approx(38.8)  # Neptune's slot

    def test_fits_everything_except_neptune_within_five_percent(self):
        result = titius_bode_prediction()
        errors = result.errors_excluding("neptune")

        assert max(abs(error) for error in errors.values()) < 0.055
        assert set(errors) == set(TITIUS_BODE_SLOTS) - {"neptune"}

    def test_neptune_is_the_failure(self):
        """The rule misses Neptune by nearly 30% — the historically decisive miss."""
        result = titius_bode_prediction()

        assert result.worst_body == "neptune"
        assert result.relative_error("neptune") > 0.25

    def test_the_belt_slot_was_a_real_prediction(self):
        """Slot 3 was empty when the rule was written; Ceres turned up there in 1801."""
        result = titius_bode_prediction()
        assert abs(result.relative_error("ceres")) < 0.02

    def test_pluto_sits_in_the_slot_neptune_missed(self):
        """Slot 7 predicts 38.8 AU. Neptune is at 30.1; Pluto is at 39.5."""
        slot_seven = titius_bode_distance(7)
        neptune = PLANETS["neptune"].semi_major_axis_au
        pluto = PLANETS["pluto"].semi_major_axis_au

        assert abs(slot_seven - pluto) / pluto < 0.03
        assert abs(slot_seven - neptune) / neptune > 0.25


class TestTitiusBodeIsNotAsGoodAsRSquaredSuggests:
    """The statistical caution, asserted so it cannot be quietly dropped.

    A fitted geometric progression scores R-squared above 0.99 while making
    predictions that are off by a fifth. Both halves are tested together because
    either one alone misleads.
    """

    def test_r_squared_looks_excellent(self):
        assert titius_bode_fit().fit.r_squared > 0.99

    def test_and_yet_the_predictions_are_poor(self):
        result = titius_bode_fit()
        assert result.max_absolute_error > 0.15

    def test_worse_than_the_parameter_free_classical_rule(self):
        """Fitting two free parameters does *worse* than the 18th-century constants.

        Excluding Neptune, which the classical rule cannot handle at all, the rule
        with no fitted parameters beats the regression comfortably.
        """
        classical = titius_bode_prediction().errors_excluding("neptune")
        fitted = titius_bode_fit().errors_excluding("neptune")

        assert max(abs(v) for v in classical.values()) < max(abs(v) for v in fitted.values())

    def test_slot_maps_cover_the_same_bodies(self):
        assert set(GEOMETRIC_SLOTS) == set(TITIUS_BODE_SLOTS)


class TestRationalApproximation:
    def test_finds_the_jupiter_saturn_ratio(self):
        fraction = best_rational_approximation(2.4816, max_integer=13)
        assert (fraction.numerator, fraction.denominator) == (5, 2)

    def test_bounds_the_numerator_as_well_as_the_denominator(self):
        """The bug this replaced: bounding only ``q`` makes every ratio "resonant".

        Pluto's period is 131.9 times Mars's. With the numerator unbounded, 1319/10
        matches to five decimals and implies a 1309th-order resonance, which is
        meaningless. Both integers must be bounded.
        """
        fraction = best_rational_approximation(131.9007, max_integer=13)
        assert fraction.numerator <= 13
        assert fraction.denominator <= 13

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.5, (1, 2)), (1.5, (3, 2)), (2.0, (2, 1)), (1.6255, (13, 8))],
    )
    def test_known_values(self, value, expected):
        fraction = best_rational_approximation(value, max_integer=13)
        assert (fraction.numerator, fraction.denominator) == expected

    def test_exhaustive_search_really_is_optimal(self):
        """Cross-check against brute force over every admissible pair."""
        generator = np.random.default_rng(7)
        for value in generator.uniform(0.2, 12.0, size=200):
            fraction = best_rational_approximation(float(value), max_integer=12)
            best = min(
                (
                    abs(value - p / q)
                    for q in range(1, 13)
                    for p in range(1, 13)
                ),
            )
            found = abs(value - fraction.numerator / fraction.denominator)
            assert found == pytest.approx(best, abs=1e-15)

    @pytest.mark.parametrize("bad", [0.0, -1.5])
    def test_rejects_non_positive_values(self, bad):
        with pytest.raises(ValueError, match="positive"):
            best_rational_approximation(bad)

    def test_rejects_bad_bound(self):
        with pytest.raises(ValueError, match="max_integer"):
            best_rational_approximation(1.5, max_integer=0)


class TestFindResonances:
    def test_surfaces_the_known_commensurabilities(self):
        found = {
            (r.inner, r.outer): (r.p, r.q) for r in find_resonances(adjacent_only=True)
        }

        # The three every textbook mentions.
        assert found[("venus", "earth")] == (13, 8)
        assert found[("jupiter", "saturn")] == (5, 2)
        assert found[("uranus", "neptune")] == (2, 1)

    def test_finds_the_neptune_pluto_lock(self):
        """The only genuine mean-motion resonance among the major bodies: 3:2."""
        found = find_resonances((*PLANET_NAMES, "pluto"), tolerance=0.02)
        pair = next(r for r in found if {r.inner, r.outer} == {"neptune", "pluto"})

        assert (pair.p, pair.q) == (3, 2)
        assert pair.order == 1
        assert pair.relative_error < 0.005

    def test_rejects_high_order_coincidences(self):
        """Nothing absurd survives the order filter."""
        for resonance in find_resonances((*PLANET_NAMES, "pluto"), tolerance=0.03):
            assert resonance.order <= 6
            assert resonance.p <= 13 and resonance.q <= 13

    def test_relaxing_the_order_bound_lets_noise_back_in(self):
        """More permissive bounds find more "resonances" that mean less.

        Demonstrates that the filter is doing real work rather than being decoration.
        """
        strict = find_resonances(max_order=2, tolerance=0.03)
        loose = find_resonances(max_order=40, max_integer=60, tolerance=0.03)
        assert len(loose) > len(strict)

    def test_ratios_are_always_greater_than_one(self):
        for resonance in find_resonances(tolerance=0.05):
            assert resonance.period_ratio > 1.0
            assert (
                PLANETS[resonance.outer].semi_major_axis_au
                > PLANETS[resonance.inner].semi_major_axis_au
            )

    def test_sorted_closest_first(self):
        errors = [r.relative_error for r in find_resonances(tolerance=0.05)]
        assert errors == sorted(errors)

    def test_adjacent_only_is_a_subset(self):
        adjacent = find_resonances(adjacent_only=True, tolerance=0.05)
        everything = find_resonances(adjacent_only=False, tolerance=0.05)
        assert len(adjacent) <= len(everything)


class TestResonanceClustering:
    """A negative result, asserted so it stays on the record."""

    def test_no_significant_clustering_under_this_null(self):
        result = monte_carlo_resonance_test(trials=4000)

        assert result.p_value > 0.2
        assert "no evidence" in result.interpretation

    def test_the_real_system_is_not_closer_than_chance(self):
        """It is in fact slightly *farther* from simple ratios than a random system.

        Worth stating plainly: popular accounts often present solar-system
        commensurabilities as remarkable. Restricted to genuinely low-order ratios
        and compared against a matched random null, they are not.
        """
        result = monte_carlo_resonance_test(trials=4000)
        assert result.observed > result.null_median

    def test_reproducible_for_a_fixed_seed(self):
        first = monte_carlo_resonance_test(trials=500, seed=42)
        second = monte_carlo_resonance_test(trials=500, seed=42)
        assert first.p_value == second.p_value

    def test_p_value_is_never_exactly_zero(self):
        """A finite number of trials cannot establish p = 0."""
        result = monte_carlo_resonance_test(trials=100)
        assert 0.0 < result.p_value <= 1.0

    def test_statistic_is_zero_for_a_perfectly_commensurable_system(self):
        """Sanity check: axes chosen so every adjacent period ratio is exactly 2:1."""
        axes = np.array([1.0, 2.0 ** (2.0 / 3.0), 4.0 ** (2.0 / 3.0)])
        assert resonance_statistic(axes) == pytest.approx(0.0, abs=1e-12)


class TestAngularMomentumBudget:
    def test_jupiter_dominates(self):
        budget = angular_momentum_budget()
        assert budget.dominant_body == "jupiter"
        assert 0.55 < budget.share("jupiter") < 0.67

    def test_the_sun_holds_almost_none_of_it(self):
        """99.8% of the mass, well under 1% of the angular momentum.

        The central fact any theory of solar-system formation has to account for.
        """
        assert angular_momentum_budget().share("sun") < 0.01

    def test_shares_sum_to_one(self):
        budget = angular_momentum_budget()
        total = budget.share("sun") + sum(budget.share(b) for b in PLANET_NAMES)
        assert total == pytest.approx(1.0)

    def test_giant_planets_hold_almost_everything(self):
        budget = angular_momentum_budget()
        giants = sum(budget.share(b) for b in ("jupiter", "saturn", "uranus", "neptune"))
        assert giants > 0.99

    def test_all_components_are_positive(self):
        budget = angular_momentum_budget()
        assert budget.sun_rotational > 0.0
        assert all(value > 0.0 for value in budget.orbital.values())


class TestBootstrap:
    def test_agrees_with_the_t_interval_on_a_fit_with_real_scatter(self):
        """Two routes to the same interval, one of which assumes normal residuals.

        Close agreement means the t-interval's normality assumption is not doing
        damage here. Disagreement would mean neither could be trusted.
        """
        distances = observed_distances()
        x = np.array([GEOMETRIC_SLOTS[body] for body in distances], dtype=float)
        y = np.log10(np.array([distances[body] for body in distances]))

        fit = fit_line(x, y)
        t_low, t_high = fit.slope_confidence_interval(0.95)
        boot_low, boot_high, samples = bootstrap_slope(x, y, trials=4000)

        assert len(samples) > 3900  # few degenerate resamples with 9 points
        assert boot_low == pytest.approx(t_low, rel=0.1)
        assert boot_high == pytest.approx(t_high, rel=0.1)

    def test_interval_contains_the_point_estimate(self):
        generator = np.random.default_rng(3)
        x = np.linspace(0, 10, 30)
        y = 2.0 + 1.5 * x + generator.normal(scale=0.5, size=30)

        fit = fit_line(x, y)
        low, high = bootstrap_slope(x, y, trials=2000)[:2]
        assert low < fit.slope < high

    def test_reproducible_for_a_fixed_seed(self):
        x = np.linspace(0, 5, 12)
        y = 1.0 + 0.5 * x + np.sin(x)
        assert bootstrap_slope(x, y, trials=500, seed=9)[:2] == bootstrap_slope(
            x, y, trials=500, seed=9
        )[:2]
