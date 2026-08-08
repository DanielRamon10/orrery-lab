"""Tests for the two models, and for the evaluation protocol itself.

The protocol is the part worth testing. It is easy to write a model evaluation that
reports an impressive number and means nothing, and two specific ways of doing so are
guarded against here:

* :class:`TestTunedThresholdBaseline` checks that the "best single cut" baseline really
  is the best, by brute force. Without that, the claim "the model beats a tuned
  threshold" would rest on the tuning being competent.
* :class:`TestLeakageIsDetected` asserts that the leaked columns *do* inflate the
  score. The demonstration only works if the trap is actually a trap.
"""

from __future__ import annotations

import numpy as np
import pytest

from orrery.koi import LEAKY_FEATURES, PHYSICAL_FEATURES, koi_cache_path, load_koi_table
from orrery.models import best_threshold, evaluate_koi_leakage, evaluate_stability_models
from orrery.stability import FEATURE_NAMES, HILL_STABILITY_THRESHOLD, build_stability_dataset

needs_koi = pytest.mark.skipif(
    not koi_cache_path().exists(),
    reason="no KOI snapshot; run scripts/fetch_exoplanets.py --koi",
)


@pytest.fixture(scope="module")
def dataset():
    """Small but large enough for a split and a stable comparison."""
    return build_stability_dataset(count=700, orbits=250.0, seed=31)


@pytest.fixture(scope="module")
def comparison(dataset):
    return evaluate_stability_models(dataset)


@pytest.fixture(scope="module")
def leakage():
    """Module-level rather than class-scoped: pytest deprecates the latter as a method."""
    return evaluate_koi_leakage(load_koi_table())


class TestTunedThresholdBaseline:
    def test_matches_brute_force(self):
        generator = np.random.default_rng(5)
        values = generator.uniform(0.0, 10.0, size=300)
        labels = (values + generator.normal(scale=2.0, size=300) > 5.0).astype(int)

        threshold, accuracy = best_threshold(values, labels)

        # Exhaustive scan over every candidate cut.
        candidates = np.concatenate([[values.min() - 1.0], np.sort(values) + 1e-12])
        best = max(np.mean((values > cut).astype(int) == labels) for cut in candidates)

        assert accuracy == pytest.approx(best, abs=1e-12)
        assert np.mean((values > threshold).astype(int) == labels) == pytest.approx(accuracy)

    def test_perfectly_separable_data(self):
        values = np.array([1.0, 2.0, 3.0, 8.0, 9.0, 10.0])
        labels = np.array([0, 0, 0, 1, 1, 1])

        threshold, accuracy = best_threshold(values, labels)
        assert accuracy == pytest.approx(1.0)
        assert 3.0 < threshold < 8.0

    def test_handles_a_single_class(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        _, accuracy = best_threshold(values, np.ones(4, dtype=int))
        assert accuracy == pytest.approx(1.0)


class TestStabilityComparison:
    def test_reports_all_three_approaches(self, comparison):
        assert [score.name for score in comparison.scores] == [
            "Gladman criterion",
            "Tuned threshold",
            "Gradient boosting",
        ]

    def test_the_analytic_criterion_is_already_informative(self, comparison):
        """Gladman is a real result, not a straw man — it should do well."""
        assert comparison.by_name("Gladman criterion").accuracy > 0.70

    def test_the_fitted_cut_is_optimal_where_it_was_fitted(self, dataset):
        """The tuned baseline must genuinely be the best single cut, by construction.

        Asserted on the data the threshold is chosen from, which is where the
        guarantee holds. On a *held-out* split it can lose to Gladman by a point or
        two on a small sample — the earlier version of this test asserted otherwise
        and was simply wrong. The claim that survives is this one: nobody can say the
        model only beat the threshold because the threshold was placed badly.
        """
        separation = dataset.features[:, FEATURE_NAMES.index("hill_separation")]
        _, tuned_accuracy = best_threshold(separation, dataset.labels)

        gladman_accuracy = np.mean(
            (separation > HILL_STABILITY_THRESHOLD).astype(int) == dataset.labels
        )
        assert tuned_accuracy >= gladman_accuracy

    def test_the_model_beats_the_best_single_threshold(self, comparison):
        """The claim that actually matters, on held-out data."""
        model = comparison.by_name("Gradient boosting")
        tuned = comparison.by_name("Tuned threshold")

        assert model.accuracy > tuned.accuracy
        assert model.roc_auc > tuned.roc_auc

    def test_cross_validated_auc_is_consistent_with_the_split(self, comparison):
        mean, deviation = comparison.cross_validated_auc
        assert 0.80 < mean < 1.0
        assert deviation < 0.05
        assert abs(mean - comparison.by_name("Gradient boosting").roc_auc) < 0.08

    def test_hill_separation_is_the_dominant_feature(self, comparison):
        """The model should rediscover what the physics already says matters most."""
        top = next(iter(comparison.feature_importance))
        assert top in {"hill_separation", "semi_major_axis_ratio"}

    def test_eccentricity_contributes(self, comparison):
        """The reason a model can beat Gladman at all.

        Gladman's criterion is derived for circular orbits, so anything the model
        gains from eccentricity is information the analytic rule structurally cannot
        use.
        """
        eccentricity_features = {
            "eccentricity_inner",
            "eccentricity_outer",
            "max_eccentricity",
            "eccentricity_crossing",
        }
        contribution = sum(
            max(0.0, comparison.feature_importance.get(name, 0.0))
            for name in eccentricity_features
        )
        assert contribution > 0.0

    def test_predictions_cover_the_test_split(self, comparison):
        for score in comparison.scores:
            assert score.predictions.shape == comparison.test_labels.shape
            assert set(np.unique(score.predictions)).issubset({0, 1})

    def test_continuous_scores_are_kept_for_the_ranking_models(self, comparison):
        """A ROC needs the ranking; hard labels collapse the curve to three points.

        The figure got this wrong once, so the shape is pinned here.
        """
        assert comparison.by_name("Gladman criterion").scores is None
        for name in ("Tuned threshold", "Gradient boosting"):
            scores = comparison.by_name(name).scores
            assert scores is not None
            assert scores.shape == comparison.test_labels.shape
            assert len(np.unique(scores)) > 10

    def test_probabilities_are_probabilities(self, comparison):
        assert np.all(comparison.model_probabilities >= 0.0)
        assert np.all(comparison.model_probabilities <= 1.0)

    def test_the_gladman_rule_is_applied_as_published(self, comparison):
        """No quiet retuning of the baseline's threshold."""
        expected = (comparison.test_hill_separation > HILL_STABILITY_THRESHOLD).astype(int)
        np.testing.assert_array_equal(
            comparison.by_name("Gladman criterion").predictions, expected
        )

    def test_reproducible(self, dataset):
        first = evaluate_stability_models(dataset)
        second = evaluate_stability_models(dataset)
        assert first.by_name("Gradient boosting").roc_auc == pytest.approx(
            second.by_name("Gradient boosting").roc_auc
        )


@needs_koi
class TestLeakageIsDetected:
    def test_physical_features_alone_do_a_respectable_job(self, leakage):
        """The honest model is genuinely useful, not a foil for the leaky one."""
        assert leakage.physical.accuracy > 0.85
        assert leakage.physical.roc_auc > 0.90

    def test_leaked_columns_inflate_the_score(self, leakage):
        """The trap has to actually work, or the demonstration proves nothing."""
        assert leakage.accuracy_gap > 0.02
        assert leakage.with_leakage.accuracy > 0.98

    def test_the_leaked_columns_are_what_the_model_leans_on(self, leakage):
        """Confirms the gap comes from leakage rather than from extra columns."""
        assert max(leakage.leaky_importance.values()) > 0.01

    def test_false_positive_flags_are_never_in_the_honest_feature_set(self):
        assert not set(PHYSICAL_FEATURES) & set(LEAKY_FEATURES)
        assert all(name.startswith("koi_") for name in PHYSICAL_FEATURES)

    def test_evaluated_on_complete_rows_only(self, leakage):
        table = load_koi_table()
        assert leakage.sample_size == int(table.complete_rows().sum())
        assert leakage.sample_size > 5000

    def test_both_models_saw_the_same_rows(self, leakage):
        assert leakage.physical.predictions.shape == leakage.with_leakage.predictions.shape
