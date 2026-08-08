"""The two machine-learning problems, and how to evaluate them without fooling yourself.

Problem 1 --- orbital stability
-------------------------------
Predict whether a two-planet system survives, from its initial conditions. The labels
come from :mod:`orrery.stability`, which integrates every system with the gravity
implemented in phase 3, so the dataset is generated rather than downloaded.

The interesting part is the comparison. There is already an analytic answer here:
Gladman's criterion, ``Delta > 2*sqrt(3)``. A model that merely beats *that* has not
necessarily earned anything, because the criterion is a fixed threshold derived for
circular coplanar orbits and this dataset is neither. So three things are measured
side by side:

1. **Gladman** --- the textbook threshold, unchanged.
2. **Tuned threshold** --- the *best possible* single cut on ``Delta``, chosen on the
   training set. This is the honest baseline: if a learned model cannot beat a
   well-placed threshold on the same single feature, it has added nothing.
3. **Gradient boosting** on all features.

Reporting only the first two would flatter the model. Most write-ups of this kind
skip the second one.

Problem 2 --- classifying Kepler signals
----------------------------------------
Confirmed planet or false positive, from the KOI table. This exists mainly to
demonstrate a **target-leakage** failure concretely: the table contains flags that
encode the reason a signal was rejected, and training on them produces a near-perfect
model that has learned nothing. See :mod:`orrery.koi` for which columns those are.

Both evaluations use a stratified split, fit only on the training half, and report on
data the model has never seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .koi import LEAKY_FEATURES, PHYSICAL_FEATURES, KoiTable
from .stability import FEATURE_NAMES, HILL_STABILITY_THRESHOLD, StabilityDataset

__all__ = [
    "ModelScore",
    "StabilityComparison",
    "LeakageComparison",
    "evaluate_stability_models",
    "evaluate_koi_leakage",
    "best_threshold",
]

RANDOM_STATE = 20260808


@dataclass(frozen=True)
class ModelScore:
    """Held-out performance of one approach.

    Attributes:
        name: Human-readable label.
        accuracy: Fraction correct on the test split.
        roc_auc: Area under the ROC curve; ``nan`` for a hard threshold with no score.
        predictions: Predicted labels on the test split.
        scores: Continuous scores behind the predictions — probabilities, or the raw
            feature for a threshold rule. ``None`` for a fixed rule that produces no
            ranking at all.

            Kept separately because a ROC curve needs the *ranking*, not the
            thresholded labels: feeding hard 0/1 predictions to ``roc_curve`` yields a
            three-point polyline that understates every model it is applied to.
        detail: Free-form notes, e.g. the fitted threshold.
    """

    name: str
    accuracy: float
    roc_auc: float
    predictions: np.ndarray
    scores: np.ndarray | None = None
    detail: str = ""


@dataclass(frozen=True)
class StabilityComparison:
    """Everything needed to judge whether the model earned its place.

    Attributes:
        scores: One :class:`ModelScore` per approach, in reporting order.
        test_labels: Ground truth for the test split.
        test_hill_separation: ``Delta`` for the test split, for plotting.
        model_probabilities: The learned model's predicted probability of stability.
        feature_importance: Permutation importance per feature, on held-out data.
        cross_validated_auc: Mean and standard deviation of AUC across CV folds.
    """

    scores: list[ModelScore]
    test_labels: np.ndarray
    test_hill_separation: np.ndarray
    model_probabilities: np.ndarray
    feature_importance: dict[str, float] = field(default_factory=dict)
    cross_validated_auc: tuple[float, float] = (float("nan"), float("nan"))

    def by_name(self, name: str) -> ModelScore:
        return next(score for score in self.scores if score.name == name)


@dataclass(frozen=True)
class LeakageComparison:
    """Physical features against physical-plus-leaky, on the same split."""

    #: Ground truth for the shared test split, so callers can draw ROC curves.
    test_labels: np.ndarray
    physical: ModelScore
    with_leakage: ModelScore
    leaky_importance: dict[str, float]
    physical_importance: dict[str, float]
    sample_size: int

    @property
    def accuracy_gap(self) -> float:
        """How much accuracy the leaked columns appear to buy."""
        return self.with_leakage.accuracy - self.physical.accuracy


def best_threshold(values: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Find the single cut on ``values`` that maximises accuracy for ``labels``.

    Assumes larger values indicate the positive class, which is the case for Hill
    separation and stability. Evaluates every midpoint between adjacent sorted values,
    so the result is the true optimum rather than a grid approximation.

    Returns:
        ``(threshold, accuracy)``.
    """
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_labels = labels[order]

    # Predicting "positive above the cut": accuracy at each split point follows from
    # the cumulative counts, so all candidate cuts are scored in one pass.
    positives_above = np.cumsum(sorted_labels[::-1])[::-1]
    negatives_below = np.cumsum(1 - sorted_labels) - (1 - sorted_labels)

    correct = positives_above + negatives_below
    best_index = int(np.argmax(correct))

    if best_index == 0:
        threshold = float(sorted_values[0] - 1e-9)
    else:
        threshold = float(0.5 * (sorted_values[best_index - 1] + sorted_values[best_index]))

    return threshold, float(correct[best_index] / len(labels))


def _permutation_importance(model, features, labels, names) -> dict[str, float]:
    """Permutation importance on held-out data, largest first.

    Permutation rather than the model's own split-count importance: split counts are
    biased towards high-cardinality features and say nothing about held-out
    performance, which is the only thing that matters here.
    """
    result = permutation_importance(
        model, features, labels, n_repeats=8, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    ranked = sorted(
        zip(names, result.importances_mean, strict=True), key=lambda item: -item[1]
    )
    return {name: float(value) for name, value in ranked}


def evaluate_stability_models(
    dataset: StabilityDataset,
    test_size: float = 0.3,
) -> StabilityComparison:
    """Compare the analytic criterion, a tuned threshold, and a learned model.

    Args:
        dataset: Output of :func:`orrery.stability.build_stability_dataset`.
        test_size: Fraction held out.

    Returns:
        A :class:`StabilityComparison`.
    """
    features_train, features_test, labels_train, labels_test = train_test_split(
        dataset.features,
        dataset.labels,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=dataset.labels,
    )

    separation_index = FEATURE_NAMES.index("hill_separation")
    separation_train = features_train[:, separation_index]
    separation_test = features_test[:, separation_index]

    scores: list[ModelScore] = []

    # 1. The textbook criterion, applied as published.
    gladman_predictions = (separation_test > HILL_STABILITY_THRESHOLD).astype(int)
    scores.append(
        ModelScore(
            name="Gladman criterion",
            accuracy=float(accuracy_score(labels_test, gladman_predictions)),
            roc_auc=float("nan"),  # a fixed rule produces no ranking
            predictions=gladman_predictions,
            scores=None,
            detail=f"Delta > {HILL_STABILITY_THRESHOLD:.3f}",
        )
    )

    # 2. The best cut on the same single feature, chosen on training data only.
    threshold, _ = best_threshold(separation_train, labels_train)
    tuned_predictions = (separation_test > threshold).astype(int)
    scores.append(
        ModelScore(
            name="Tuned threshold",
            accuracy=float(accuracy_score(labels_test, tuned_predictions)),
            roc_auc=float(roc_auc_score(labels_test, separation_test)),
            predictions=tuned_predictions,
            scores=separation_test,
            detail=f"Delta > {threshold:.3f}, fitted on the training split",
        )
    )

    # 3. Gradient boosting on everything.
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=6, random_state=RANDOM_STATE
    )
    model.fit(features_train, labels_train)

    probabilities = model.predict_proba(features_test)[:, 1]
    model_predictions = (probabilities >= 0.5).astype(int)
    scores.append(
        ModelScore(
            name="Gradient boosting",
            accuracy=float(accuracy_score(labels_test, model_predictions)),
            roc_auc=float(roc_auc_score(labels_test, probabilities)),
            predictions=model_predictions,
            scores=probabilities,
            detail=f"{len(FEATURE_NAMES)} features",
        )
    )

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cross_validated = cross_val_score(
        model, dataset.features, dataset.labels, cv=folds, scoring="roc_auc"
    )

    return StabilityComparison(
        scores=scores,
        test_labels=labels_test,
        test_hill_separation=separation_test,
        model_probabilities=probabilities,
        feature_importance=_permutation_importance(
            model, features_test, labels_test, FEATURE_NAMES
        ),
        cross_validated_auc=(float(cross_validated.mean()), float(cross_validated.std())),
    )


def evaluate_koi_leakage(table: KoiTable, test_size: float = 0.3) -> LeakageComparison:
    """Train with and without the leaked columns, on an identical split.

    The gap between the two is the whole point. A model given the false-positive flags
    reaches accuracy that would look like a triumph and is worthless: those flags are
    the labelling decision, restated.

    Args:
        table: A loaded KOI snapshot.
        test_size: Fraction held out.
    """
    complete = table.complete_rows()
    physical = table.physical[complete]
    leaky = table.leaky[complete]
    labels = table.labels[complete]

    combined = np.hstack([physical, leaky])
    indices = np.arange(len(labels))

    train_index, test_index = train_test_split(
        indices, test_size=test_size, random_state=RANDOM_STATE, stratify=labels
    )

    def fit_and_score(matrix: np.ndarray, name: str, names: tuple[str, ...]) -> tuple:
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, random_state=RANDOM_STATE
        )
        model.fit(matrix[train_index], labels[train_index])

        probabilities = model.predict_proba(matrix[test_index])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        score = ModelScore(
            name=name,
            accuracy=float(accuracy_score(labels[test_index], predictions)),
            roc_auc=float(roc_auc_score(labels[test_index], probabilities)),
            predictions=predictions,
            scores=probabilities,
            detail=f"{matrix.shape[1]} features",
        )
        importance = _permutation_importance(
            model, matrix[test_index], labels[test_index], names
        )
        return score, importance

    physical_score, physical_importance = fit_and_score(
        physical, "Physical features only", PHYSICAL_FEATURES
    )
    leaked_score, leaked_importance = fit_and_score(
        combined, "With leaked vetting flags", PHYSICAL_FEATURES + LEAKY_FEATURES
    )

    return LeakageComparison(
        test_labels=labels[test_index],
        physical=physical_score,
        with_leakage=leaked_score,
        leaky_importance={
            name: value
            for name, value in leaked_importance.items()
            if name in LEAKY_FEATURES
        },
        physical_importance=physical_importance,
        sample_size=int(complete.sum()),
    )
