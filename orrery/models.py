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

There is a third way to flatter it, which :func:`evaluate_inclination_transfer`
exists to close: report only in-distribution numbers. Every feature above is blind to
mutual inclination, and :mod:`orrery.stability` shows that a few degrees of tilt
changes which systems survive. So the model can be scored on systems drawn from a
world it was never shown, with its inputs held byte-identical and only the truth
moved. What that measures is worth stating carefully --- see :class:`TransferScore`,
because on a test set where 91% of systems survive, accuracy alone will call a rule
"better" for reasons that have nothing to do with the rule.

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

from collections.abc import Sequence
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
    "TransferScore",
    "evaluate_stability_models",
    "evaluate_inclination_transfer",
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
class TransferScore:
    """How a coplanar-trained stability model does on tilted systems.

    Attributes:
        median_mutual_inclination_deg: Median angle between the two orbit planes.
        stable_fraction: Fraction of the shifted set that actually survived.
        gladman_accuracy: The analytic criterion on the same rows, for reference. It
            cannot see inclination either, so it degrades too --- which is the point:
            the model is not being singled out.
        tuned_accuracy: The single best cut on ``Delta``, fitted on the coplanar
            training set and applied unchanged.
        model_accuracy: Gradient boosting, likewise fitted coplanar and unchanged.
        model_roc_auc: Ranking quality. Separated from accuracy on purpose: a model
            can keep ordering systems correctly while its 0.5 cut lands in the wrong
            place, and those two failures call for different fixes.
        false_alarm_rate: Fraction of *surviving* systems the model called unstable.
        majority_baseline: Accuracy of predicting the commoner class every time.

            This column is the reason the rest of the table can be read at all.
            Inclination pushes the stable fraction toward 1, and *any* rule that says
            "stable" often enough scores better on a lopsided set --- so an accuracy
            that goes up does not mean the rule got smarter. A number below this
            baseline is a rule that would be beaten by a constant.
        recalibrated_accuracy: Best accuracy reachable by moving the probability cut,
            with the model itself untouched.

            Fitted on the very rows it is scored on, so it is an upper bound and not a
            result --- the same caveat that applies to the tuned threshold in
            :func:`evaluate_stability_models`. It is here to separate "the model no
            longer knows which systems are risky" from "the model still knows, but is
            answering at the wrong cut-off".
        sample_size: Rows scored.
    """

    median_mutual_inclination_deg: float
    stable_fraction: float
    gladman_accuracy: float
    tuned_accuracy: float
    model_accuracy: float
    model_roc_auc: float
    false_alarm_rate: float
    majority_baseline: float
    recalibrated_accuracy: float
    sample_size: int


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


def evaluate_inclination_transfer(
    training: StabilityDataset,
    shifted: Sequence[StabilityDataset],
) -> list[TransferScore]:
    """Score a coplanar-trained model on systems whose orbits are no longer coplanar.

    Every feature in :data:`FEATURE_NAMES` is blind to inclination, so tilting the
    orbits leaves the design matrix untouched and moves only the truth. That makes
    this a clean covariate-free shift: whatever the model loses, it loses because the
    world changed underneath it, not because its inputs did.

    Args:
        training: Coplanar dataset. The model and the tuned threshold are fitted on
            **all** of it --- there is no need to hold anything back here, because the
            evaluation sets are separate draws.
        shifted: Datasets to score, which must come from a different ``seed`` than
            ``training`` or the evaluation is on memorised rows. Passing a coplanar
            dataset first is strongly recommended: without that control there is no
            way to tell an inclination effect apart from an ordinary sampling
            difference.

    Returns:
        One :class:`TransferScore` per entry in ``shifted``, in order.
    """
    separation_index = FEATURE_NAMES.index("hill_separation")
    threshold, _ = best_threshold(
        training.features[:, separation_index], training.labels
    )

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=6, random_state=RANDOM_STATE
    )
    model.fit(training.features, training.labels)

    results: list[TransferScore] = []
    for dataset in shifted:
        labels = dataset.labels
        separation = dataset.features[:, separation_index]
        probabilities = model.predict_proba(dataset.features)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        # A model that has learned "coplanar systems this tight break" keeps saying so
        # once they stop breaking, so the error to watch is the false alarm: called
        # unstable, actually survived.
        actually_stable = labels == 1
        false_alarms = (
            float(np.mean(predictions[actually_stable] == 0))
            if actually_stable.any()
            else float("nan")
        )

        # AUC needs both classes present, which stops being true once almost nothing
        # is unstable --- and that is a result, not an error to paper over.
        both_classes = 0 < labels.mean() < 1
        _, recalibrated = best_threshold(probabilities, labels)

        results.append(
            TransferScore(
                median_mutual_inclination_deg=float(
                    np.median(dataset.mutual_inclination_deg)
                ),
                stable_fraction=dataset.stable_fraction,
                gladman_accuracy=float(
                    accuracy_score(
                        labels, (separation > HILL_STABILITY_THRESHOLD).astype(int)
                    )
                ),
                tuned_accuracy=float(
                    accuracy_score(labels, (separation > threshold).astype(int))
                ),
                model_accuracy=float(accuracy_score(labels, predictions)),
                model_roc_auc=(
                    float(roc_auc_score(labels, probabilities))
                    if both_classes
                    else float("nan")
                ),
                false_alarm_rate=false_alarms,
                majority_baseline=float(max(labels.mean(), 1.0 - labels.mean())),
                recalibrated_accuracy=float(recalibrated),
                sample_size=len(dataset),
            )
        )

    return results


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
