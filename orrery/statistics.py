r"""Statistics on the solar system: what the numbers actually support.

Phases 1 and 3 established *where the planets are*. This module asks what patterns
that arrangement contains, and — the part that matters — how confident one is
allowed to be about each of them. Three of the four questions here have a defensible
statistical answer; one of them mostly does not, and saying so is the point.

What is here
------------
**Kepler's third law**, fitted rather than assumed. The exponent comes out of a
regression with a standard error attached, and is tested against the theoretical
3/2. This is the well-posed case: a real physical law, and the data agree to
machine precision.

**The Titius-Bode rule**, evaluated as a parameter-free prediction rather than fitted.
Its two constants were written down in the 1770s, so comparing them to the data is a
genuine out-of-sample test. It lands within 5% for every slot out to Uranus — including
the asteroid belt, whose slot was empty at the time — and then misses Neptune by 29%.

Alongside it, a *fitted* geometric progression, kept as a cautionary contrast: it
scores an R-squared above 0.99 while making predictions off by a fifth. When the
response grows monotonically across two orders of magnitude, R-squared is close to
uninformative, and the per-point errors are the only thing worth reading.

**Orbital resonances**, found by bounding *both* integers of the approximating
fraction. Bounding only the denominator — the obvious approach, and the one this
module used first — makes every ratio look resonant: Pluto's period is 131.9 times
Mars's, which ``1319/10`` matches to five decimals while implying a 1309th-order
resonance that means nothing. With small integers on both sides, exactly the
commensurabilities the literature discusses fall out and nothing else.

**A Monte Carlo test** of whether those near-commensurabilities are more than
coincidence. The answer, under an explicit matched null, is **no**: the real
adjacent period ratios sit slightly *farther* from low-order fractions than random
systems of the same size and extent. A negative result, reported as one.

Angular momentum
----------------
Also included because it is the most counter-intuitive true fact about the solar
system: the Sun holds 99.8% of the mass and about half a percent of the angular
momentum. Jupiter alone holds around 60%.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import stats

from .constants import AU_KM, GM_BODY, GM_SUN, MEAN_RADIUS_KM, ROTATION_PERIOD_DAYS
from .elements import PLANET_NAMES, PLANETS

__all__ = [
    "LinearFit",
    "fit_line",
    "fit_power_law",
    "keplers_third_law_fit",
    "TitiusBodeResult",
    "titius_bode_distance",
    "titius_bode_prediction",
    "titius_bode_fit",
    "observed_distances",
    "TITIUS_BODE_SLOTS",
    "best_rational_approximation",
    "Resonance",
    "find_resonances",
    "resonance_statistic",
    "ResonanceTest",
    "monte_carlo_resonance_test",
    "AngularMomentumBudget",
    "angular_momentum_budget",
    "bootstrap_slope",
]


# ---------------------------------------------------------------------------
# Regression with uncertainty attached
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearFit:
    """An ordinary-least-squares fit of ``y = intercept + slope * x``.

    Carries the uncertainty rather than just the point estimate. A slope with no
    standard error beside it cannot be compared to a theoretical prediction, which
    is the only thing anyone wants to do with it.

    Attributes:
        slope: Fitted gradient.
        intercept: Fitted offset.
        slope_stderr: Standard error of the slope.
        intercept_stderr: Standard error of the intercept.
        r_squared: Fraction of variance explained.
        residuals: ``y - (intercept + slope * x)``.
        n: Number of points.
    """

    slope: float
    intercept: float
    slope_stderr: float
    intercept_stderr: float
    r_squared: float
    residuals: np.ndarray
    n: int

    @property
    def degrees_of_freedom(self) -> int:
        """``n - 2``: two parameters were estimated from the data."""
        return self.n - 2

    def slope_confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """Two-sided confidence interval for the slope, from the t-distribution."""
        if self.slope_stderr == 0.0:
            return (self.slope, self.slope)
        critical = stats.t.ppf(0.5 + confidence / 2.0, self.degrees_of_freedom)
        margin = critical * self.slope_stderr
        return (self.slope - margin, self.slope + margin)

    def test_slope(self, hypothesised: float) -> tuple[float, float]:
        """Two-sided t-test of ``slope == hypothesised``.

        Returns:
            ``(t_statistic, p_value)``. A large p-value means the data are
            *consistent with* the hypothesis, which is the most a test like this can
            ever say — it is not evidence the hypothesis is exactly true.

        Note:
            When the fit is essentially perfect the standard error underflows to
            zero and the t-statistic is infinite. That is reported as
            ``(inf, 0.0)`` if the slope differs at all, and ``(0.0, 1.0)`` if it
            matches to floating-point precision — which is what happens with
            Kepler's third law, because the "data" are themselves derived from it.
        """
        if self.slope_stderr == 0.0:
            matches = abs(self.slope - hypothesised) < 1e-12
            return (0.0, 1.0) if matches else (float("inf"), 0.0)

        t_statistic = (self.slope - hypothesised) / self.slope_stderr
        p_value = 2.0 * stats.t.sf(abs(t_statistic), self.degrees_of_freedom)
        return (float(t_statistic), float(p_value))


def fit_line(x: np.ndarray, y: np.ndarray) -> LinearFit:
    """Least-squares straight-line fit with standard errors.

    Written out rather than delegated to :func:`scipy.stats.linregress` so that
    every quantity in :class:`LinearFit` is visibly derived from the same residuals.

    Raises:
        ValueError: If fewer than three points are supplied, since the residual
            variance needs at least one degree of freedom.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape; got {x.shape} and {y.shape}")
    if x.size < 3:
        raise ValueError(f"need at least 3 points to estimate an error; got {x.size}")

    n = x.size
    mean_x = x.mean()
    mean_y = y.mean()

    sum_xx = float(np.sum((x - mean_x) ** 2))
    if sum_xx == 0.0:
        raise ValueError("all x values are identical; the slope is undefined")

    slope = float(np.sum((x - mean_x) * (y - mean_y)) / sum_xx)
    intercept = float(mean_y - slope * mean_x)

    residuals = y - (intercept + slope * x)
    residual_sum_squares = float(np.sum(residuals**2))
    total_sum_squares = float(np.sum((y - mean_y) ** 2))

    # Residual variance, using n-2 because two parameters were fitted.
    residual_variance = residual_sum_squares / (n - 2)

    return LinearFit(
        slope=slope,
        intercept=intercept,
        slope_stderr=float(np.sqrt(residual_variance / sum_xx)),
        intercept_stderr=float(
            np.sqrt(residual_variance * (1.0 / n + mean_x**2 / sum_xx))
        ),
        r_squared=1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else 1.0,
        residuals=residuals,
        n=n,
    )


def fit_power_law(x: np.ndarray, y: np.ndarray) -> LinearFit:
    """Fit ``y = C * x**k`` by regressing ``log10 y`` on ``log10 x``.

    The returned slope is the exponent ``k``; ``10**intercept`` is ``C``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("a power law can only be fitted to strictly positive data")
    return fit_line(np.log10(x), np.log10(y))


def keplers_third_law_fit(bodies: tuple[str, ...] = PLANET_NAMES) -> LinearFit:
    """Recover the exponent in ``P**2 ~ a**3`` from the planetary data.

    The exponent 3/2 is never supplied; it comes out of the regression. Because the
    periods are themselves computed from the semi-major axes via the very law being
    tested, this is a *consistency* check on the unit system rather than an
    independent confirmation of Kepler — and it is a sharp one, since any error in
    ``GM`` or in the AU-day convention would move the exponent off 1.5.
    """
    axes = np.array([PLANETS[body].semi_major_axis_au for body in bodies])
    periods = np.array([PLANETS[body].period_days for body in bodies])
    return fit_power_law(axes, periods)


# ---------------------------------------------------------------------------
# Titius-Bode
# ---------------------------------------------------------------------------

#: Slot index ``n`` in the classical rule ``a = 0.4 + 0.3 * 2**n``.
#:
#: Mercury is the special case: the rule's limit as ``n -> -inf`` is 0.4 AU, so
#: Mercury is conventionally assigned that constant term directly, represented here
#: by ``None``.
#:
#: Slot 3 is the **asteroid belt**, not a planet, and it is the historically
#: important one. The gap in the sequence is what sent astronomers hunting there,
#: and Ceres turned up in 1801 at very nearly the predicted distance. Including that
#: slot is not curve-fitting after the fact — the prediction genuinely came first.
TITIUS_BODE_SLOTS: dict[str, int | None] = {
    "mercury": None,
    "venus": 0,
    "earth": 1,
    "mars": 2,
    "ceres": 3,
    "jupiter": 4,
    "saturn": 5,
    "uranus": 6,
    "neptune": 7,
}

#: Mean distance of Ceres from the Sun, in AU. Stands in for the asteroid belt.
CERES_SEMI_MAJOR_AXIS_AU = 2.7658

#: Sequential index for the *fitted* geometric progression, which has no special
#: case for Mercury.
GEOMETRIC_SLOTS: dict[str, int] = {
    body: index for index, body in enumerate(TITIUS_BODE_SLOTS)
}


def titius_bode_distance(slot: int | None) -> float:
    """The classical rule ``a = 0.4 + 0.3 * 2**n``, in AU.

    A *parameter-free prediction*, not a fit — the constants 0.4 and 0.3 were
    written down in the 18th century and are not adjusted to the data here. That is
    what makes the comparison in :func:`titius_bode_prediction` meaningful.
    """
    if slot is None:
        return 0.4  # Mercury: the rule's limiting value
    return 0.4 + 0.3 * 2.0**slot


def observed_distances() -> dict[str, float]:
    """Semi-major axis of every Titius-Bode slot, with Ceres for the belt."""
    return {
        body: (
            CERES_SEMI_MAJOR_AXIS_AU
            if body == "ceres"
            else PLANETS[body].semi_major_axis_au
        )
        for body in TITIUS_BODE_SLOTS
    }


@dataclass(frozen=True)
class TitiusBodeResult:
    """How well a distance rule reproduces the planets.

    Attributes:
        predicted_au: Predicted distance per body.
        actual_au: Observed distance per body.
        label: Which rule produced the predictions.
        fit: The regression, for the fitted variant; ``None`` for the classical rule,
            which has no free parameters.
    """

    predicted_au: dict[str, float]
    actual_au: dict[str, float]
    label: str
    fit: LinearFit | None = None

    def relative_error(self, body: str) -> float:
        """Signed fractional error of the prediction for one body."""
        return (self.predicted_au[body] - self.actual_au[body]) / self.actual_au[body]

    @property
    def worst_body(self) -> str:
        """Whichever body the rule fits least well."""
        return max(self.predicted_au, key=lambda body: abs(self.relative_error(body)))

    @property
    def max_absolute_error(self) -> float:
        """Largest absolute relative error over all bodies."""
        return max(abs(self.relative_error(body)) for body in self.predicted_au)

    def errors_excluding(self, *bodies: str) -> dict[str, float]:
        """Per-body errors with some bodies dropped, for isolating one failure."""
        return {
            body: self.relative_error(body)
            for body in self.predicted_au
            if body not in bodies
        }


def titius_bode_prediction() -> TitiusBodeResult:
    r"""Evaluate the classical rule :math:`a_n = 0.4 + 0.3 \cdot 2^n`.

    **No parameters are fitted.** The two constants come from Titius (1766) and Bode
    (1772), and the slot assignment from :data:`TITIUS_BODE_SLOTS`. Every number
    below is therefore a genuine out-of-sample prediction against data, most of which
    was collected afterwards.

    The result is the interesting one: the rule lands within about 5% for every body
    out to Uranus — including the asteroid belt, whose slot was empty when the rule
    was written — and then misses **Neptune by roughly 29%**. Pluto, meanwhile, sits
    close to the slot Neptune failed to occupy.

    Warning:
        A pattern getting eight of nine right is not the same as a law. There is no
        accepted physical derivation, the slot assignment involves judgement (Mercury
        is a special case, the belt counts as one slot, Pluto is left out), and nine
        points is very few. Treat it as a striking regularity with an unexplained
        origin, and see :func:`titius_bode_fit` for why the usual goodness-of-fit
        statistic is not the reassurance it appears to be.
    """
    actual = observed_distances()
    predicted = {
        body: titius_bode_distance(slot) for body, slot in TITIUS_BODE_SLOTS.items()
    }
    return TitiusBodeResult(
        predicted_au=predicted, actual_au=actual, label="classical 0.4 + 0.3 x 2^n"
    )


def titius_bode_fit(exclude: tuple[str, ...] = ()) -> TitiusBodeResult:
    r"""Fit a plain geometric progression :math:`a_n = a_0 r^n` instead.

    Provided as a **cautionary contrast** rather than as an improvement. Regressing
    ``log a`` on the slot index gives an R-squared around 0.99, which looks like a
    resounding success — and yet the individual predictions are off by up to 20%,
    far worse than the classical rule's 5%.

    The lesson generalises well beyond astronomy: R-squared measures how much of the
    *variance* a model explains, and when the response spans two orders of magnitude
    monotonically, almost any increasing curve explains nearly all of it. With nine
    points and a quantity growing like this, a high R-squared carries very little
    information. The per-point errors do.

    Args:
        exclude: Bodies held out of the fit but still predicted.
    """
    actual = observed_distances()
    fitted = tuple(body for body in GEOMETRIC_SLOTS if body not in exclude)

    slots = np.array([GEOMETRIC_SLOTS[body] for body in fitted], dtype=float)
    values = np.array([actual[body] for body in fitted])
    fit = fit_line(slots, np.log10(values))

    predicted = {
        body: float(10.0 ** (fit.intercept + fit.slope * slot))
        for body, slot in GEOMETRIC_SLOTS.items()
    }
    return TitiusBodeResult(
        predicted_au=predicted,
        actual_au=actual,
        label="fitted geometric progression",
        fit=fit,
    )


# ---------------------------------------------------------------------------
# Resonances
# ---------------------------------------------------------------------------


def best_rational_approximation(value: float, max_integer: int = 12) -> Fraction:
    """Closest fraction ``p/q`` to ``value`` with **both** ``p`` and ``q`` bounded.

    Bounding only the denominator — which is what
    :meth:`fractions.Fraction.limit_denominator` does — is useless here. With the
    numerator free, every real number has a close approximation: Pluto's period is
    131.9 times Mars's, and ``1319/10`` matches that to five decimal places while
    meaning nothing whatsoever. A "resonance" of order 1309 is not a resonance.

    Physically meaningful commensurabilities involve *small integers on both sides*,
    so both are bounded and the search is a short exhaustive scan. Exhaustive is
    exact, and with a bound in the low tens it is also instant.

    Args:
        value: A positive ratio.
        max_integer: Largest value allowed for either ``p`` or ``q``.

    Example:
        >>> best_rational_approximation(2.4816, max_integer=12)
        Fraction(5, 2)
        >>> best_rational_approximation(131.9007, max_integer=12)
        Fraction(12, 1)
    """
    if value <= 0:
        raise ValueError(f"period ratios are positive; got {value}")
    if max_integer < 1:
        raise ValueError(f"max_integer must be at least 1; got {max_integer}")

    best = Fraction(1, 1)
    best_error = abs(value - 1.0)

    for denominator in range(1, max_integer + 1):
        # The nearest numerator for this denominator, clamped to the bound.
        numerator = min(max(round(value * denominator), 1), max_integer)
        error = abs(value - numerator / denominator)
        if error < best_error:
            best, best_error = Fraction(numerator, denominator), error

    return best


@dataclass(frozen=True)
class Resonance:
    """A near-commensurability between two orbital periods.

    Attributes:
        inner: Name of the inner (shorter-period) body.
        outer: Name of the outer body.
        period_ratio: ``P_outer / P_inner``, always greater than 1.
        p: Numerator of the closest simple ratio.
        q: Denominator.
        relative_error: ``|ratio - p/q| / (p/q)``.
    """

    inner: str
    outer: str
    period_ratio: float
    p: int
    q: int
    relative_error: float

    @property
    def order(self) -> int:
        """``|p - q|`` — the resonance order.

        Low-order resonances are the dynamically strong ones, so this is the number
        that decides whether a commensurability matters physically.
        """
        return abs(self.p - self.q)

    def __str__(self) -> str:
        return (
            f"{self.outer}:{self.inner} = {self.p}:{self.q} "
            f"(ratio {self.period_ratio:.4f}, off by {self.relative_error:.2%})"
        )


def find_resonances(
    bodies: tuple[str, ...] = PLANET_NAMES,
    max_integer: int = 13,
    max_order: int = 6,
    tolerance: float = 0.03,
    adjacent_only: bool = False,
) -> list[Resonance]:
    """Search body pairs for near-integer period commensurabilities.

    Args:
        bodies: Bodies to consider, any order.
        max_integer: Bound on both ``p`` and ``q``. Raising it finds more matches
            that mean less.
        max_order: Bound on ``|p - q|``. This is the filter that does the real work:
            resonance strength falls off sharply with order, so a 53rd-order
            "commensurability" is a numerical coincidence rather than a dynamical
            relationship. Set high to see everything the numbers permit.
        tolerance: Maximum relative error to report.
        adjacent_only: Restrict to neighbouring pairs, which are the ones with any
            real dynamical coupling.

    Returns:
        Matches sorted by relative error, closest first.

    Example:
        The defaults surface exactly the commensurabilities the literature discusses
        — Venus-Earth 13:8, Jupiter-Saturn 5:2, Uranus-Neptune 2:1 — and nothing else.
    """
    ordered = sorted(bodies, key=lambda body: PLANETS[body].semi_major_axis_au)
    found: list[Resonance] = []

    for inner_index, inner in enumerate(ordered):
        for outer_index in range(inner_index + 1, len(ordered)):
            if adjacent_only and outer_index != inner_index + 1:
                continue

            outer = ordered[outer_index]
            ratio = PLANETS[outer].period_days / PLANETS[inner].period_days
            fraction = best_rational_approximation(ratio, max_integer)
            approximation = fraction.numerator / fraction.denominator
            error = abs(ratio - approximation) / approximation

            if error <= tolerance and abs(fraction.numerator - fraction.denominator) <= max_order:
                found.append(
                    Resonance(
                        inner=inner,
                        outer=outer,
                        period_ratio=float(ratio),
                        p=fraction.numerator,
                        q=fraction.denominator,
                        relative_error=float(error),
                    )
                )

    return sorted(found, key=lambda resonance: resonance.relative_error)


def resonance_statistic(
    semi_major_axes: np.ndarray,
    max_integer: int = 13,
) -> float:
    """Mean closeness of adjacent period ratios to a simple fraction.

    The test statistic for :func:`monte_carlo_resonance_test`. Smaller means the
    system's adjacent pairs sit nearer to low-order commensurabilities.

    Periods come from Kepler's third law, so only the axes are needed — which is
    what lets the Monte Carlo generate synthetic systems by drawing axes alone.
    """
    axes = np.sort(np.asarray(semi_major_axes, dtype=float))
    periods = axes**1.5  # any consistent unit; only ratios are used

    errors = []
    for index in range(len(periods) - 1):
        ratio = periods[index + 1] / periods[index]
        fraction = best_rational_approximation(float(ratio), max_integer)
        approximation = fraction.numerator / fraction.denominator
        errors.append(abs(ratio - approximation) / approximation)

    return float(np.mean(errors))


@dataclass(frozen=True)
class ResonanceTest:
    """Result of the Monte Carlo test for resonance clustering.

    Attributes:
        observed: The statistic for the real solar system.
        null_samples: Statistic for each synthetic system.
        p_value: Fraction of synthetic systems at least as extreme as observed.
        trials: Number of synthetic systems drawn.
    """

    observed: float
    null_samples: np.ndarray
    p_value: float
    trials: int

    @property
    def null_median(self) -> float:
        return float(np.median(self.null_samples))

    @property
    def interpretation(self) -> str:
        """A deliberately hedged one-line reading of the p-value."""
        if self.p_value < 0.01:
            strength = "strong"
        elif self.p_value < 0.05:
            strength = "moderate"
        elif self.p_value < 0.2:
            strength = "weak"
        else:
            strength = "no"
        return (
            f"{strength} evidence against the null (p = {self.p_value:.3f}); "
            "note the result depends on how the null was constructed"
        )


def monte_carlo_resonance_test(
    bodies: tuple[str, ...] = PLANET_NAMES,
    trials: int = 20_000,
    max_integer: int = 13,
    seed: int = 20260726,
) -> ResonanceTest:
    """Are the planets' period ratios nearer to simple fractions than chance?

    **The null hypothesis, stated explicitly**, because the answer depends entirely
    on it: the same number of bodies, spanning the same radial range as the real
    system, with ``log10(a)`` drawn uniformly at random and then sorted.

    That null holds the body count and the overall extent fixed, and randomises only
    the spacing. It is a reasonable choice, and it is not the only reasonable choice
    — a null that also randomised the range, or drew from the observed spacing
    distribution, would give a different p-value. This is why the function returns
    the null samples too: the distribution is the result, and the single number is a
    summary of it.

    Args:
        bodies: Real bodies to test.
        trials: Synthetic systems to draw.
        max_integer: Passed to :func:`resonance_statistic`.
        seed: Fixed so the p-value is reproducible.

    Returns:
        A :class:`ResonanceTest`. Small ``p_value`` means the real spacing is closer
        to commensurable than the null typically produces.
    """
    axes = np.array([PLANETS[body].semi_major_axis_au for body in bodies])
    observed = resonance_statistic(axes, max_integer)

    log_low, log_high = np.log10(axes.min()), np.log10(axes.max())
    generator = np.random.default_rng(seed)

    null_samples = np.empty(trials)
    for trial in range(trials):
        synthetic = 10.0 ** generator.uniform(log_low, log_high, size=len(axes))
        null_samples[trial] = resonance_statistic(synthetic, max_integer)

    # One-sided: the alternative is "closer to commensurable", i.e. a smaller
    # statistic. The +1 corrections keep the p-value from ever being exactly zero,
    # which would overstate what a finite number of trials can show.
    p_value = float((np.sum(null_samples <= observed) + 1) / (trials + 1))

    return ResonanceTest(
        observed=float(observed),
        null_samples=null_samples,
        p_value=p_value,
        trials=trials,
    )


# ---------------------------------------------------------------------------
# Angular momentum
# ---------------------------------------------------------------------------

#: Moment-of-inertia factor of the Sun, ``I / (M R^2)``.
#:
#: A uniform sphere would be 0.4. The Sun is strongly centrally condensed, so almost
#: all of its mass sits at small radius and contributes little rotational inertia.
#: Helioseismology puts the factor near 0.070.
SUN_MOMENT_OF_INERTIA_FACTOR = 0.070


@dataclass(frozen=True)
class AngularMomentumBudget:
    """Where the solar system's angular momentum actually is.

    All values are ``G`` times the true angular momentum, in AU^2/day units — the
    same GM convention as :mod:`orrery.nbody`. Only the shares matter, and those are
    unaffected.

    Attributes:
        orbital: Orbital angular momentum per planet.
        sun_rotational: The Sun's spin angular momentum.
        total: Everything summed.
    """

    orbital: dict[str, float]
    sun_rotational: float
    total: float

    def share(self, component: str) -> float:
        """Fraction of the total held by one planet, or by ``"sun"``."""
        value = self.sun_rotational if component == "sun" else self.orbital[component]
        return value / self.total

    @property
    def dominant_body(self) -> str:
        return max(self.orbital, key=lambda body: self.orbital[body])


def angular_momentum_budget(
    bodies: tuple[str, ...] = PLANET_NAMES,
) -> AngularMomentumBudget:
    r"""Compute the angular momentum held by each planet and by the Sun's spin.

    Orbital angular momentum uses the closed form for an ellipse:

    .. math::  L_i = GM_i \sqrt{GM_\odot\, a_i (1 - e_i^2)}

    The Sun's rotational term is :math:`I\omega` with :math:`I = k M R^2`.

    The result is the most counter-intuitive true statement about the solar system:
    the Sun carries 99.8% of the mass and well under one percent of the angular
    momentum. Any theory of how the system formed has to explain that transfer.
    """
    orbital: dict[str, float] = {}
    for body in bodies:
        elements = PLANETS[body]
        specific = np.sqrt(
            GM_SUN * elements.semi_major_axis_au * (1.0 - elements.eccentricity**2)
        )
        orbital[body] = float(GM_BODY[body] * specific)

    radius_au = MEAN_RADIUS_KM["sun"] / AU_KM
    angular_speed = 2.0 * np.pi / ROTATION_PERIOD_DAYS["sun"]  # radians per day
    sun_rotational = float(
        SUN_MOMENT_OF_INERTIA_FACTOR * GM_SUN * radius_au**2 * angular_speed
    )

    return AngularMomentumBudget(
        orbital=orbital,
        sun_rotational=sun_rotational,
        total=sum(orbital.values()) + sun_rotational,
    )


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def bootstrap_slope(
    x: np.ndarray,
    y: np.ndarray,
    trials: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260726,
) -> tuple[float, float, np.ndarray]:
    """Bootstrap confidence interval for a fitted slope.

    Resamples the ``(x, y)`` pairs with replacement. Useful as a cross-check on the
    t-based interval from :meth:`LinearFit.slope_confidence_interval`, because the
    bootstrap makes no assumption that the residuals are normal — with eight planets
    that assumption is doing real work and deserves testing.

    Returns:
        ``(low, high, samples)``, the interval bounds and every bootstrap slope.

    Note:
        With a sample this small the bootstrap is itself shaky: many resamples will
        contain duplicated points and a few will be nearly degenerate. Treat wide
        disagreement with the t-interval as a sign that neither should be trusted
        rather than as a reason to prefer one.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    generator = np.random.default_rng(seed)

    slopes = []
    for _ in range(trials):
        indices = generator.integers(0, len(x), size=len(x))
        resampled_x, resampled_y = x[indices], y[indices]
        # A resample can draw the same point repeatedly, leaving no spread in x and
        # no defined slope; skip those rather than poison the distribution.
        if np.ptp(resampled_x) == 0:
            continue
        mean_x, mean_y = resampled_x.mean(), resampled_y.mean()
        slopes.append(
            float(
                np.sum((resampled_x - mean_x) * (resampled_y - mean_y))
                / np.sum((resampled_x - mean_x) ** 2)
            )
        )

    samples = np.array(slopes)
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(samples, tail)),
        float(np.quantile(samples, 1.0 - tail)),
        samples,
    )
