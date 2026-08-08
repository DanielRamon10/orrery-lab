"""Render the phase 4 figure: what the numbers support, and what they do not.

Four panels, two of which are deliberately negative results:

(a) **Titius-Bode.** The 18th-century rule, evaluated with no fitted parameters. It
    lands within 5% everywhere out to Uranus — including the asteroid-belt slot,
    which was empty when the rule was written — and then misses Neptune by 29%.
(b) **Resonance clustering.** A Monte Carlo test against a matched random null. The
    answer is that the solar system's adjacent period ratios are *not* unusually
    close to low-order fractions. Shown as the full null distribution rather than a
    lone p-value, because the distribution is the actual result.
(c) **Angular momentum.** The Sun holds 99.8% of the mass and 0.6% of the angular
    momentum. Jupiter holds 61%.
(d) **Six thousand other planetary systems**, for the context a sample of one cannot
    provide — with the selection effects that shape the picture drawn on top of it.

Panel (d) needs a catalogue snapshot::

    python scripts/fetch_exoplanets.py

Everything else works offline. If the snapshot is missing, (d) explains itself and
the other three still render.

Usage::

    python scripts/plot_statistics.py

Writes ``docs/images/phase4-statistics-light.png`` and ``-dark.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from orrery.elements import PLANET_NAMES, PLANETS  # noqa: E402
from orrery.exoplanets import ExoplanetCatalogue, load_catalogue  # noqa: E402
from orrery.statistics import (  # noqa: E402
    angular_momentum_budget,
    find_resonances,
    monte_carlo_resonance_test,
    titius_bode_prediction,
)

# ---------------------------------------------------------------------------
# Theme. Categorical slots 1-3, validated all-pairs in both modes (the scatter in
# panel (d) is an all-pairs form). The light-mode aqua falls below 3:1 contrast, so
# it carries a direct label — the documented relief.
# ---------------------------------------------------------------------------

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "page": "#f9f9f7",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "ink_muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ("#2a78d6", "#eb6834", "#1baf7a"),
    },
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "ink_muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ("#3987e5", "#d95926", "#199e70"),
    },
}

MARKER_SIZE = 7.0
LINE_WIDTH = 1.6


def style_axes(ax, theme: dict) -> None:
    ax.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.6, zorder=0)
    ax.tick_params(colors=theme["ink_muted"], labelsize=8, length=3, width=0.6)
    for spine in ax.spines.values():
        spine.set_color(theme["axis"])
        spine.set_linewidth(0.8)


# ---------------------------------------------------------------------------
# (a) Titius-Bode
# ---------------------------------------------------------------------------


def plot_titius_bode(ax, theme) -> None:
    """Dumbbell per body: predicted against actual.

    The connector length *is* the error, which is why a dumbbell beats two bar
    series here — the reader measures the gap directly instead of subtracting two
    lengths by eye.
    """
    style_axes(ax, theme)
    ax.grid(False, axis="y")
    ax.set_title(
        "(a) Titius–Bode, with no fitted parameters",
        color=theme["ink"],
        fontsize=10,
        pad=10,
        loc="left",
    )
    ax.set_xlabel("distance from the Sun (AU, log scale)", color=theme["ink_muted"], fontsize=8)
    ax.set_xscale("log")

    result = titius_bode_prediction()
    bodies = list(result.predicted_au)
    positions = np.arange(len(bodies))[::-1]

    actual_colour, predicted_colour = theme["series"][0], theme["series"][1]

    for body, y in zip(bodies, positions, strict=True):
        actual = result.actual_au[body]
        predicted = result.predicted_au[body]
        error = result.relative_error(body)

        ax.plot([actual, predicted], [y, y], color=theme["ink_muted"], linewidth=1.2, zorder=2)
        for value, colour in ((actual, actual_colour), (predicted, predicted_colour)):
            ax.plot(
                value,
                y,
                marker="o",
                markersize=MARKER_SIZE,
                color=colour,
                markeredgecolor=theme["surface"],
                markeredgewidth=1.4,
                zorder=4,
            )

        # Only the sizeable errors get a number; labelling every row would bury the
        # one that matters.
        if abs(error) > 0.02:
            ax.annotate(
                f"{error:+.0%}",
                (max(actual, predicted), y),
                textcoords="offset points",
                xytext=(11, -3),
                fontsize=7.5,
                color=theme["ink"] if abs(error) > 0.2 else theme["ink_secondary"],
                fontweight="bold" if abs(error) > 0.2 else "normal",
            )

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [body.capitalize() if body != "ceres" else "Ceres (belt)" for body in bodies],
        fontsize=8,
        color=theme["ink_secondary"],
    )
    left, right = ax.get_xlim()
    ax.set_xlim(left * 0.7, right * 2.4)

    for label, colour in (("actual", actual_colour), ("rule", predicted_colour)):
        ax.plot(
            [], [], marker="o", linestyle="none", markersize=MARKER_SIZE, color=colour,
            markeredgecolor=theme["surface"], markeredgewidth=1.4, label=label,
        )
    # Upper right: the inner-planet rows have no data out at large distance.
    ax.legend(
        loc="upper right", frameon=False, fontsize=7.5,
        labelcolor=theme["ink_secondary"], handletextpad=0.4,
    )

    ax.text(
        0.02,
        0.30,
        "Within 5% for every slot out to Uranus —\n"
        "including the belt, whose slot was empty in 1772.\n"
        "Then Neptune, off by 29%.",
        transform=ax.transAxes,
        fontsize=7.5,
        color=theme["ink_muted"],
        va="top",
    )


# ---------------------------------------------------------------------------
# (b) Resonance Monte Carlo
# ---------------------------------------------------------------------------


def plot_resonance_test(ax, theme, result) -> None:
    """The null distribution, with the observed value marked.

    A single series, so no legend box. Showing the whole distribution rather than
    just the p-value is the point: it makes visible that the observed statistic sits
    in the bulk of the null, not in a tail.
    """
    style_axes(ax, theme)
    ax.set_title(
        "(b) Are the period ratios unusually commensurable?",
        color=theme["ink"],
        fontsize=10,
        pad=10,
        loc="left",
    )
    ax.set_xlabel(
        "mean distance of adjacent period ratios from a low-order fraction",
        color=theme["ink_muted"],
        fontsize=8,
    )
    ax.set_ylabel("random systems", color=theme["ink_muted"], fontsize=8)

    # Log x. The statistic is a positive, strongly right-skewed quantity: on a
    # linear axis 99% of the mass lands in the first bin and the whole distribution
    # collapses to one spike against an empty tail, which hides the comparison this
    # panel exists to show.
    positive = result.null_samples[result.null_samples > 0]
    low = min(positive.min(), result.observed) * 0.7
    high = max(positive.max(), result.observed) * 1.4
    bins = np.logspace(np.log10(low), np.log10(high), 55)

    ax.set_xscale("log")
    ax.hist(
        positive,
        bins=bins,
        color=theme["series"][0],
        alpha=0.85,
        edgecolor=theme["surface"],
        linewidth=0.4,
        zorder=2,
    )

    top = ax.get_ylim()[1]
    ax.axvline(result.observed, color=theme["ink"], linewidth=1.8, zorder=4)
    ax.annotate(
        f"solar system\n{result.observed:.4f}",
        (result.observed, top * 0.97),
        textcoords="offset points",
        xytext=(9, 0),
        fontsize=8,
        color=theme["ink"],
        va="top",
    )

    ax.axvline(
        result.null_median,
        color=theme["ink_muted"],
        linewidth=1.2,
        linestyle=(0, (5, 4)),
        zorder=3,
    )
    ax.annotate(
        f"null median\n{result.null_median:.4f}",
        (result.null_median, top * 0.62),
        textcoords="offset points",
        xytext=(-9, 0),
        fontsize=7.5,
        color=theme["ink_muted"],
        ha="right",
        va="top",
    )

    ax.text(
        0.5,
        -0.20,
        f"p = {result.p_value:.2f} over {result.trials:,} draws. The real system sits in the\n"
        "bulk of the null — slightly farther from simple ratios than chance.\n"
        "Null: same number of bodies and radial span, with log-uniform spacing.",
        transform=ax.transAxes,
        fontsize=7.5,
        color=theme["ink_muted"],
        ha="center",
        va="top",
    )


# ---------------------------------------------------------------------------
# (c) Angular momentum
# ---------------------------------------------------------------------------


def plot_angular_momentum(ax, theme) -> None:
    """Share of the total per body, on a log axis.

    Log, because the range is five orders of magnitude: Mercury holds 0.003% and
    Jupiter 61%. A linear axis would render every terrestrial planet as nothing.
    """
    style_axes(ax, theme)
    ax.grid(False, axis="y")
    ax.set_title(
        "(c) Where the angular momentum actually is",
        color=theme["ink"],
        fontsize=10,
        pad=10,
        loc="left",
    )
    ax.set_xlabel("share of the system total (%, log scale)", color=theme["ink_muted"], fontsize=8)
    ax.set_xscale("log")

    budget = angular_momentum_budget()
    components = ["sun", *PLANET_NAMES]
    positions = np.arange(len(components))[::-1]

    for component, y in zip(components, positions, strict=True):
        share = budget.share(component) * 100.0
        # The Sun is the odd one out — spin, not orbit — so it is inked rather than
        # coloured, marking it as a different kind of quantity.
        colour = theme["ink_secondary"] if component == "sun" else theme["series"][0]

        ax.plot([1e-4, share], [y, y], color=colour, linewidth=1.4, zorder=2)
        ax.plot(
            share, y, marker="o", markersize=MARKER_SIZE, color=colour,
            markeredgecolor=theme["surface"], markeredgewidth=1.4, zorder=4,
        )
        ax.annotate(
            f"{share:.3g}%",
            (share, y),
            textcoords="offset points",
            xytext=(11, -3),
            fontsize=7.5,
            color=theme["ink_secondary"],
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(
        ["Sun (spin)", *[body.capitalize() for body in PLANET_NAMES]],
        fontsize=8,
        color=theme["ink_secondary"],
    )
    ax.set_xlim(1.5e-3, 400.0)

    # Below the axes: every row already reaches the right-hand side, so there is no
    # empty corner inside the plot to put this in.
    ax.text(
        0.5,
        -0.14,
        "The Sun carries 99.8% of the mass and 0.6% of the angular momentum.\n"
        "Any account of how the system formed has to explain that transfer.",
        transform=ax.transAxes,
        fontsize=7.5,
        color=theme["ink_muted"],
        ha="center",
        va="top",
    )


# ---------------------------------------------------------------------------
# (d) Exoplanets
# ---------------------------------------------------------------------------


def plot_exoplanets(ax, theme, catalogue: ExoplanetCatalogue | None) -> None:
    """Mass against period for the confirmed catalogue, solar system overlaid.

    Marker size deliberately departs from the usual minimum: with six thousand
    points, large marks merge into one solid blob and the density structure — which
    is the whole content — disappears. The solar system reference points keep the
    full size, and are inked rather than given a fourth series colour, because they
    are the baseline rather than another category.
    """
    style_axes(ax, theme)
    ax.set_title(
        "(d) 5,981 confirmed exoplanets, and us",
        color=theme["ink"],
        fontsize=10,
        pad=10,
        loc="left",
    )
    ax.set_xlabel("orbital period (days, log scale)", color=theme["ink_muted"], fontsize=8)
    ax.set_ylabel("mass (Earth masses, log scale)", color=theme["ink_muted"], fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")

    if catalogue is None:
        ax.text(
            0.5,
            0.5,
            "No catalogue snapshot found.\n\nRun:  python scripts/fetch_exoplanets.py",
            transform=ax.transAxes,
            fontsize=9,
            color=theme["ink_muted"],
            ha="center",
            va="center",
        )
        return

    usable = catalogue.with_finite("period_days", "mass_earth")

    groups = (
        ("Transit", theme["series"][0]),
        ("Radial Velocity", theme["series"][1]),
    )
    plotted = np.zeros(len(usable), dtype=bool)

    for method, colour in groups:
        mask = usable.mask_by_method(method)
        plotted |= mask
        ax.plot(
            usable.period_days[mask],
            usable.mass_earth[mask],
            linestyle="none",
            marker="o",
            markersize=2.4,
            markeredgewidth=0,
            color=colour,
            alpha=0.42,
            zorder=2,
        )

    other = ~plotted
    if other.any():
        ax.plot(
            usable.period_days[other],
            usable.mass_earth[other],
            linestyle="none",
            marker="o",
            markersize=2.8,
            markeredgewidth=0,
            color=theme["series"][2],
            alpha=0.7,
            zorder=3,
        )

    # Solar system reference, in ink.
    earth_masses = {
        "mercury": 0.0553, "venus": 0.815, "earth": 1.0, "mars": 0.107,
        "jupiter": 317.8, "saturn": 95.16, "uranus": 14.54, "neptune": 17.15,
    }
    # Uranus and Neptune are near-twins in both mass and period, so their labels
    # need pushing apart by hand; the rest are fine offset up and to the right.
    label_offsets = {"uranus": (-34.0, -11.0), "neptune": (8.0, 4.0), "venus": (-38.0, -2.0)}

    for body, mass in earth_masses.items():
        period = PLANETS[body].period_days
        ax.plot(
            period, mass, marker="D", markersize=MARKER_SIZE - 1.0,
            color=theme["ink"], markeredgecolor=theme["surface"], markeredgewidth=1.2,
            zorder=6,
        )
        ax.annotate(
            body.capitalize(),
            (period, mass),
            textcoords="offset points",
            xytext=label_offsets.get(body, (8.0, 4.0)),
            fontsize=7.5,
            color=theme["ink"],
            zorder=6,
        )

    for label, colour in (
        ("Transit", theme["series"][0]),
        ("Radial velocity", theme["series"][1]),
        ("Other methods", theme["series"][2]),
    ):
        ax.plot([], [], marker="o", linestyle="none", markersize=5.5, color=colour, label=label)
    ax.plot(
        [], [], marker="D", linestyle="none", markersize=5.5,
        color=theme["ink"], label="Solar system",
    )
    ax.legend(
        loc="lower right", frameon=False, fontsize=7.5,
        labelcolor=theme["ink_secondary"], handletextpad=0.4, ncol=2,
    )

    # Below the axes: with six thousand points there is no clear space inside.
    ax.text(
        0.5,
        -0.20,
        "The crowd at short periods is largely a selection effect: a transit needs the\n"
        "orbit nearly edge-on, with probability ≈ R★/a, and a radial-velocity signal\n"
        "scales as M/√a. Every solar-system planet sits where both are weakest.",
        transform=ax.transAxes,
        fontsize=7.5,
        color=theme["ink_muted"],
        ha="center",
        va="top",
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_figure(mode, resonance_result, catalogue, output: Path) -> None:
    theme = THEMES[mode]

    # Taller than the other phases' figures: two panels carry their explanatory note
    # below the axes rather than inside, which needs the extra vertical room.
    figure, axes = plt.subplots(2, 2, figsize=(13.6, 11.8), facecolor=theme["page"])
    figure.suptitle(
        "orrery-lab · phase 4 · patterns, and how much they support",
        color=theme["ink"], fontsize=13, x=0.045, ha="left", y=0.975,
    )
    figure.text(
        0.045,
        0.944,
        "Two of these four panels are negative results. Both are reported as such.",
        color=theme["ink_muted"], fontsize=9, ha="left",
    )

    plot_titius_bode(axes[0][0], theme)
    plot_resonance_test(axes[0][1], theme, resonance_result)
    plot_angular_momentum(axes[1][0], theme)
    plot_exoplanets(axes[1][1], theme, catalogue)

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), h_pad=3.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, facecolor=theme["page"])
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20_000)
    parser.add_argument("--outdir", default="docs/images")
    args = parser.parse_args()

    print(f"Monte Carlo resonance test, {args.trials:,} trials ...")
    resonance_result = monte_carlo_resonance_test(trials=args.trials)

    try:
        catalogue = load_catalogue()
        print(f"loaded {len(catalogue)} exoplanets from the cache")
    except FileNotFoundError:
        catalogue = None
        print("no exoplanet snapshot; panel (d) will say so")

    outdir = Path(args.outdir)
    for mode in ("light", "dark"):
        target = outdir / f"phase4-statistics-{mode}.png"
        build_figure(mode, resonance_result, catalogue, target)
        print(f"wrote {target}")

    # The same findings as text, which is also the accessible table view.
    titius = titius_bode_prediction()
    print("\nTitius–Bode, classical rule (no fitted parameters):")
    for body in titius.predicted_au:
        print(
            f"  {body:<9} actual {titius.actual_au[body]:8.4f} AU   "
            f"rule {titius.predicted_au[body]:8.4f} AU   {titius.relative_error(body):+7.2%}"
        )
    others = titius.errors_excluding("neptune")
    print(f"  worst excluding Neptune: {max(abs(v) for v in others.values()):.2%}")

    print("\nLow-order resonances (both integers <= 13, order <= 6):")
    for resonance in find_resonances((*PLANET_NAMES, "pluto"), tolerance=0.02):
        print(f"  {resonance}  order {resonance.order}")

    print(f"\nResonance clustering: p = {resonance_result.p_value:.4f}")
    print(f"  {resonance_result.interpretation}")

    budget = angular_momentum_budget()
    print("\nAngular momentum share:")
    print(f"  {'sun (spin)':<12} {budget.share('sun'):8.4%}")
    for body in PLANET_NAMES:
        print(f"  {body:<12} {budget.share(body):8.4%}")


if __name__ == "__main__":
    main()
