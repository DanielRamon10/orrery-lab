"""Render the three-body figure: where pairwise stability criteria stop working.

Two planets destabilise by approaching each other, and the Hill separation predicts
that well. Three destabilise mainly through **resonance overlap** — each pair sitting
comfortably outside its own Hill limit while resonances belonging to *different* pairs
overlap in between, opening a chaotic band no pairwise number can express.

Four panels:

(a) Survival against the tightest adjacent separation, for two- and three-planet systems
    drawn from the same sampler and run for the same length. The only difference between
    the curves is the third body.
(b) The gap between them, which is the effect isolated.
(c) The separation needed for a given survival rate — the boundary, shifted.
(d) One three-planet system coming apart, with every adjacent pair predicted safe.

Usage::

    python scripts/plot_three_body.py
    python scripts/plot_three_body.py --quick

Writes ``docs/images/phase3-three-body-light.png`` and ``-dark.png``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from orrery.constants import GM_SUN  # noqa: E402
from orrery.nbody import integrate  # noqa: E402
from orrery.stability import (  # noqa: E402
    HILL_STABILITY_THRESHOLD,
    _multiplanet_initial_state,
    build_multiplanet_dataset,
)

THEMES = {
    "light": {
        "surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b",
        "ink_secondary": "#52514e", "ink_muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ("#2a78d6", "#eb6834", "#1baf7a"),
    },
    "dark": {
        "surface": "#1a1a19", "page": "#0d0d0d", "ink": "#ffffff",
        "ink_secondary": "#c3c2b7", "ink_muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": ("#3987e5", "#d95926", "#199e70"),
    },
}

MARKER_SIZE = 7.0
LINE_WIDTH = 1.8
EDGES = np.array([2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0])
SURVIVAL_LEVELS = (0.50, 0.90, 0.99)


def style_axes(ax, theme: dict) -> None:
    ax.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.6, zorder=0)
    ax.tick_params(colors=theme["ink_muted"], labelsize=8, length=3, width=0.6)
    for spine in ax.spines.values():
        spine.set_color(theme["axis"])
        spine.set_linewidth(0.8)


def required_separation(dataset, level: float) -> float | None:
    """Smallest separation at which survival first reaches ``level`` and stays there."""
    centres, rates, _ = dataset.survival_by_separation(np.arange(2.0, 14.5, 0.5), 8)
    for index, centre in enumerate(centres):
        if np.all(rates[index:] >= level):
            return float(centre)
    return None


def plot_survival(ax, theme, runs) -> None:
    style_axes(ax, theme)
    ax.set_title(
        "(a) The same separations, one extra planet",
        color=theme["ink"], fontsize=10, pad=10, loc="left",
    )
    ax.set_xlabel("tightest adjacent separation Δ (mutual Hill radii)",
                  color=theme["ink_muted"], fontsize=8)
    ax.set_ylabel("fraction still stable", color=theme["ink_muted"], fontsize=8)

    for planets, colour in ((2, theme["series"][0]), (3, theme["series"][1])):
        centres, rates, counts = runs[planets].survival_by_separation(EDGES)
        errors = np.sqrt(rates * (1 - rates) / counts)
        ax.errorbar(
            centres, rates, yerr=errors, color=colour, linewidth=LINE_WIDTH,
            marker="o", markersize=MARKER_SIZE - 2, markeredgecolor=theme["surface"],
            markeredgewidth=1.2, capsize=2.5, elinewidth=0.9, zorder=3,
            label=f"{planets} planets",
        )

    ax.axvline(HILL_STABILITY_THRESHOLD, color=theme["ink_secondary"], linewidth=1.2, zorder=2)
    ax.annotate(
        "Gladman", (HILL_STABILITY_THRESHOLD, 0.30), textcoords="offset points",
        xytext=(-13, 0), fontsize=7.5, color=theme["ink_secondary"], rotation=90, va="bottom",
    )
    ax.set_ylim(-0.03, 1.08)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=theme["ink_secondary"])


def plot_gap(ax, theme, runs) -> None:
    style_axes(ax, theme)
    ax.set_title(
        "(b) What the third planet costs",
        color=theme["ink"], fontsize=10, pad=10, loc="left",
    )
    ax.set_xlabel("tightest adjacent separation Δ", color=theme["ink_muted"], fontsize=8)
    ax.set_ylabel("survival lost to the third planet", color=theme["ink_muted"], fontsize=8)

    centres2, rates2, _ = runs[2].survival_by_separation(EDGES)
    centres3, rates3, _ = runs[3].survival_by_separation(EDGES)

    shared = min(len(centres2), len(centres3))
    gap = rates2[:shared] - rates3[:shared]

    ax.axhline(0.0, color=theme["axis"], linewidth=1.0, zorder=1)
    ax.bar(
        centres2[:shared], gap, width=0.55, color=theme["series"][2],
        edgecolor=theme["surface"], linewidth=0.8, zorder=3,
    )
    ax.axvline(HILL_STABILITY_THRESHOLD, color=theme["ink_secondary"], linewidth=1.2, zorder=2)

    ax.text(
        0.5, -0.22,
        "The gap persists well past the point where every pair is individually safe.\n"
        "A criterion that only looks at pairs cannot see it.",
        transform=ax.transAxes, fontsize=7.5, color=theme["ink_muted"],
        ha="center", va="top",
    )


def plot_required_separation(ax, theme, runs) -> None:
    style_axes(ax, theme)
    ax.grid(False, axis="y")
    ax.set_title(
        "(c) How much room three planets need",
        color=theme["ink"], fontsize=10, pad=10, loc="left",
    )
    ax.set_xlabel("separation Δ required (mutual Hill radii)",
                  color=theme["ink_muted"], fontsize=8)

    positions = np.arange(len(SURVIVAL_LEVELS))[::-1]
    height = 0.18

    for offset, (planets, colour) in zip(
        (+height, -height), ((2, theme["series"][0]), (3, theme["series"][1])), strict=True
    ):
        for level, y in zip(SURVIVAL_LEVELS, positions, strict=True):
            value = required_separation(runs[planets], level)
            if value is None:
                continue
            ax.plot([0, value], [y + offset, y + offset], color=colour, linewidth=1.4, zorder=2)
            ax.plot(
                value, y + offset, marker="o", markersize=MARKER_SIZE, color=colour,
                markeredgecolor=theme["surface"], markeredgewidth=1.4, zorder=4,
            )
            ax.annotate(
                f"{value:.1f}", (value, y + offset), textcoords="offset points",
                xytext=(10, -3), fontsize=7.5, color=theme["ink_secondary"],
            )

    ax.axvline(
        HILL_STABILITY_THRESHOLD, color=theme["ink_secondary"],
        linewidth=1.2, linestyle=(0, (5, 4)), zorder=1,
    )
    ax.annotate(
        f"Gladman {HILL_STABILITY_THRESHOLD:.2f}", (HILL_STABILITY_THRESHOLD, positions[0] + 0.42),
        textcoords="offset points", xytext=(5, 0), fontsize=7.5, color=theme["ink_secondary"],
    )

    ax.set_yticks(positions)
    ax.set_yticklabels([f"{level:.0%} survive" for level in SURVIVAL_LEVELS],
                       fontsize=8, color=theme["ink_secondary"])
    ax.set_xlim(0, 12)
    for planets, colour in ((2, theme["series"][0]), (3, theme["series"][1])):
        ax.plot([], [], marker="o", linestyle="none", markersize=MARKER_SIZE,
                color=colour, label=f"{planets} planets")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right",
              labelcolor=theme["ink_secondary"])


def plot_disruption(ax, theme, runs) -> None:
    """One three-planet system coming apart, with every pair predicted safe."""
    style_axes(ax, theme)
    ax.set_title(
        "(d) A system every pairwise test calls safe",
        color=theme["ink"], fontsize=10, pad=10, loc="left",
    )
    ax.set_xlabel("orbits of the innermost planet", color=theme["ink_muted"], fontsize=8)
    ax.set_ylabel("semi-major axis (AU)", color=theme["ink_muted"], fontsize=8)

    dataset = runs[3]
    sample = dataset.sample
    separations = sample.adjacent_hill_separations()

    # A disrupted system whose *every* adjacent pair clears Gladman's threshold.
    safe_pairs = separations.min(axis=1) > HILL_STABILITY_THRESHOLD + 1.0
    candidates = np.flatnonzero((dataset.labels == 0) & safe_pairs)
    if candidates.size == 0:
        ax.text(0.5, 0.5, "no such system in this run", transform=ax.transAxes,
                ha="center", va="center", color=theme["ink_muted"], fontsize=9)
        return

    index = int(candidates[np.argmax(dataset.max_axis_change[candidates])])
    positions, velocities, gms = _multiplanet_initial_state(sample, GM_SUN)

    period = 2.0 * np.pi * np.sqrt(1.0 / GM_SUN)
    trajectory = integrate(
        positions[index], velocities[index], gms[index],
        duration_days=period * dataset.orbits_simulated,
        dt=period / 40, integrator="leapfrog", sample_every=200,
    )

    relative_position = trajectory.positions[:, 1:, :] - trajectory.positions[:, 0:1, :]
    relative_velocity = trajectory.velocities[:, 1:, :] - trajectory.velocities[:, 0:1, :]
    distance = np.linalg.norm(relative_position, axis=-1)
    speed_squared = np.einsum("snk,snk->sn", relative_velocity, relative_velocity)
    mu = GM_SUN + gms[index, 1:]
    axes_history = 1.0 / (2.0 / distance - speed_squared / mu)

    orbits = trajectory.times / period
    for planet in range(3):
        ax.plot(orbits, axes_history[:, planet], color=theme["series"][planet],
                linewidth=1.4, zorder=3, label=f"planet {planet + 1}")

    # A negative semi-major axis means the orbit is no longer bound: that planet has
    # been thrown out. Left alone the axis limits follow it to -280 AU and flatten
    # everything else into a line, hiding the run-up that is the actual physics — so
    # the view is clipped to the bound range and the ejection marked instead.
    bound = axes_history[axes_history > 0]
    ax.set_ylim(0, float(np.percentile(bound, 99.5)) * 1.35)

    escapes = np.flatnonzero(np.any(axes_history < 0, axis=1))
    if escapes.size:
        moment = float(orbits[escapes[0]])
        ax.axvline(moment, color=theme["ink"], linewidth=1.2, zorder=4)
        ax.annotate(
            f"planet ejected\nat orbit {moment:,.0f}",
            (moment, ax.get_ylim()[1] * 0.94),
            textcoords="offset points", xytext=(-9, 0), ha="right", va="top",
            fontsize=7.5, color=theme["ink"], zorder=5,
        )

    ax.legend(frameon=False, fontsize=7.5, loc="lower left", labelcolor=theme["ink_secondary"])
    ax.text(
        0.5, -0.22,
        f"Adjacent separations Δ = {separations[index, 0]:.1f} and {separations[index, 1]:.1f}, "
        "both clear of Gladman's 3.46.\nThe orbits hold for hundreds of revolutions, drift, "
        "and then one planet is thrown out of the system.",
        transform=ax.transAxes, fontsize=7.5, color=theme["ink_muted"],
        ha="center", va="top",
    )


def build_figure(mode, runs, output: Path) -> None:
    theme = THEMES[mode]
    figure, axes = plt.subplots(2, 2, figsize=(13.4, 10.8), facecolor=theme["page"])
    figure.suptitle(
        "orrery-lab · three bodies · where pairwise stability criteria stop working",
        color=theme["ink"], fontsize=13, x=0.045, ha="left", y=0.975,
    )
    figure.text(
        0.045, 0.944,
        "Two- and three-planet systems from the same sampler, the same separations "
        "and the same integration length.",
        color=theme["ink_muted"], fontsize=9, ha="left",
    )

    plot_survival(axes[0][0], theme, runs)
    plot_gap(axes[0][1], theme, runs)
    plot_required_separation(axes[1][0], theme, runs)
    plot_disruption(axes[1][1], theme, runs)

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), h_pad=4.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, facecolor=theme["page"])
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--outdir", default="docs/images")
    args = parser.parse_args()

    count = 400 if args.quick else 1200
    orbits = 400.0 if args.quick else 2000.0

    started = time.perf_counter()
    runs = {}
    for planets in (2, 3):
        print(f"simulating {count:,} systems of {planets} planets for {orbits:.0f} orbits ...")
        runs[planets] = build_multiplanet_dataset(count=count, planets=planets, orbits=orbits)
        print(f"  {runs[planets].stable_fraction:.1%} stable")

    # Console output stays ASCII: a Windows terminal defaults to cp1252 and raises on
    # the Greek letter, which the figure itself renders without any trouble.
    print("\nSurvival by tightest adjacent separation:")
    print(f"{'separation':>14}{'2 planets':>14}{'3 planets':>14}")
    centres2, rates2, counts2 = runs[2].survival_by_separation(EDGES)
    centres3, rates3, counts3 = runs[3].survival_by_separation(EDGES)
    for index in range(min(len(centres2), len(centres3))):
        print(
            f"  {EDGES[index]:>5.1f}-{EDGES[index + 1]:<6.1f}"
            f"{rates2[index]:>12.1%}{rates3[index]:>14.1%}"
        )

    print("\nAbove Gladman's threshold, every adjacent pair predicted safe:")
    for planets in (2, 3):
        dataset = runs[planets]
        above = dataset.min_hill_separation > HILL_STABILITY_THRESHOLD
        print(
            f"  {planets} planets: {dataset.labels[above].mean():.1%} survive "
            f"(n = {int(above.sum()):,})"
        )

    print("\nSeparation required for a given survival rate:")
    for level in SURVIVAL_LEVELS:
        values = [required_separation(runs[planets], level) for planets in (2, 3)]
        rendered = "   ".join(
            f"{planets} planets: {value:.1f}" if value else f"{planets} planets: >14"
            for planets, value in zip((2, 3), values, strict=True)
        )
        print(f"  {level:.0%}:  {rendered}")

    outdir = Path(args.outdir)
    for mode in ("light", "dark"):
        target = outdir / f"phase3-three-body-{mode}.png"
        build_figure(mode, runs, target)
        print(f"\nwrote {target}")

    print(f"\ntotal {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
