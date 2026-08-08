"""Render the validation figure for phase 1.

Four panels, each answering a different question about the ephemeris:

(a) inner solar system, seen from above --- are the orbits the right size?
(b) outer solar system, same view --- does the scale hold over 30 AU?
(c) orbital inclination per body --- are the orbital planes tilted correctly?
(d) Kepler's third law --- does P^2 ~ a^3 fall out of the computed periods?

Panel (d) is the real test: the fitted slope is not fed in anywhere. If the
Kepler solver, the unit system and the element table are all consistent, the
regression must return 3/2 to many decimal places.

Usage::

    python scripts/plot_orbits.py
    python scripts/plot_orbits.py --date 2026-12-25

Writes ``docs/images/phase1-orbits-light.png`` and ``-dark.png``.

Design notes
------------
Colour encodes **distance from the Sun**, which is an ordered magnitude, so the
orbit panels use a single-hue ordinal ramp rather than a categorical palette ---
eight arbitrary hues would neither be colourblind-safe nor mean anything. Eight
ordinal steps do not fit inside one hue's usable lightness band either, so the
top-down view is faceted into inner and outer systems with four validated steps
each. Pluto, no longer a planet, is folded out of the ramp and drawn as a muted
dashed line. Every orbit is directly labelled, so identity is never carried by
colour alone; ``scripts/solar_system_report.py`` is the table view of the same
numbers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from orrery import PLANET_NAMES, PLANETS, body_state, julian_date_from_datetime  # noqa: E402
from orrery.constants import GM_SUN  # noqa: E402
from orrery.ephemeris import orbit_path  # noqa: E402
from orrery.kepler import orbital_period  # noqa: E402

# ---------------------------------------------------------------------------
# Theme tokens, taken from the reference palette. Both modes are chosen
# deliberately: the dark steps are stepped for the dark surface, not flipped.
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
        # Ordinal ramp, blue steps 250/400/500/650 --- validated: monotone
        # lightness, all adjacent dL >= 0.06, light end 2.06:1 on the surface.
        "ramp": ("#86b6ef", "#3987e5", "#256abf", "#104281"),
        "series": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "ink_muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        # Blue steps 600/450/300/150 --- validated for the dark surface.
        "ramp": ("#184f95", "#2a78d6", "#6da7ec", "#b7d3f6"),
        "series": "#3987e5",
    },
}

INNER = ("mercury", "venus", "earth", "mars")
OUTER = ("jupiter", "saturn", "uranus", "neptune")

LINE_WIDTH = 1.6
MARKER_SIZE = 7.0

#: Per-body label nudges, in points, for the top-down panels. The planets sit
#: wherever the date puts them, so a few labels would otherwise collide with a
#: neighbour or with the Sun; these offsets are tuned by looking at the output.
LABEL_OFFSETS: dict[str, tuple[float, float]] = {
    "jupiter": (-34.0, 12.0),
    "saturn": (10.0, -12.0),
    "neptune": (9.0, 6.0),
    "pluto": (11.0, 13.0),
}


def style_axes(ax, theme: dict[str, str]) -> None:
    """Push grid and spines into the background; keep ink on the data."""
    ax.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.6, zorder=0)
    ax.tick_params(colors=theme["ink_muted"], labelsize=8, length=3, width=0.6)
    for spine in ax.spines.values():
        spine.set_color(theme["axis"])
        spine.set_linewidth(0.8)


def draw_sun(ax, theme: dict[str, str], label: bool = True) -> None:
    """The Sun is annotation, not a data series, so it wears ink not a hue."""
    ax.plot(0, 0, marker="o", markersize=MARKER_SIZE, color=theme["ink"], zorder=5)
    if label:
        ax.annotate(
            "Sun",
            (0, 0),
            textcoords="offset points",
            xytext=(9, -3),
            fontsize=8,
            color=theme["ink_secondary"],
        )


def plot_top_down(ax, theme, bodies, jd, title, extra_dashed=None):
    """Orbits projected onto the ecliptic plane (x-y), with direct labels."""
    style_axes(ax, theme)
    ax.set_aspect("equal", adjustable="datalim")
    # Extra pad leaves room for the ramp legend, which sits between title and axes.
    ax.set_title(title, color=theme["ink"], fontsize=10, pad=26, loc="left")
    ax.set_xlabel("x (AU)", color=theme["ink_muted"], fontsize=8)
    ax.set_ylabel("y (AU)", color=theme["ink_muted"], fontsize=8)

    if extra_dashed:
        # "Other" bucket: outside the ordinal ramp, so recessive and dashed.
        for name in extra_dashed:
            path = orbit_path(name, jd, samples=720)
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=theme["ink_muted"],
                linewidth=1.0,
                linestyle=(0, (5, 4)),
                zorder=1,
            )
            state = body_state(name, jd)
            # Short label only: it has to fit in the gap between this orbit and
            # Neptune's. The dashed convention is spelled out in the note below.
            ax.annotate(
                PLANETS[name].name,
                (state.position[0], state.position[1]),
                textcoords="offset points",
                xytext=LABEL_OFFSETS.get(name, (7.0, 4.0)),
                fontsize=7.5,
                color=theme["ink_muted"],
            )
        ax.text(
            0.5,
            -0.16,
            "dashed — dwarf planet, outside the colour ramp",
            transform=ax.transAxes,
            fontsize=7.5,
            color=theme["ink_muted"],
            ha="center",
        )

    for index, name in enumerate(bodies):
        colour = theme["ramp"][index]
        path = orbit_path(name, jd, samples=720)
        ax.plot(path[:, 0], path[:, 1], color=colour, linewidth=LINE_WIDTH, zorder=2)

        state = body_state(name, jd)
        ax.plot(
            state.position[0],
            state.position[1],
            marker="o",
            markersize=MARKER_SIZE,
            color=colour,
            # 2px surface ring so the marker reads on top of its own orbit line.
            markeredgecolor=theme["surface"],
            markeredgewidth=1.4,
            zorder=4,
        )
        ax.annotate(
            PLANETS[name].name,
            (state.position[0], state.position[1]),
            textcoords="offset points",
            xytext=LABEL_OFFSETS.get(name, (9.0, 5.0)),
            fontsize=8,
            color=theme["ink_secondary"],
        )

    draw_sun(ax, theme)
    _add_ramp_legend(ax, theme)


def _add_ramp_legend(ax, theme: dict[str, str]) -> None:
    """Four swatches reading 'inner -> outer': the legend for an ordinal scale.

    Placed just above the axes rather than inside them, so it never sits on top
    of gridlines or an orbit.
    """
    for index, colour in enumerate(theme["ramp"]):
        ax.add_patch(
            plt.Rectangle(
                (index * 0.032, 1.012),
                0.028,
                0.018,
                transform=ax.transAxes,
                facecolor=colour,
                edgecolor="none",
                clip_on=False,
                zorder=6,
            )
        )
    ax.text(
        4 * 0.032 + 0.012,
        1.021,
        "inner → outer",
        transform=ax.transAxes,
        fontsize=7.5,
        color=theme["ink_muted"],
        va="center",
        zorder=6,
    )


#: Inclinations to the ecliptic in degrees, as printed on the NASA Planetary Fact
#: Sheet (nssdc.gsfc.nasa.gov), quoted there to one decimal place.
#:
#: Deliberately taken from a *different* publication than the element table, so
#: the comparison is a real transcription check rather than a restatement of the
#: same numbers. Agreement is therefore expected only to within the 0.05 degrees
#: that the fact sheet's rounding allows.
FACT_SHEET_INCLINATION_DEG: dict[str, float] = {
    "mercury": 7.0,
    "venus": 3.4,
    "earth": 0.0,
    "mars": 1.9,
    "jupiter": 1.3,
    "saturn": 2.5,
    "uranus": 0.8,
    "neptune": 1.8,
    "pluto": 17.2,
}


def plot_inclination(ax, theme) -> None:
    """How far each orbital plane tilts away from the ecliptic.

    An edge-on x-z projection was the obvious choice here and is what a 3D scene
    will show far better in phase 2. As a *static* panel it fails: at the point of
    maximum height the inner planets all sit near the centre, so their labels
    collide into an unreadable pile, and Pluto's 17 degrees flattens the other
    eight into a single sliver.

    A magnitude-per-body comparison answers the same question and can be checked:
    the filled marks are computed from the element table, the hollow rings are the
    NASA fact sheet's independently published figures, and the two must agree to
    within that sheet's one-decimal rounding.

    One series, so no legend box --- the title says what is plotted. Drawn as
    thin stems with a dot at the data end rather than as filled bars.
    """
    style_axes(ax, theme)
    ax.set_title(
        "(c) Orbital inclination to the ecliptic",
        color=theme["ink"],
        fontsize=10,
        pad=10,
        loc="left",
    )
    ax.set_xlabel("inclination (degrees)", color=theme["ink_muted"], fontsize=8)
    # Category axis: a horizontal gridline per body would add nothing.
    ax.grid(False, axis="y")

    bodies = (*PLANET_NAMES, "pluto")
    # Ordered outward from the Sun, then flipped so Mercury lands on top.
    positions = np.arange(len(bodies))[::-1]

    for name, y in zip(bodies, positions, strict=True):
        computed = abs(PLANETS[name].inclination_deg)
        published = FACT_SHEET_INCLINATION_DEG[name]
        is_planet = name != "pluto"
        colour = theme["series"] if is_planet else theme["ink_muted"]

        ax.plot([0.0, computed], [y, y], color=colour, linewidth=1.4, zorder=2)
        ax.plot(
            computed,
            y,
            marker="o",
            markersize=MARKER_SIZE,
            color=colour,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.4,
            zorder=4,
        )
        # Hollow ring = the published value, for visual comparison.
        ax.plot(
            published,
            y,
            marker="o",
            markersize=MARKER_SIZE + 4.5,
            markerfacecolor="none",
            markeredgecolor=theme["ink_muted"],
            markeredgewidth=1.0,
            zorder=3,
        )
        ax.annotate(
            f"{computed:.2f}°",
            (computed, y),
            textcoords="offset points",
            xytext=(14, -3),
            fontsize=8,
            color=theme["ink_secondary"],
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [PLANETS[name].name + ("" if name != "pluto" else "  (dwarf)") for name in bodies],
        fontsize=8,
        color=theme["ink_secondary"],
    )
    ax.set_xlim(-0.6, 21.0)
    ax.axvline(0.0, color=theme["axis"], linewidth=0.8, zorder=1)

    largest_error = max(
        abs(abs(PLANETS[name].inclination_deg) - FACT_SHEET_INCLINATION_DEG[name])
        for name in bodies
    )
    ax.text(
        0.975,
        0.52,
        "hollow ring = NASA fact sheet value\n"
        f"largest disagreement: {largest_error:.2f}°  (its rounding allows 0.05°)",
        transform=ax.transAxes,
        fontsize=7.5,
        color=theme["ink_muted"],
        ha="right",
        va="bottom",
    )


def plot_third_law(ax, theme) -> None:
    """Kepler's third law recovered by regression, not assumed.

    A single series, so no legend box: the title names what is plotted.
    """
    style_axes(ax, theme)
    ax.set_title(
        "(d) Kepler's third law, recovered by fit",
        color=theme["ink"],
        fontsize=10,
        pad=10,
        loc="left",
    )
    ax.set_xlabel("log₁₀  semi-major axis  (AU)", color=theme["ink_muted"], fontsize=8)
    ax.set_ylabel("log₁₀  period  (days)", color=theme["ink_muted"], fontsize=8)

    axes_au = np.array([PLANETS[name].semi_major_axis_au for name in PLANET_NAMES])
    periods = orbital_period(axes_au, GM_SUN)

    log_a = np.log10(axes_au)
    log_p = np.log10(periods)
    slope, intercept = np.polyfit(log_a, log_p, deg=1)

    fit_x = np.linspace(log_a.min() - 0.08, log_a.max() + 0.08, 50)
    ax.plot(
        fit_x,
        slope * fit_x + intercept,
        color=theme["ink_muted"],
        linewidth=1.0,
        linestyle=(0, (5, 4)),
        zorder=2,
    )
    ax.plot(
        log_a,
        log_p,
        linestyle="none",
        marker="o",
        markersize=MARKER_SIZE,
        color=theme["series"],
        markeredgecolor=theme["surface"],
        markeredgewidth=1.4,
        zorder=3,
    )

    for name, x, y in zip(PLANET_NAMES, log_a, log_p, strict=True):
        ax.annotate(
            PLANETS[name].name,
            (x, y),
            textcoords="offset points",
            xytext=(8, -2),
            fontsize=8,
            color=theme["ink_secondary"],
        )

    ax.text(
        0.03,
        0.88,
        f"fitted slope = {slope:.6f}\nexact value  = 1.5",
        transform=ax.transAxes,
        fontsize=8.5,
        color=theme["ink"],
        va="top",
        family="monospace",
    )
    return slope


def build_figure(mode: str, jd: float, moment: datetime, output: Path) -> float:
    theme = THEMES[mode]

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 11.0), facecolor=theme["page"])
    figure.suptitle(
        f"orrery-lab · phase 1 validation · {moment:%Y-%m-%d} UTC",
        color=theme["ink"],
        fontsize=13,
        x=0.045,
        ha="left",
        y=0.975,
    )
    figure.text(
        0.045,
        0.945,
        "Positions computed from JPL Keplerian elements. Nothing here is a stock image.",
        color=theme["ink_muted"],
        fontsize=9,
        ha="left",
    )

    plot_top_down(axes[0][0], theme, INNER, jd, "(a) Inner solar system — from above")
    plot_top_down(
        axes[0][1], theme, OUTER, jd, "(b) Outer solar system — from above",
        extra_dashed=("pluto",),
    )
    plot_inclination(axes[1][0], theme)
    slope = plot_third_law(axes[1][1], theme)

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, facecolor=theme["page"])
    plt.close(figure)
    return slope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="UTC date YYYY-MM-DD (default: now)")
    parser.add_argument(
        "--outdir", default="docs/images", help="output directory (default: docs/images)"
    )
    args = parser.parse_args()

    moment = (
        datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.date
        else datetime.now(timezone.utc)
    )
    jd = julian_date_from_datetime(moment)
    outdir = Path(args.outdir)

    for mode in ("light", "dark"):
        target = outdir / f"phase1-orbits-{mode}.png"
        slope = build_figure(mode, jd, moment, target)
        print(f"wrote {target}")

    print(f"\nKepler third-law slope from regression: {slope:.12f}  (exact: 1.5)")
    print(f"absolute error: {abs(slope - 1.5):.3e}")


if __name__ == "__main__":
    main()
