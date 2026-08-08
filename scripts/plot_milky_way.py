"""Render the phase 6 figure: 120,000 real stars, and what the sample is not.

Four panels:

(a) **The Hertzsprung–Russell diagram**, from the whole sample. Colour against
    intrinsic brightness — the plot that revealed how stars live and die. The main
    sequence and the giant branch are both visible, and the giants dominate.
(b) **The same diagram within 25 parsecs.** Close enough that the survey sees almost
    everything, so the main sequence stands alone and clean.
(c) **Why those two look different.** The correlation between colour and brightness
    *reverses sign* with distance. Physics does not do that; a brightness-limited
    survey does. This is Malmquist bias, measured.
(d) **The galaxy, edge-on.** The same stars in galactic coordinates, showing the disc
    the Sun sits inside.

Needs a snapshot::

    python scripts/fetch_gaia.py

Usage::

    python scripts/plot_milky_way.py

Writes ``docs/images/phase6-milkyway-light.png`` and ``-dark.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

from orrery.gaia import load_gaia_sample  # noqa: E402

# Single-hue sequential ramps, blue, monotone in lightness. Density is a magnitude,
# so a one-hue ramp is the right encoding; the dark version is stepped for the dark
# surface rather than being an inversion of the light one.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "page": "#f9f9f7",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "ink_muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "ramp": ["#eef4fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"],
        "series": ("#2a78d6", "#eb6834"),
    },
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "ink_muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "ramp": ["#14243c", "#184f95", "#2a78d6", "#6da7ec", "#cde2fb"],
        "series": ("#3987e5", "#d95926"),
    },
}

#: Distance shells for the Malmquist panel, in parsecs.
SHELLS = ((0, 25), (25, 50), (50, 100), (100, 200), (200, 500), (500, 6000))

#: Radius within which the sample is close enough to volume-limited, in parsecs.
NEARBY_PARSEC = 25.0

MARKER_SIZE = 7.0


def style_axes(ax, theme: dict) -> None:
    ax.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.6, zorder=0)
    ax.tick_params(colors=theme["ink_muted"], labelsize=8, length=3, width=0.6)
    for spine in ax.spines.values():
        spine.set_color(theme["axis"])
        spine.set_linewidth(0.8)


def ramp(theme: dict) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("density", theme["ramp"])


def plot_hr_diagram(ax, theme, colour, absolute, title, subtitle, bins=200) -> None:
    """Colour against absolute magnitude, as a density.

    A scatter of 120,000 points is a solid blob; the structure only appears as a
    density. Log-scaled counts, because the main sequence is orders of magnitude
    denser than the branches that make the plot interesting.
    """
    style_axes(ax, theme)
    ax.set_title(title, color=theme["ink"], fontsize=10, pad=10, loc="left")
    ax.set_xlabel("colour  BP − RP   (redder →)", color=theme["ink_muted"], fontsize=8)
    ax.set_ylabel("absolute magnitude  M_G", color=theme["ink_muted"], fontsize=8)

    good = np.isfinite(colour) & np.isfinite(absolute)
    ax.hist2d(
        colour[good],
        absolute[good],
        bins=bins,
        range=[[-0.6, 3.2], [-6.0, 13.0]],
        cmap=ramp(theme),
        norm=LogNorm(vmin=1),
        zorder=2,
    )

    # Magnitudes run backwards: brighter stars go up.
    ax.set_ylim(13.0, -6.0)
    ax.set_xlim(-0.6, 3.2)
    ax.text(
        0.03,
        0.04,
        subtitle,
        transform=ax.transAxes,
        fontsize=7.5,
        color=theme["ink_muted"],
        va="bottom",
    )


def plot_malmquist(ax, theme, stars) -> None:
    """Correlation between colour and brightness, shell by shell."""
    style_axes(ax, theme)
    ax.set_title(
        "(c) The sample changes with distance",
        color=theme["ink"],
        fontsize=10,
        pad=10,
        loc="left",
    )
    ax.set_xlabel("distance shell (parsecs)", color=theme["ink_muted"], fontsize=8)
    ax.set_ylabel("correlation of colour with M_G", color=theme["ink_muted"], fontsize=8)

    absolute = stars.absolute_g()
    distance = stars.distance_parsec()
    usable = np.isfinite(absolute) & np.isfinite(stars.bp_rp)

    centres, correlations, faintest, counts = [], [], [], []
    for low, high in SHELLS:
        shell = usable & (distance >= low) & (distance < high)
        if shell.sum() < 50:
            continue
        centres.append(np.sqrt(max(low, 1) * high))  # geometric centre, for a log axis
        correlations.append(np.corrcoef(stars.bp_rp[shell], absolute[shell])[0, 1])
        faintest.append(absolute[shell].max())
        counts.append(int(shell.sum()))

    ax.set_xscale("log")
    ax.axhline(0.0, color=theme["axis"], linewidth=1.0, zorder=1)
    ax.plot(
        centres,
        correlations,
        color=theme["series"][0],
        linewidth=1.8,
        marker="o",
        markersize=MARKER_SIZE,
        markeredgecolor=theme["surface"],
        markeredgewidth=1.4,
        zorder=3,
    )

    for centre, correlation, count in zip(centres, correlations, counts, strict=True):
        ax.annotate(
            f"{count:,}",
            (centre, correlation),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            fontsize=7,
            color=theme["ink_muted"],
        )

    ax.annotate(
        "main sequence visible",
        (centres[0], correlations[0]),
        textcoords="offset points",
        xytext=(6, -20),
        fontsize=7.5,
        color=theme["ink_secondary"],
    )
    ax.annotate(
        "only giants left",
        (centres[-2], correlations[-2]),
        textcoords="offset points",
        xytext=(-10, 14),
        ha="right",
        fontsize=7.5,
        color=theme["ink_secondary"],
    )

    ax.text(
        0.5,
        -0.20,
        "The trend reverses. No physical relationship does that — a brightness-limited\n"
        "survey does, by dropping faint stars first as distance grows (Malmquist bias).\n"
        "Numbers above each point are the stars in that shell.",
        transform=ax.transAxes,
        fontsize=7.5,
        color=theme["ink_muted"],
        ha="center",
        va="top",
    )


def plot_galaxy_edge_on(ax, theme, stars) -> None:
    """The disc, seen from within it.

    Galactic x points at the centre of the Milky Way, z at the north galactic pole,
    so this projection shows the Sun's own disc edge-on.
    """
    style_axes(ax, theme)
    ax.set_title(
        "(d) The disc, edge-on, from inside it",
        color=theme["ink"],
        fontsize=10,
        pad=10,
        loc="left",
    )
    ax.set_xlabel(
        "galactic x (parsecs) — towards the galactic centre →",
        color=theme["ink_muted"],
        fontsize=8,
    )
    ax.set_ylabel("galactic z (parsecs)", color=theme["ink_muted"], fontsize=8)

    positions = stars.cartesian_galactic()
    usable = np.isfinite(positions[:, 0])

    ax.hist2d(
        positions[usable, 0],
        positions[usable, 2],
        bins=200,
        # Aspect is kept equal below, so the frame is 2:1 to fill the panel rather
        # than leaving the disc stranded in a band of empty space.
        range=[[-1500, 1500], [-750, 750]],
        cmap=ramp(theme),
        norm=LogNorm(vmin=1),
        zorder=2,
    )

    ax.plot(
        0,
        0,
        marker="+",
        markersize=11,
        markeredgewidth=1.6,
        color=theme["ink"],
        zorder=5,
    )
    ax.annotate(
        "the Sun",
        (0, 0),
        textcoords="offset points",
        xytext=(9, 6),
        fontsize=8,
        color=theme["ink"],
        zorder=5,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.5,
        -0.30,
        "Flattened because the disc is flattened — but the sharp edges are the survey's,\n"
        "not the galaxy's: this cut reaches roughly a kiloparsec, and the Milky Way is\n"
        "thirty times wider than the whole frame.",
        transform=ax.transAxes,
        fontsize=7.5,
        color=theme["ink_muted"],
        ha="center",
        va="top",
    )


def build_figure(mode, stars, output: Path) -> None:
    theme = THEMES[mode]

    figure, axes = plt.subplots(2, 2, figsize=(13.6, 12.0), facecolor=theme["page"])
    figure.suptitle(
        "orrery-lab · phase 6 · 120,000 real stars, and what the sample is not",
        color=theme["ink"],
        fontsize=13,
        x=0.045,
        ha="left",
        y=0.975,
    )
    figure.text(
        0.045,
        0.945,
        "Gaia DR3, brighter than G = 8.6, parallax measured to better than 10%.",
        color=theme["ink_muted"],
        fontsize=9,
        ha="left",
    )

    absolute = stars.absolute_g()
    distance = stars.distance_parsec()
    nearby = np.isfinite(distance) & (distance < NEARBY_PARSEC)

    plot_hr_diagram(
        axes[0][0],
        theme,
        stars.bp_rp,
        absolute,
        "(a) Hertzsprung–Russell, whole sample",
        f"{int(np.isfinite(absolute).sum()):,} stars. The dense diagonal is the main\n"
        "sequence; the cloud above it is red giants, which dominate here.",
    )
    plot_hr_diagram(
        axes[0][1],
        theme,
        stars.bp_rp[nearby],
        absolute[nearby],
        f"(b) The same, within {NEARBY_PARSEC:.0f} parsecs",
        f"{int(nearby.sum()):,} stars. Close enough that almost nothing is missed,\n"
        "so the main sequence stands alone — correlation +0.88.",
        bins=60,
    )
    plot_malmquist(axes[1][0], theme, stars)
    plot_galaxy_edge_on(axes[1][1], theme, stars)

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), h_pad=4.4)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, facecolor=theme["page"])
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="docs/images")
    args = parser.parse_args()

    try:
        stars = load_gaia_sample()
    except FileNotFoundError as error:
        raise SystemExit(f"{error}") from error

    print(f"loaded {len(stars):,} stars")

    absolute = stars.absolute_g()
    distance = stars.distance_parsec()
    usable = np.isfinite(absolute) & np.isfinite(stars.bp_rp)

    print("\nColour–brightness correlation by distance shell:")
    for low, high in SHELLS:
        shell = usable & (distance >= low) & (distance < high)
        if shell.sum() < 50:
            continue
        correlation = np.corrcoef(stars.bp_rp[shell], absolute[shell])[0, 1]
        print(
            f"  {low:>4}–{high:<5} pc   n = {int(shell.sum()):>6,}   "
            f"corr = {correlation:+.3f}   faintest M_G = {absolute[shell].max():+.2f}"
        )

    outdir = Path(args.outdir)
    for mode in ("light", "dark"):
        target = outdir / f"phase6-milkyway-{mode}.png"
        build_figure(mode, stars, target)
        print(f"\nwrote {target}")


if __name__ == "__main__":
    main()
