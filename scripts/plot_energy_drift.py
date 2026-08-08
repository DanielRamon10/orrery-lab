"""Render the phase 3 figure: why the choice of integrator matters.

Four panels, telling one argument in order:

(a) **60 orbits.** RK4 is comfortably the most accurate. Fourth order beats second
    order, exactly as advertised. If the story stopped here, RK4 would be the right
    answer.
(b) **1500 orbits.** The symplectic methods have not got any worse — their error is
    still oscillating inside the same band — while RK4's has climbed straight past
    them. This crossover is the whole argument for symplectic integration, and it is
    measured here rather than asserted.
(c) **Convergence order.** Halve the step, watch the error fall by 2^order. Confirms
    each scheme is what it claims to be; a wrong coefficient would show up as a
    wrong slope.
(d) **The real solar system**, and a diagnostic worth knowing. Some planets' orbital
    elements wander because of genuine mutual perturbation, others because the step
    is too coarse. Varying the step tells you which is which: real physics does not
    care about your step size.

Usage::

    python scripts/plot_energy_drift.py
    python scripts/plot_energy_drift.py --quick     # shorter spans, for iterating

Writes ``docs/images/phase3-integrators-light.png`` and ``-dark.png``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from orrery.constants import GM_BODY, GM_SUN, J2000_JD  # noqa: E402
from orrery.initial_conditions import (  # noqa: E402
    solar_system_state,
    two_body_period_days,
    two_body_state,
)
from orrery.nbody import INTEGRATORS, integrate  # noqa: E402

# ---------------------------------------------------------------------------
# Theme --- the project's reference palette. Categorical slots 1-4 for the four
# integrators; validated on the adjacent pairlist in both modes. The light-mode
# validator warns that two slots fall below 3:1 contrast, so every line carries a
# direct label, which is the documented relief.
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
        "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
    },
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "ink_muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ("#3987e5", "#d95926", "#199e70", "#c98500"),
    },
}

# Fixed slot per integrator, so a panel that omits one does not repaint the others.
SERIES_SLOT = {"leapfrog": 0, "yoshida4": 1, "rk4": 2, "euler": 3}

LABELS = {
    "leapfrog": "Leapfrog (symplectic, 2nd)",
    "yoshida4": "Yoshida 4 (symplectic, 4th)",
    "rk4": "RK4 (not symplectic, 4th)",
    "euler": "Euler (not symplectic, 1st)",
}

# The test orbit: eccentric enough to be a real test, not so much as to need a
# tiny step.
SEMI_MAJOR_AXIS_AU = 1.0
ECCENTRICITY = 0.3
COMPANION_GM = GM_BODY["jupiter"]

LINE_WIDTH = 1.6
MARKER_SIZE = 7.0

#: Direct-label placement per integrator: (x as a fraction of the span, above/below).
#:
#: Staggered in x so labels do not pile up against the right edge, and alternating
#: above/below because RK4's rising curve necessarily crosses the flat symplectic
#: bands — put both labels on the same side and they collide exactly where the
#: comparison is most interesting.
LABEL_PLACEMENT: dict[str, tuple[float, int]] = {
    "euler": (0.25, +1),
    "yoshida4": (0.30, -1),
    # Above its own curve, past the point where it has overtaken leapfrog. Below
    # would put the text inside leapfrog's oscillation band, which is 1.5 decades
    # tall and swamps it.
    "rk4": (0.55, +1),
    "leapfrog": (0.80, +1),
}

#: Step sizes for the convergence panel.
#:
#: Euler needs its own, far smaller, range. At the steps that suit the other three
#: its energy error has already saturated near 100% — an orbit cannot be more than
#: completely wrong — and a saturated curve is flat, so fitting a slope there
#: measures the ceiling rather than the order. Convergence order is only defined in
#: the asymptotic regime, and Euler reaches it much later than the rest.
CONVERGENCE_STEPS = {
    "leapfrog": [8.0, 4.0, 2.0, 1.0, 0.5],
    "yoshida4": [8.0, 4.0, 2.0, 1.0, 0.5],
    "rk4": [8.0, 4.0, 2.0, 1.0, 0.5],
    # Euler needs to get down to a few hundredths of a day before its error is small
    # enough (a percent or so) to be in the linear regime rather than approaching
    # the 100% ceiling.
    "euler": [0.2, 0.1, 0.05, 0.025],
}


def style_axes(ax, theme: dict) -> None:
    ax.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.6, zorder=0)
    ax.tick_params(colors=theme["ink_muted"], labelsize=8, length=3, width=0.6)
    for spine in ax.spines.values():
        spine.set_color(theme["axis"])
        spine.set_linewidth(0.8)


def colour_for(theme: dict, integrator: str) -> str:
    return theme["series"][SERIES_SLOT[integrator]]


# ---------------------------------------------------------------------------
# Computation --- done once, then drawn into both themes.
# ---------------------------------------------------------------------------


def run_drift(integrator: str, orbits: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Energy drift of a two-body orbit over many revolutions."""
    state = two_body_state(SEMI_MAJOR_AXIS_AU, ECCENTRICITY, GM_SUN, COMPANION_GM)
    period = two_body_period_days(SEMI_MAJOR_AXIS_AU, GM_SUN, COMPANION_GM)

    trajectory = integrate(
        *state.as_arrays(),
        duration_days=period * orbits,
        dt=dt,
        integrator=integrator,
        sample_every=max(1, int(orbits * period / dt / 700)),
    )
    return trajectory.times / period, trajectory.relative_energy_error


CONVERGENCE_ORBITS = 2.0


def run_convergence(integrator: str) -> tuple[np.ndarray, np.ndarray]:
    """Peak energy drift over a couple of orbits, as a function of step size."""
    period = two_body_period_days(SEMI_MAJOR_AXIS_AU, GM_SUN, COMPANION_GM)
    steps = CONVERGENCE_STEPS[integrator]
    drifts = []

    for dt in steps:
        state = two_body_state(SEMI_MAJOR_AXIS_AU, ECCENTRICITY, GM_SUN, COMPANION_GM)
        trajectory = integrate(
            *state.as_arrays(),
            duration_days=period * CONVERGENCE_ORBITS,
            dt=dt,
            integrator=integrator,
            sample_every=5,
        )
        drifts.append(trajectory.relative_energy_error.max())

    return np.array(steps), np.array(drifts)


def osculating_axis_spread(trajectory, names: tuple[str, ...], body: str) -> float:
    """Peak-to-peak variation of a body's osculating semi-major axis, as a fraction.

    Uses the semi-major axis rather than the raw distance because the axis is
    independent of where the planet happens to be on its orbit.
    """
    sun = names.index("sun")
    index = names.index(body)

    relative_position = trajectory.positions[:, index, :] - trajectory.positions[:, sun, :]
    relative_velocity = trajectory.velocities[:, index, :] - trajectory.velocities[:, sun, :]

    distance = np.linalg.norm(relative_position, axis=-1)
    speed_squared = np.einsum("ij,ij->i", relative_velocity, relative_velocity)
    axis = 1.0 / (2.0 / distance - speed_squared / (GM_SUN + GM_BODY[body]))

    return float((axis.max() - axis.min()) / axis.mean())


def run_solar_system_diagnostic(
    bodies: tuple[str, ...], years: float, steps: tuple[float, float]
) -> dict[str, tuple[float, float]]:
    """Element wander at two step sizes, per body."""
    spreads: dict[str, list[float]] = {body: [] for body in bodies}

    for dt in steps:
        state = solar_system_state(J2000_JD)
        trajectory = integrate(
            *state.as_arrays(),
            duration_days=365.25 * years,
            dt=dt,
            integrator="leapfrog",
            names=state.names,
            sample_every=max(1, int(200 * 2.0 / dt)),
        )
        for body in bodies:
            spreads[body].append(osculating_axis_spread(trajectory, state.names, body))

    return {body: (values[0], values[1]) for body, values in spreads.items()}


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def plot_drift_panel(ax, theme, results, title, subtitle) -> None:
    style_axes(ax, theme)
    ax.set_title(title, color=theme["ink"], fontsize=10, pad=10, loc="left")
    ax.set_xlabel("orbits completed", color=theme["ink_muted"], fontsize=8)
    ax.set_ylabel("relative energy error", color=theme["ink_muted"], fontsize=8)
    ax.set_yscale("log")

    for integrator, (orbits, drift) in results.items():
        colour = colour_for(theme, integrator)
        # The first sample is exactly zero drift, which a log axis cannot show.
        visible = drift > 0
        visible_orbits = orbits[visible]
        visible_drift = drift[visible]

        ax.plot(visible_orbits, visible_drift, color=colour, linewidth=LINE_WIDTH, zorder=2)

        # Direct label — the relief the light-mode contrast warning requires, and it
        # beats hunting through a legend box.
        #
        # Anchored to the local *upper envelope* rather than to a single sample: the
        # symplectic traces oscillate across three decades every orbit, so a label
        # pinned to one point lands in a trough about half the time and reads as
        # belonging to whatever line passes through there.
        fraction, direction = LABEL_PLACEMENT[integrator]
        anchor = int(fraction * (len(visible_drift) - 1))
        window = slice(max(0, anchor - 25), min(len(visible_drift), anchor + 25))
        envelope = (
            visible_drift[window].max() if direction > 0 else visible_drift[window].min()
        )
        ax.annotate(
            LABELS[integrator],
            (visible_orbits[anchor], envelope),
            textcoords="offset points",
            xytext=(0, 8 if direction > 0 else -14),
            ha="center",
            fontsize=7.5,
            color=theme["ink_secondary"],
            zorder=5,
        )

    ax.text(
        0.03,
        0.05,
        subtitle,
        transform=ax.transAxes,
        fontsize=7.5,
        color=theme["ink_muted"],
        va="bottom",
    )


def plot_convergence_panel(ax, theme, results) -> None:
    style_axes(ax, theme)
    ax.set_title(
        "(c) Convergence: halve the step, divide the error by 2^order",
        color=theme["ink"],
        fontsize=10,
        pad=10,
        loc="left",
    )
    ax.set_xlabel("step size (days)", color=theme["ink_muted"], fontsize=8)
    ax.set_ylabel(
        f"peak energy error over {CONVERGENCE_ORBITS:.0f} orbits",
        color=theme["ink_muted"],
        fontsize=8,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")

    for integrator, (steps, drifts) in results.items():
        colour = colour_for(theme, integrator)
        ax.plot(
            steps,
            drifts,
            color=colour,
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE - 1.5,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.2,
            zorder=2,
        )

        # Fitted slope on the log-log axes *is* the order of accuracy.
        slope, _ = np.polyfit(np.log10(steps), np.log10(drifts), deg=1)
        expected = INTEGRATORS[integrator][2]
        ax.annotate(
            f"{LABELS[integrator].split(' (')[0]} — slope {slope:.2f} (order {expected})",
            (steps[-1], drifts[-1]),
            textcoords="offset points",
            xytext=(8, -2),
            fontsize=7.5,
            color=theme["ink_secondary"],
        )

    # Upper right: Euler stops at 0.2 days, so the large-step, large-error corner is
    # the only region no line passes through.
    ax.text(
        0.975,
        0.96,
        "Euler measured over its own, much finer steps: at the others' step\n"
        "sizes its error has saturated near 100%, where no order can be fitted.",
        transform=ax.transAxes,
        fontsize=7.0,
        color=theme["ink_muted"],
        ha="right",
        va="top",
    )


def plot_solar_diagnostic_panel(ax, theme, spreads, steps, years) -> None:
    """Dumbbell chart: does the wander move when the step moves?

    Two marks per body joined by a line. A long connector means the quantity is an
    artefact of the step size; marks on top of each other mean it is real physics.
    The form is chosen for exactly that reading — the length of the connector *is*
    the finding.
    """
    style_axes(ax, theme)
    ax.grid(False, axis="y")
    ax.set_title(
        "(d) Real solar system: which wander is physical?",
        color=theme["ink"],
        fontsize=10,
        pad=10,
        loc="left",
    )
    ax.set_xlabel(
        f"osculating semi-major axis, peak-to-peak over {years:.0f} years (%)",
        color=theme["ink_muted"],
        fontsize=8,
    )
    ax.set_xscale("log")

    bodies = list(spreads)
    positions = np.arange(len(bodies))[::-1]

    coarse_colour = theme["series"][0]
    fine_colour = theme["series"][1]

    for body, y in zip(bodies, positions, strict=True):
        coarse, fine = (value * 100 for value in spreads[body])

        ax.plot(
            [coarse, fine],
            [y, y],
            color=theme["ink_muted"],
            linewidth=1.2,
            zorder=2,
        )
        for value, colour in ((coarse, coarse_colour), (fine, fine_colour)):
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

        ratio = spreads[body][0] / spreads[body][1]
        verdict = "numerical" if ratio > 2.0 else "physical"
        ax.annotate(
            verdict,
            (max(coarse, fine), y),
            textcoords="offset points",
            xytext=(12, -3),
            fontsize=7.5,
            color=theme["ink_secondary"],
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [body.capitalize() for body in bodies], fontsize=8, color=theme["ink_secondary"]
    )
    # Headroom on the right so the verdict label beside the widest mark is not clipped.
    left, right = ax.get_xlim()
    ax.set_xlim(left, right * 2.2)

    # Two series, so a legend is required; placed where the data is not.
    for index, (label, colour) in enumerate(
        ((f"step {steps[0]:g} d", coarse_colour), (f"step {steps[1]:g} d", fine_colour))
    ):
        ax.plot(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=MARKER_SIZE,
            color=colour,
            markeredgecolor=theme["surface"],
            markeredgewidth=1.4,
            label=label,
        )
        del index
    # Lower left: the outer planets all sit above 0.5% so that corner is empty.
    legend = ax.legend(
        loc="lower left",
        frameon=False,
        fontsize=7.5,
        labelcolor=theme["ink_secondary"],
        handletextpad=0.4,
    )
    legend.set_zorder(6)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_figure(mode, short, long, convergence, spreads, config, output: Path) -> None:
    theme = THEMES[mode]

    figure, axes = plt.subplots(2, 2, figsize=(13.4, 10.4), facecolor=theme["page"])
    figure.suptitle(
        "orrery-lab · phase 3 · symplectic integration, measured",
        color=theme["ink"],
        fontsize=13,
        x=0.045,
        ha="left",
        y=0.975,
    )
    figure.text(
        0.045,
        0.943,
        f"Two-body orbit, a = 1 AU, e = 0.3, step {config['dt']:g} days throughout. Energy is "
        "conserved exactly in theory, so every wiggle below is the integrator's own error.",
        color=theme["ink_muted"],
        fontsize=9,
        ha="left",
    )

    plot_drift_panel(
        axes[0][0],
        theme,
        short,
        f"(a) {config['short_orbits']:.0f} orbits — fourth order wins",
        "RK4 beats leapfrog here. Order of accuracy is doing exactly what it promises.",
    )
    plot_drift_panel(
        axes[0][1],
        theme,
        long,
        f"(b) {config['long_orbits']:.0f} orbits, same step — the symplectic crossover",
        "The symplectic pair are unchanged: bounded error, forever. RK4 has climbed past both.",
    )
    plot_convergence_panel(axes[1][0], theme, convergence)
    plot_solar_diagnostic_panel(
        axes[1][1], theme, spreads, config["solar_steps"], config["solar_years"]
    )

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, facecolor=theme["page"])
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="shorter spans, for iterating")
    parser.add_argument("--outdir", default="docs/images")
    args = parser.parse_args()

    # Panels (a) and (b) deliberately share a step size. They differ only in how long
    # the integration runs, which is the entire point of the comparison: change two
    # things at once and the crossover could be attributed to either.
    config = {
        "dt": 5.0,
        "short_orbits": 60.0,
        "long_orbits": 300.0 if args.quick else 1500.0,
        "solar_years": 20.0 if args.quick else 100.0,
        "solar_steps": (2.0, 0.5),
    }

    started = time.perf_counter()

    print(f"(a) {config['short_orbits']:.0f} orbits at dt = {config['dt']} d ...")
    short = {
        name: run_drift(name, config["short_orbits"], config["dt"])
        for name in ("leapfrog", "yoshida4", "rk4", "euler")
    }

    print(f"(b) {config['long_orbits']:.0f} orbits at dt = {config['dt']} d ...")
    # Euler is omitted: at this span its error exceeds 100%, which would flatten the
    # log axis and hide the comparison that the panel exists to make.
    long = {
        name: run_drift(name, config["long_orbits"], config["dt"])
        for name in ("leapfrog", "yoshida4", "rk4")
    }

    print("(c) convergence sweep ...")
    convergence = {
        name: run_convergence(name) for name in ("leapfrog", "yoshida4", "rk4", "euler")
    }

    print(f"(d) solar system, {config['solar_years']:.0f} years at two step sizes ...")
    spreads = run_solar_system_diagnostic(
        ("mercury", "mars", "saturn", "uranus", "neptune"),
        config["solar_years"],
        config["solar_steps"],
    )

    outdir = Path(args.outdir)
    for mode in ("light", "dark"):
        target = outdir / f"phase3-integrators-{mode}.png"
        build_figure(mode, short, long, convergence, spreads, config, target)
        print(f"wrote {target}")

    print(f"\ntotal {time.perf_counter() - started:.1f} s")

    print("\nPeak relative energy error:")
    for name, (_, drift) in short.items():
        print(f"  {LABELS[name]:<34} {config['short_orbits']:>6.0f} orbits: {drift.max():.3e}")
    for name, (_, drift) in long.items():
        print(f"  {LABELS[name]:<34} {config['long_orbits']:>6.0f} orbits: {drift.max():.3e}")

    print("\nOsculating semi-major axis wander (step 2 d vs 0.5 d):")
    for body, (coarse, fine) in spreads.items():
        verdict = "numerical" if coarse / fine > 2.0 else "physical"
        print(f"  {body:<9} {coarse * 100:8.4f}%  ->  {fine * 100:8.4f}%   {verdict}")


if __name__ == "__main__":
    main()
