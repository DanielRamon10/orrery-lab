"""What happens to the phase 5 stability model when the orbits stop being coplanar.

Phase 5 trained a classifier on two-planet systems that all shared a plane, and it beat
both Gladman's criterion and a tuned threshold on held-out data. Phase 3's inclination
sweep then showed that coplanarity is not a harmless simplification: a few degrees of
tilt changes which systems survive. So the model was trained on one world and this asks
what it does in a neighbouring one.

The setup is a clean distribution shift. Every feature the model sees is blind to
inclination --- masses, separations, eccentricities --- so tilting the orbits leaves the
design matrix **byte-identical** and moves only the labels. Anything the model loses, it
loses because the world changed, not because its inputs did.

Two lenses are needed to read the result, and using only one of them gives the wrong
answer:

* **Accuracy alone lies.** Inclination pushes the surviving fraction toward 1, so a rule
  that says "stable" scores better for free. Every accuracy here is therefore drawn
  against the accuracy of predicting the commoner class every time.
* **AUC is base-rate blind** and measures whether the model still *ranks* systems by
  risk, which turns out to be the half that survives.

Usage::

    python scripts/plot_inclination_transfer.py [--quick] [--outdir docs/images]

Writes ``docs/images/phase5-transfer-{light,dark}.png``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from orrery.models import evaluate_inclination_transfer  # noqa: E402
from orrery.stability import build_stability_dataset  # noqa: E402

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

#: Tilts to evaluate at. The first entry must be 0: without a coplanar control drawn
#: from the same seed there is no way to separate an inclination effect from an
#: ordinary sampling difference between the training and evaluation draws.
TILTS = (0.0, 5.0, 10.0, 20.0, 40.0)

#: Deliberately not the training seed. The features carry no inclination, so evaluating
#: a tilted set drawn with the training seed would be scoring the model on its own
#: training rows and every number would be memorised.
EVALUATION_SEED = 771202

CAPTIONS = (
    "Every rule here is blind to inclination, so every rule degrades. What matters is\n"
    "the dashed line: past a few degrees all three sit below the accuracy of simply\n"
    "assuming nothing ever breaks.",
    "The model never stopped ranking systems by risk: AUC barely moves. It stopped\n"
    "being useful because its 0.5 cut was calibrated for a world where a third of\n"
    "systems break, applied to one where a tenth do.",
)


def style_axes(ax, theme: dict) -> None:
    ax.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.6, zorder=0)
    ax.tick_params(colors=theme["ink_muted"], labelsize=8, length=3, width=0.6)
    for spine in ax.spines.values():
        spine.set_color(theme["axis"])
        spine.set_linewidth(0.8)


def plot_accuracy(ax, theme, scores) -> None:
    """Each rule's accuracy, against the accuracy of not having a rule at all."""
    style_axes(ax, theme)
    ax.set_title(
        "(a) All three fall below \"assume it survives\"",
        color=theme["ink"], fontsize=10, pad=10, loc="left",
    )
    ax.set_xlabel("median mutual inclination between orbits (degrees)",
                  color=theme["ink_muted"], fontsize=8)
    ax.set_ylabel("accuracy on the shifted set", color=theme["ink_muted"], fontsize=8)

    angles = [score.median_mutual_inclination_deg for score in scores]

    ax.plot(
        angles, [score.majority_baseline for score in scores],
        color=theme["ink_secondary"], linewidth=1.4, linestyle="--", zorder=2,
        label="always predict \"stable\"",
    )

    for attribute, colour, label in (
        ("gladman_accuracy", theme["series"][2], "Gladman criterion"),
        ("tuned_accuracy", theme["series"][0], "tuned threshold"),
        ("model_accuracy", theme["series"][1], "gradient boosting"),
    ):
        ax.plot(
            angles, [getattr(score, attribute) for score in scores],
            color=colour, linewidth=LINE_WIDTH, marker="o",
            markersize=MARKER_SIZE - 2, markeredgecolor=theme["surface"],
            markeredgewidth=1.2, zorder=3, label=label,
        )

    # The four series between them cover the middle of the panel, so room for the
    # legend is made below the data rather than taken from it. Placing the legend
    # outside the axes instead would work visually but re-runs the tight_layout
    # squeeze that the captions had to be moved to figure coordinates to escape.
    values = [score.majority_baseline for score in scores] + [
        getattr(score, attribute)
        for score in scores
        for attribute in ("gladman_accuracy", "tuned_accuracy", "model_accuracy")
    ]
    low, high = min(values), max(values)
    span = high - low
    ax.set_ylim(low - 0.55 * span, high + 0.08 * span)

    ax.legend(
        frameon=False, fontsize=8, loc="lower left", ncol=2, columnspacing=1.4,
        handlelength=1.8, labelcolor=theme["ink_secondary"],
    )


def plot_two_lenses(ax, theme, scores) -> None:
    """Ranking quality against decision quality, on one pair of axes."""
    style_axes(ax, theme)
    ax.set_title(
        "(b) The ranking survives; the cut-off does not",
        color=theme["ink"], fontsize=10, pad=10, loc="left",
    )
    ax.set_xlabel("median mutual inclination between orbits (degrees)",
                  color=theme["ink_muted"], fontsize=8)
    ax.set_ylabel("ROC AUC  /  accuracy above the baseline",
                  color=theme["ink_muted"], fontsize=8)

    angles = [score.median_mutual_inclination_deg for score in scores]

    ax.plot(
        angles, [score.model_roc_auc for score in scores],
        color=theme["series"][0], linewidth=LINE_WIDTH, marker="o",
        markersize=MARKER_SIZE - 2, markeredgecolor=theme["surface"],
        markeredgewidth=1.2, zorder=3, label="AUC (base-rate blind)",
    )

    for attribute, colour, style, label in (
        ("model_accuracy", theme["series"][1], "-", "lift over the baseline, cut at 0.5"),
        ("recalibrated_accuracy", theme["series"][2], "--", "lift after moving the cut"),
    ):
        ax.plot(
            angles,
            [getattr(s, attribute) - s.majority_baseline for s in scores],
            color=colour, linewidth=LINE_WIDTH, linestyle=style, marker="o",
            markersize=MARKER_SIZE - 2, markeredgecolor=theme["surface"],
            markeredgewidth=1.2, zorder=3, label=label,
        )

    ax.axhline(0.0, color=theme["axis"], linewidth=1.0, zorder=1)
    ax.legend(frameon=False, fontsize=8, loc="center right", labelcolor=theme["ink_secondary"])


def build_figure(mode, scores, output: Path) -> None:
    theme = THEMES[mode]
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.0), facecolor=theme["page"])
    figure.suptitle(
        "orrery-lab · a coplanar-trained model meeting inclined systems",
        color=theme["ink"], fontsize=13, x=0.045, ha="left", y=0.955,
    )
    figure.text(
        0.045, 0.878,
        "The features cannot see inclination, so the design matrix is identical "
        "throughout and only the truth moves.",
        color=theme["ink_muted"], fontsize=9, ha="left",
    )

    plot_accuracy(axes[0], theme, scores)
    plot_two_lenses(axes[1], theme, scores)

    figure.tight_layout(rect=(0.0, 0.20, 1.0, 0.83))

    # Figure coordinates, not axes coordinates: tight_layout measures an axes' text
    # children, so a caption hung below the axes would squeeze the plot to fit itself.
    for ax, caption in zip(axes, CAPTIONS, strict=True):
        box = ax.get_position()
        figure.text(
            (box.x0 + box.x1) / 2, box.y0 - 0.115, caption,
            fontsize=7.5, color=theme["ink_muted"], ha="center", va="top",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, facecolor=theme["page"])
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller sample, for iterating")
    parser.add_argument("--outdir", default="docs/images")
    args = parser.parse_args()

    train_count = 800 if args.quick else 3000
    eval_count = 400 if args.quick else 1500
    orbits = 300.0 if args.quick else 1500.0

    started = time.perf_counter()
    print(f"training on {train_count:,} coplanar systems, {orbits:.0f} orbits each ...")
    training = build_stability_dataset(count=train_count, orbits=orbits)
    print(f"  {training.stable_fraction:.1%} stable")

    shifted = []
    for tilt in TILTS:
        dataset = build_stability_dataset(
            count=eval_count, orbits=orbits, seed=EVALUATION_SEED,
            max_inclination_deg=tilt,
        )
        shifted.append(dataset)
        print(f"  eval set tilted {tilt:>4.0f} deg: {dataset.stable_fraction:.1%} stable")

    scores = evaluate_inclination_transfer(training, shifted)

    print()
    header = ("mutual", "stable", "always", "Gladman", "tuned", "model", "recal", "AUC")
    print("".join(f"{name:>9}" for name in header) + f"{'false alarm':>13}")
    for score in scores:
        print(
            f"{score.median_mutual_inclination_deg:>8.1f}d"
            f"{score.stable_fraction:>9.1%}"
            f"{score.majority_baseline:>9.3f}"
            f"{score.gladman_accuracy:>9.3f}"
            f"{score.tuned_accuracy:>9.3f}"
            f"{score.model_accuracy:>9.3f}"
            f"{score.recalibrated_accuracy:>9.3f}"
            f"{score.model_roc_auc:>9.3f}"
            f"{score.false_alarm_rate:>13.1%}"
        )

    control, worst = scores[0], scores[-1]
    print(
        f"\nlift over always-stable: {control.model_accuracy - control.majority_baseline:+.3f}"
        f" coplanar -> {worst.model_accuracy - worst.majority_baseline:+.3f} at"
        f" {worst.median_mutual_inclination_deg:.0f} deg"
    )
    print(
        f"AUC over the same range: {control.model_roc_auc:.3f} -> {worst.model_roc_auc:.3f}"
        "  (the ranking is intact; the threshold is not)"
    )
    print(
        f"moving the cut alone recovers {worst.model_accuracy:.3f} ->"
        f" {worst.recalibrated_accuracy:.3f}, but its remaining lift is only"
        f" {worst.recalibrated_accuracy - worst.majority_baseline:+.3f}"
    )

    outdir = Path(args.outdir)
    for mode in ("light", "dark"):
        target = outdir / f"phase5-transfer-{mode}.png"
        build_figure(mode, scores, target)
        print(f"wrote {target}")

    print(f"\ntotal {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
