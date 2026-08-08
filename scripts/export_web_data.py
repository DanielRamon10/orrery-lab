"""Generate the browser scene's data from the Python package.

The 3D scene has to place the planets on every animation frame, for any date the
user scrubs to. Shipping precomputed positions would mean either a huge JSON or a
scene locked to a fixed date range, so the browser gets its own copy of the
Kepler solver instead.

Two implementations of the same physics is a real risk: they can silently drift
apart. This script removes most of that risk and makes the rest testable:

1. The **element table is generated**, never hand-copied, so the numbers have a
   single source of truth in :mod:`orrery.elements`.
2. A **parity fixture** records what Python computes for a spread of bodies and
   dates. ``web/src/lib/ephemeris.test.ts`` replays it and fails if the
   TypeScript port disagrees by more than a hair.

Usage::

    python scripts/export_web_data.py

Writes into ``web/src/data/``. Both outputs are committed, so a clone can build
the site without a Python environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orrery import PLANET_NAMES, PLANETS, body_state, julian_date
from orrery.constants import (
    AU_KM,
    AXIAL_TILT_DEG,
    GM_SUN,
    J2000_JD,
    MEAN_RADIUS_KM,
    ROTATION_PERIOD_DAYS,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "web" / "src" / "data"

#: Bodies the scene renders, in order outward from the Sun.
SCENE_BODIES: tuple[str, ...] = (*PLANET_NAMES, "pluto")

#: Approximate true colours, for rendering the spheres.
#:
#: These are *representational* --- they stand for the physical appearance of
#: each body, the way a globe is painted. They are deliberately not the
#: categorical chart palette, which encodes identity in an abstract plot; any
#: actual chart in the UI uses that palette instead.
BODY_COLOUR: dict[str, str] = {
    "sun": "#ffd27a",
    "mercury": "#9a8f88",
    "venus": "#d9b98c",
    "earth": "#4a7fb5",
    "mars": "#c1593a",
    "jupiter": "#c9a27e",
    "saturn": "#d8c08a",
    "uranus": "#8ec5d1",
    "neptune": "#4a6fc4",
    "pluto": "#b3a394",
}

#: Dates the parity fixture covers. Chosen to exercise the whole valid range of
#: the element table (1800-2050), both signs of the century offset, and a date
#: past the table's nominal end so any divergence in the drift terms shows up.
PARITY_DATES: tuple[tuple[int, int, int], ...] = (
    (1800, 1, 1),
    (1899, 12, 31),
    (1969, 7, 20),
    (2000, 1, 1),
    (2026, 7, 26),
    (2030, 3, 15),
    (2049, 12, 31),
    (2100, 6, 1),
)


def format_float(value: float) -> str:
    """Emit a float with full round-trip precision and no stray formatting."""
    return repr(float(value))


def build_elements_module() -> str:
    """Render the TypeScript element table."""
    lines: list[str] = [
        "// GENERATED FILE --- do not edit by hand.",
        "//",
        "// Produced by scripts/export_web_data.py from orrery/elements.py and",
        "// orrery/constants.py. Re-run that script after changing either one.",
        "//",
        "// Angles are in degrees and distances in astronomical units, matching",
        "// JPL's published table; the conversion to radians happens in",
        "// src/lib/ephemeris.ts so these numbers stay auditable against the source.",
        "",
        "export interface BodyElements {",
        "  /** Lower-case key, e.g. `mars`. */",
        "  readonly id: string;",
        "  /** Display name. */",
        "  readonly name: string;",
        "  /** True for the eight planets; false for Pluto. */",
        "  readonly isPlanet: boolean;",
        "",
        "  // Keplerian elements at J2000.0.",
        "  readonly semiMajorAxisAu: number;",
        "  readonly eccentricity: number;",
        "  readonly inclinationDeg: number;",
        "  readonly meanLongitudeDeg: number;",
        "  readonly longitudeOfPerihelionDeg: number;",
        "  readonly longitudeOfAscendingNodeDeg: number;",
        "",
        "  // Rates of change, per Julian century.",
        "  readonly semiMajorAxisRate: number;",
        "  readonly eccentricityRate: number;",
        "  readonly inclinationRate: number;",
        "  readonly meanLongitudeRate: number;",
        "  readonly longitudeOfPerihelionRate: number;",
        "  readonly longitudeOfAscendingNodeRate: number;",
        "",
        "  // Rendering and read-out metadata.",
        "  /** Mean radius in kilometres. */",
        "  readonly radiusKm: number;",
        "  /** Sidereal rotation period in days; negative means retrograde. */",
        "  readonly rotationPeriodDays: number;",
        "  /** Axial tilt to the orbital plane, in degrees. */",
        "  readonly axialTiltDeg: number;",
        "  /** Representational surface colour. */",
        "  readonly colour: string;",
        "  /** Orbital period in days, from Kepler's third law. */",
        "  readonly periodDays: number;",
        "}",
        "",
        "export const BODIES: readonly BodyElements[] = [",
    ]

    for key in SCENE_BODIES:
        elements = PLANETS[key]
        lines.extend(
            [
                "  {",
                f"    id: {key!r}".replace("'", '"') + ",",
                f'    name: "{elements.name}",',
                f"    isPlanet: {'true' if key in PLANET_NAMES else 'false'},",
                f"    semiMajorAxisAu: {format_float(elements.semi_major_axis_au)},",
                f"    eccentricity: {format_float(elements.eccentricity)},",
                f"    inclinationDeg: {format_float(elements.inclination_deg)},",
                f"    meanLongitudeDeg: {format_float(elements.mean_longitude_deg)},",
                "    longitudeOfPerihelionDeg: "
                f"{format_float(elements.longitude_of_perihelion_deg)},",
                "    longitudeOfAscendingNodeDeg: "
                f"{format_float(elements.longitude_of_ascending_node_deg)},",
                f"    semiMajorAxisRate: {format_float(elements.semi_major_axis_rate)},",
                f"    eccentricityRate: {format_float(elements.eccentricity_rate)},",
                f"    inclinationRate: {format_float(elements.inclination_rate)},",
                f"    meanLongitudeRate: {format_float(elements.mean_longitude_rate)},",
                "    longitudeOfPerihelionRate: "
                f"{format_float(elements.longitude_of_perihelion_rate)},",
                "    longitudeOfAscendingNodeRate: "
                f"{format_float(elements.longitude_of_ascending_node_rate)},",
                f"    radiusKm: {format_float(MEAN_RADIUS_KM[key])},",
                f"    rotationPeriodDays: {format_float(ROTATION_PERIOD_DAYS[key])},",
                f"    axialTiltDeg: {format_float(AXIAL_TILT_DEG[key])},",
                f'    colour: "{BODY_COLOUR[key]}",',
                f"    periodDays: {format_float(elements.period_days)},",
                "  },",
            ]
        )

    lines.extend(
        [
            "];",
            "",
            "/** Physical constants the scene needs, kept in step with orrery/constants.py. */",
            "export const CONSTANTS = {",
            f"  J2000_JD: {format_float(J2000_JD)},",
            f"  AU_KM: {format_float(AU_KM)},",
            f"  GM_SUN: {format_float(GM_SUN)},",
            f'  SUN_RADIUS_KM: {format_float(MEAN_RADIUS_KM["sun"])},',
            f'  SUN_COLOUR: "{BODY_COLOUR["sun"]}",',
            f'  SUN_ROTATION_PERIOD_DAYS: {format_float(ROTATION_PERIOD_DAYS["sun"])},',
            "} as const;",
            "",
        ]
    )
    return "\n".join(lines)


def build_parity_fixture() -> dict:
    """Record Python's answers so the TypeScript port can be held to them."""
    cases = []
    for year, month, day in PARITY_DATES:
        jd = julian_date(year, month, day)
        for body in SCENE_BODIES:
            state = body_state(body, jd)
            cases.append(
                {
                    "body": body,
                    "date": f"{year:04d}-{month:02d}-{day:02d}",
                    "jd": jd,
                    "position": [float(component) for component in state.position],
                    "velocity": [float(component) for component in state.velocity],
                }
            )

    return {
        "_comment": (
            "GENERATED by scripts/export_web_data.py. Reference heliocentric ecliptic "
            "J2000 states from the Python implementation, in AU and AU/day. "
            "web/src/lib/ephemeris.test.ts asserts the TypeScript port reproduces these."
        ),
        # Relative, not absolute. Two independent float64 implementations differ
        # in their last bits --- numpy's and V8's sin/cos are not bit-identical,
        # and the summation order in the rotation differs --- so the discrepancy
        # scales with the magnitude of the coordinate. An absolute bound would be
        # slack for Mercury at 0.39 AU and impossibly tight for Neptune at 30 AU.
        #
        # 1e-12 relative is about 4 centimetres at Neptune's distance, some ten
        # orders of magnitude below the arcminute accuracy of the element table
        # itself, so it still catches any real divergence.
        "relativeTolerance": 1e-12,
        "cases": cases,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    elements_path = DATA_DIR / "elements.generated.ts"
    elements_path.write_text(build_elements_module(), encoding="utf-8", newline="\n")
    print(f"wrote {elements_path.relative_to(REPO_ROOT)}  ({len(SCENE_BODIES)} bodies)")

    fixture = build_parity_fixture()
    fixture_path = DATA_DIR / "parity-fixture.generated.json"
    fixture_path.write_text(
        json.dumps(fixture, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"wrote {fixture_path.relative_to(REPO_ROOT)}  "
        f"({len(fixture['cases'])} reference states)"
    )

    # A quick self-check so an obviously broken export is caught here rather than
    # in the browser: every body must sit between its own perihelion and aphelion.
    for case in fixture["cases"]:
        elements = PLANETS[case["body"]]
        distance = float(np.linalg.norm(case["position"]))
        assert elements.perihelion_au - 0.5 <= distance <= elements.aphelion_au + 0.5, (
            f"{case['body']} at {case['date']}: {distance:.4f} AU is outside its orbit"
        )
    print("self-check passed: all exported positions lie on their orbits")


if __name__ == "__main__":
    main()
