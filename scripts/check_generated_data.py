"""Verify the committed generated web data still matches the Python source.

Why this is not just ``git diff``
---------------------------------
The obvious check — regenerate the files and diff the bytes — was the first thing
CI did, and it failed immediately on a clean checkout. Not because anything was
stale, but because the parity fixture stores full-precision floats and the last bit
differs between machines::

    -  0.03987383378003804      (numpy on Windows)
    +  0.039873833780038045     (numpy on the Linux runner)

That is one unit in the last place. IEEE 754 does not promise bit-identical results
for a chain of transcendental functions across different libm and BLAS builds, so a
byte comparison was demanding a guarantee that does not exist.

The check that was actually wanted is *"has anyone changed the physics without
regenerating the data?"* — and that question is numerical. So:

* ``elements.generated.ts`` is still compared **exactly**. It contains decimal
  literals transcribed straight from the dataclass, with no arithmetic in between,
  so it genuinely must be byte-identical everywhere.
* ``parity-fixture.generated.json`` is compared **numerically**, at a tolerance far
  tighter than the one the parity test itself uses. Real staleness — an edited
  element, a changed epoch — moves these values by orders of magnitude more than a
  rounding difference, so nothing is lost by allowing the last bits to wobble.

Usage::

    python scripts/check_generated_data.py

Exits non-zero, with a diagnosis, if regeneration is genuinely needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The repository root has to be importable before the project imports below, so
# that `python scripts/check_generated_data.py` works from a fresh checkout with
# nothing installed but the dependencies.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from orrery import body_state  # noqa: E402
from scripts.export_web_data import DATA_DIR, build_elements_module  # noqa: E402

#: Relative tolerance for the fixture comparison.
#:
#: Two orders of magnitude tighter than the 1e-12 the TypeScript parity test allows,
#: so this still fails on any change with physical meaning, while sitting far above
#: the ~1e-16 wobble of a last-bit difference.
STALENESS_TOLERANCE = 1e-14


def check_elements_module() -> list[str]:
    """Byte-compare the generated element table."""
    path = DATA_DIR / "elements.generated.ts"
    if not path.exists():
        return [f"{path} is missing — run scripts/export_web_data.py"]

    committed = path.read_text(encoding="utf-8")
    regenerated = build_elements_module()

    if committed.replace("\r\n", "\n") != regenerated.replace("\r\n", "\n"):
        return [
            f"{path.name} differs from what orrery/elements.py produces. "
            "It is pure transcription with no arithmetic, so this is a real "
            "mismatch: run scripts/export_web_data.py and commit the result."
        ]
    return []


def check_parity_fixture() -> list[str]:
    """Numerically compare every stored reference state against a fresh one."""
    path = DATA_DIR / "parity-fixture.generated.json"
    if not path.exists():
        return [f"{path} is missing — run scripts/export_web_data.py"]

    fixture = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    worst = 0.0

    for case in fixture["cases"]:
        state = body_state(case["body"], case["jd"])

        for label, stored, fresh in (
            ("position", case["position"], state.position),
            ("velocity", case["velocity"], state.velocity),
        ):
            stored_array = np.asarray(stored, dtype=float)
            fresh_array = np.asarray(fresh, dtype=float)
            scale = max(float(np.linalg.norm(stored_array)), 1e-30)
            relative = float(np.max(np.abs(fresh_array - stored_array))) / scale
            worst = max(worst, relative)

            if relative > STALENESS_TOLERANCE:
                problems.append(
                    f"{case['body']} at {case['date']}: {label} differs by "
                    f"{relative:.3e} relative (tolerance {STALENESS_TOLERANCE:.0e})"
                )

    if problems:
        problems.append(
            "The fixture no longer matches the Python implementation. "
            "Run scripts/export_web_data.py and commit the result."
        )
    else:
        print(
            f"  parity fixture: {len(fixture['cases'])} states, "
            f"worst relative difference {worst:.2e}"
        )
    return problems


def main() -> int:
    print("Checking generated web data against the Python source ...")

    problems = check_elements_module() + check_parity_fixture()

    if problems:
        print("\nSTALE:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print("Generated data is current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
