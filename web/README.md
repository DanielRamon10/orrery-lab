# orrery-lab · web

The interactive 3D scene. See the [project README](../README.md) for what this is and
why it exists.

## Commands

```bash
npm install
npm run dev        # dev server at http://localhost:5173/orrery-lab/
npm run test       # 104 tests, including 72 parity cases against Python
npm run lint
npm run build      # type-check then bundle into dist/
npm run preview    # serve the production build locally
```

The dev server URL includes the `/orrery-lab/` prefix because `base` is set for
GitHub Pages, which serves from a repository sub-path. Override it for a fork:

```bash
VITE_BASE=/my-fork/ npm run build
```

## A warning about `src/data/`

`src/data/elements.generated.ts` and `src/data/parity-fixture.generated.json` are
**generated** from the Python package and must not be edited by hand. Regenerate them
from the repository root:

```bash
python scripts/export_web_data.py
```

CI fails if they are stale. The reason for generating rather than copying: the physics
exists in two languages, and the numbers need a single source of truth. The parity
fixture is the other half of that guard — it holds the TypeScript port to what Python
computes, to 1e-12 relative, across 9 bodies and 8 dates spanning 1800–2100.

## Layout

| Path | What lives there |
|---|---|
| `src/lib/kepler.ts` | Port of `orrery/kepler.py` — the equation solver. |
| `src/lib/ephemeris.ts` | Port of `orrery/ephemeris.py` — elements to 3D state vectors. |
| `src/lib/ephemeris.test.ts` | Parity against Python, plus the physical-law checks. |
| `src/lib/scale.ts` | The distance and radius scale modes, and why they exist. |
| `src/scene/` | Sun, planets, orbit lines, camera framing. |
| `src/state/` | The clock — a mutable ref rather than React state, deliberately. |
| `src/ui/` | Time controls, scale toggles, live read-out. |
