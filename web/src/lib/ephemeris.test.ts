/**
 * Holds the TypeScript ephemeris to the Python one.
 *
 * The project deliberately implements the same physics twice --- once in Python
 * for the analysis and once here so the 3D scene can compute any date on the fly.
 * Two implementations can silently drift apart, and that would be the worst kind
 * of bug: the scene would look perfectly plausible while being wrong.
 *
 * So the Python package is treated as the reference. `scripts/export_web_data.py`
 * records what it computes for a spread of bodies and dates, and this suite
 * replays every one of those cases through the port.
 *
 * The remaining tests re-check the physical laws directly, so a failure can be
 * localised: if parity breaks but the laws still hold, the two ports disagree on
 * the element table or the epoch; if the laws break too, the solver is wrong.
 */

import { describe, expect, it } from "vitest";

import fixture from "../data/parity-fixture.generated.json";
import { BODIES, CONSTANTS } from "../data/elements.generated";
import {
  bodyState,
  dateFromJulianDate,
  eclipticToWorld,
  julianDateFromDate,
  orbitPath,
  propagateElements,
  stateFromElements,
} from "./ephemeris";
import { normalizeAngle, orbitalPeriod, solveKepler } from "./kepler";

/**
 * Agreement is checked *relative* to each coordinate's own magnitude.
 *
 * The two implementations are not expected to be bit-identical: numpy's `sin`
 * and V8's `Math.sin` can differ in the last bit, and the rotation sums in a
 * different order. That discrepancy scales with magnitude, so an absolute bound
 * would be slack for Mercury at 0.39 AU and unmeetable for Neptune at 30 AU.
 *
 * At 1e-12 relative this is roughly four centimetres at Neptune's distance ---
 * ten orders of magnitude tighter than the arcminute accuracy of the underlying
 * element table, so a genuine divergence still fails loudly.
 */
const RELATIVE_TOLERANCE = fixture.relativeTolerance;

/** Assert two values agree to {@link RELATIVE_TOLERANCE}, scaled by `magnitude`. */
function expectRelativelyClose(actual: number, expected: number, magnitude: number): void {
  const allowed = RELATIVE_TOLERANCE * Math.max(magnitude, 1);
  expect(Math.abs(actual - expected)).toBeLessThan(allowed);
}

describe("parity with the Python reference implementation", () => {
  it("has a non-empty fixture to check against", () => {
    // Guards against the fixture silently regenerating as empty, which would
    // turn every parity assertion below into a vacuous pass.
    expect(fixture.cases.length).toBeGreaterThan(50);
  });

  it.each(fixture.cases)(
    "$body at $date matches Python",
    ({ body, jd, position, velocity }) => {
      const state = bodyState(body, jd);
      const distance = Math.hypot(...position);
      const speed = Math.hypot(...velocity);

      for (let axis = 0; axis < 3; axis += 1) {
        expectRelativelyClose(state.position[axis], position[axis], distance);
        expectRelativelyClose(state.velocity[axis], velocity[axis], speed);
      }
    },
  );

  it("agrees on distance from the Sun for every case", () => {
    for (const { body, jd, position } of fixture.cases) {
      const expected = Math.hypot(...position);
      expectRelativelyClose(bodyState(body, jd).distanceAu, expected, expected);
    }
  });
});

describe("Kepler's equation", () => {
  const eccentricities = [0, 0.0067, 0.0934, 0.2488, 0.5, 0.8, 0.95, 0.99];

  it("satisfies M = E - e sin E across the full grid", () => {
    for (const eccentricity of eccentricities) {
      for (let step = 0; step <= 72; step += 1) {
        const mean = -Math.PI + (step / 72) * 2 * Math.PI;
        const eccentric = solveKepler(mean, eccentricity);

        // Compared modulo a revolution: the solver returns E in [-pi, pi), so an
        // input of M = +pi legitimately comes back as E = -pi.
        const residual = normalizeAngle(
          eccentric - eccentricity * Math.sin(eccentric) - mean,
        );
        expect(Math.abs(residual)).toBeLessThan(1e-11);
      }
    }
  });

  it("collapses to E = M on a circular orbit", () => {
    for (let step = 0; step < 72; step += 1) {
      const mean = -Math.PI + (step / 72) * 2 * Math.PI;
      expect(solveKepler(mean, 0)).toBeCloseTo(mean, 14);
    }
  });

  it("leaves the apsides fixed", () => {
    for (const eccentricity of eccentricities) {
      expect(Math.abs(solveKepler(0, eccentricity))).toBeLessThan(1e-13);
      expect(Math.abs(Math.abs(solveKepler(Math.PI, eccentricity)) - Math.PI)).toBeLessThan(
        1e-11,
      );
    }
  });

  it("rejects non-elliptical eccentricities", () => {
    expect(() => solveKepler(0.5, 1)).toThrow(RangeError);
    expect(() => solveKepler(0.5, -0.1)).toThrow(RangeError);
    expect(() => solveKepler(0.5, 1.4)).toThrow(RangeError);
  });

  it("recovers Kepler's third law by regression", () => {
    // The exponent 3/2 is never fed in; it must fall out of the fitted slope.
    const planets = BODIES.filter((body) => body.isPlanet);
    const logA = planets.map((body) => Math.log10(body.semiMajorAxisAu));
    const logP = planets.map((body) =>
      Math.log10(orbitalPeriod(body.semiMajorAxisAu, CONSTANTS.GM_SUN)),
    );

    const meanLogA = logA.reduce((sum, value) => sum + value, 0) / logA.length;
    const meanLogP = logP.reduce((sum, value) => sum + value, 0) / logP.length;
    let covariance = 0;
    let variance = 0;
    for (let index = 0; index < logA.length; index += 1) {
      covariance += (logA[index] - meanLogA) * (logP[index] - meanLogP);
      variance += (logA[index] - meanLogA) ** 2;
    }

    expect(covariance / variance).toBeCloseTo(1.5, 12);
  });
});

describe("conservation laws", () => {
  const cross = (a: readonly number[], b: readonly number[]) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];

  it.each(BODIES.map((body) => body.id))(
    "%s conserves angular momentum around its orbit",
    (id) => {
      const elements = propagateElements(
        BODIES.find((body) => body.id === id)!,
        CONSTANTS.J2000_JD,
      );

      // The closed form: |h| = sqrt(GM a (1 - e^2)).
      const expected = Math.sqrt(
        CONSTANTS.GM_SUN * elements.semiMajorAxisAu * (1 - elements.eccentricity ** 2),
      );

      for (let step = 0; step < 24; step += 1) {
        const meanAnomaly = (step / 24) * 2 * Math.PI;
        const state = stateFromElements({ ...elements, meanAnomaly });
        const magnitude = Math.hypot(...cross(state.position, state.velocity));

        expect(magnitude / expected).toBeCloseTo(1, 10);
      }
    },
  );

  it.each(BODIES.map((body) => body.id))("%s satisfies the vis-viva equation", (id) => {
    const elements = propagateElements(
      BODIES.find((body) => body.id === id)!,
      CONSTANTS.J2000_JD,
    );

    for (let step = 0; step < 24; step += 1) {
      const meanAnomaly = (step / 24) * 2 * Math.PI;
      const state = stateFromElements({ ...elements, meanAnomaly });

      const speedSquared = state.velocity.reduce((sum, value) => sum + value * value, 0);
      const expected =
        CONSTANTS.GM_SUN * (2 / state.distanceAu - 1 / elements.semiMajorAxisAu);

      expect(speedSquared / expected).toBeCloseTo(1, 10);
    }
  });
});

describe("orbit sampling", () => {
  it("returns a closed curve", () => {
    const path = orbitPath("mars", CONSTANTS.J2000_JD, 128);
    expect(path).toHaveLength(129);

    for (let axis = 0; axis < 3; axis += 1) {
      expect(path[0][axis]).toBeCloseTo(path[path.length - 1][axis], 12);
    }
  });

  it("reaches perihelion and aphelion", () => {
    for (const body of BODIES) {
      const path = orbitPath(body.id, CONSTANTS.J2000_JD, 2048);
      const distances = path.map((point) => Math.hypot(...point));

      const perihelion = body.semiMajorAxisAu * (1 - body.eccentricity);
      const aphelion = body.semiMajorAxisAu * (1 + body.eccentricity);

      expect(Math.min(...distances) / perihelion).toBeCloseTo(1, 4);
      expect(Math.max(...distances) / aphelion).toBeCloseTo(1, 4);
    }
  });
});

describe("time conversions", () => {
  it("puts the J2000 epoch at the right instant", () => {
    // J2000.0 is 2000-01-01 12:00 TT; to the precision that matters here, UTC.
    const j2000 = new Date("2000-01-01T12:00:00Z");
    expect(julianDateFromDate(j2000)).toBeCloseTo(CONSTANTS.J2000_JD, 6);
  });

  it("round-trips dates", () => {
    for (const iso of [
      "1800-01-01T00:00:00Z",
      "1969-07-20T20:17:40Z",
      "2026-07-26T15:42:13Z",
      "2100-12-31T23:59:59Z",
    ]) {
      const moment = new Date(iso);
      const recovered = dateFromJulianDate(julianDateFromDate(moment));
      expect(Math.abs(recovered.getTime() - moment.getTime())).toBeLessThan(1);
    }
  });
});

describe("frame conversion for the scene", () => {
  it("maps the ecliptic pole onto three.js up", () => {
    // Compared component-wise rather than with toEqual: negating a zero yields
    // -0, which toEqual treats as distinct from 0 even though they are the same
    // number for every purpose the scene cares about.
    const world = eclipticToWorld([0, 0, 1]);
    expect(world[0]).toBeCloseTo(0, 15);
    expect(world[1]).toBeCloseTo(1, 15);
    expect(world[2]).toBeCloseTo(0, 15);
  });

  it("preserves vector length", () => {
    const vector = [3, -2, 1.5] as const;
    expect(Math.hypot(...eclipticToWorld(vector))).toBeCloseTo(Math.hypot(...vector), 14);
  });

  it("preserves handedness", () => {
    // A right-handed basis must stay right-handed, or the scene would render a
    // mirror image and every planet would orbit the wrong way.
    const [x, y, z] = [
      eclipticToWorld([1, 0, 0]),
      eclipticToWorld([0, 1, 0]),
      eclipticToWorld([0, 0, 1]),
    ];
    const determinant =
      x[0] * (y[1] * z[2] - y[2] * z[1]) -
      x[1] * (y[0] * z[2] - y[2] * z[0]) +
      x[2] * (y[0] * z[1] - y[1] * z[0]);

    expect(determinant).toBeCloseTo(1, 14);
  });
});
