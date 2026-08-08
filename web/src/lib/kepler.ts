/**
 * Kepler's equation, in the browser.
 *
 * A port of `orrery/kepler.py`. The scene needs to solve this on every animation
 * frame for every body, so it runs here rather than being fetched: a scrubbable
 * timeline over centuries is not something you can precompute into a payload.
 *
 * The Python module is the reference implementation, and
 * `ephemeris.test.ts` holds this port to the positions Python produces.
 */

const TWO_PI = 2 * Math.PI;

/**
 * Wrap an angle in radians into a single revolution.
 *
 * Wrapping to `[-pi, pi)` is not cosmetic: it keeps Newton's starting guess close
 * to the root, which is what makes the iteration converge in a handful of steps.
 */
export function normalizeAngle(angle: number, centered = true): number {
  let wrapped = angle % TWO_PI;
  if (wrapped < 0) wrapped += TWO_PI;
  if (centered && wrapped >= Math.PI) wrapped -= TWO_PI;
  return wrapped;
}

/**
 * Solve `M = E - e sin E` for the eccentric anomaly `E`.
 *
 * Newton-Raphson, seeded with Danby's starting value, falling back to bisection
 * for anything Newton fails to converge on. The fallback cannot fail: rearranging
 * the equation to `E = M + e sin E` shows the root is always trapped inside
 * `[M - e, M + e]`, so that interval is a guaranteed bracket.
 *
 * @param meanAnomaly Mean anomaly in radians.
 * @param eccentricity Orbital eccentricity, `0 <= e < 1`.
 * @param tolerance Absolute tolerance on the residual, in radians.
 * @returns Eccentric anomaly in radians, wrapped to `[-pi, pi)`.
 * @throws RangeError if the eccentricity is not that of an ellipse.
 */
export function solveKepler(
  meanAnomaly: number,
  eccentricity: number,
  tolerance = 1e-13,
  maxIterations = 64,
): number {
  if (!(eccentricity >= 0) || eccentricity >= 1) {
    throw new RangeError(
      `solveKepler handles elliptical orbits only; got eccentricity ${eccentricity}`,
    );
  }

  const mean = normalizeAngle(meanAnomaly);

  // Danby's seed: markedly better than E0 = M once the orbit is eccentric.
  let eccentric = mean + Math.sign(Math.sin(mean)) * 0.85 * eccentricity;

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    const residual = eccentric - eccentricity * Math.sin(eccentric) - mean;
    if (Math.abs(residual) < tolerance) return normalizeAngle(eccentric);

    // The derivative is >= 1 - e > 0 for any ellipse, so it never vanishes.
    const derivative = 1 - eccentricity * Math.cos(eccentric);
    eccentric -= residual / derivative;
  }

  // Newton stalled. Bisect the guaranteed bracket instead.
  let low = mean - eccentricity;
  let high = mean + eccentricity;
  for (let iteration = 0; iteration < 200; iteration += 1) {
    const middle = 0.5 * (low + high);
    const residual = middle - eccentricity * Math.sin(middle) - mean;
    if (residual > 0) high = middle;
    else low = middle;
    if (high - low < tolerance) break;
  }
  return normalizeAngle(0.5 * (low + high));
}

/**
 * Convert eccentric anomaly to true anomaly --- the angle actually subtended at
 * the Sun, which is what a telescope would measure.
 *
 * Uses the half-angle form so that precision holds up near aphelion.
 */
export function trueAnomalyFromEccentric(
  eccentricAnomaly: number,
  eccentricity: number,
): number {
  const half = 0.5 * eccentricAnomaly;
  return (
    2 *
    Math.atan2(
      Math.sqrt(1 + eccentricity) * Math.sin(half),
      Math.sqrt(1 - eccentricity) * Math.cos(half),
    )
  );
}

/** Mean angular rate `n = sqrt(GM / a^3)`. This is Kepler's third law. */
export function meanMotion(semiMajorAxisAu: number, gm: number): number {
  return Math.sqrt(gm / semiMajorAxisAu ** 3);
}

/** Orbital period `P = 2 pi / n`, in the time unit of `gm`. */
export function orbitalPeriod(semiMajorAxisAu: number, gm: number): number {
  return TWO_PI / meanMotion(semiMajorAxisAu, gm);
}
