/**
 * The scale problem, and how this scene handles it.
 *
 * The solar system cannot be drawn honestly and legibly at the same time. Earth's
 * radius is 4.3e-5 AU; Neptune orbits at 30 AU. A true-scale render that fits
 * Neptune on screen makes every planet smaller than one pixel, and the Sun itself
 * barely a dot. Every solar-system illustration you have ever seen lies about this,
 * usually without saying so.
 *
 * Rather than pick one lie and hide it, the scene exposes both axes of distortion
 * as explicit modes and always reports the exaggeration factor in the UI:
 *
 * - **Distance** — `linear` is physically proportional. `compressed` applies a
 *   power law, which pulls Neptune in close enough that the inner planets are not
 *   a single knot at the centre. Ordering and relative spacing survive; absolute
 *   ratios do not.
 * - **Radius** — `readable` exaggerates the bodies so they are visible, but keeps
 *   them *ordered and comparable* by using a power law rather than a flat minimum
 *   size, so Jupiter still reads as far bigger than Mars. `true` is physically
 *   exact, which is worth switching to once: it is the only honest way to feel how
 *   empty the solar system actually is.
 */

import { CONSTANTS } from "../data/elements.generated";

export type DistanceMode = "linear" | "compressed";
export type RadiusMode = "readable" | "true";

/** Scene units per AU in linear mode. Sets the overall size of the model. */
export const UNITS_PER_AU = 10;

/**
 * Exponent for compressed distance. 0.55 is chosen so that the ratio between
 * Neptune's and Mercury's orbits drops from about 78:1 to roughly 11:1 --- enough
 * to see the inner planets move while Neptune stays on screen.
 */
const DISTANCE_COMPRESSION_EXPONENT = 0.55;

/**
 * Exponent for readable radii. 0.4 keeps the *ordering* of every body and a clear
 * sense of relative size (Jupiter reads as much larger than Earth) while
 * compressing the real 29:1 Jupiter-to-Mercury radius ratio into about 4:1, which
 * fits on screen alongside the orbits.
 */
const RADIUS_COMPRESSION_EXPONENT = 0.4;

/** Reference radius for the readable law: Earth. */
const EARTH_RADIUS_KM = 6371;

/** Scene radius given to Earth in readable mode; everything else scales from it. */
const EARTH_READABLE_RADIUS = 0.32;

/** Convert a heliocentric distance in AU to scene units. */
export function scaleDistance(distanceAu: number, mode: DistanceMode): number {
  if (mode === "linear") return distanceAu * UNITS_PER_AU;
  return Math.pow(distanceAu, DISTANCE_COMPRESSION_EXPONENT) * UNITS_PER_AU;
}

/**
 * Scale a whole position vector.
 *
 * In compressed mode the direction is preserved and only the magnitude is bent,
 * so the geometry stays recognisable: conjunctions still line up, and an inclined
 * orbit still leaves the plane by a proportional amount.
 */
export function scalePosition(
  [x, y, z]: readonly [number, number, number],
  mode: DistanceMode,
): [number, number, number] {
  if (mode === "linear") {
    return [x * UNITS_PER_AU, y * UNITS_PER_AU, z * UNITS_PER_AU];
  }

  const distance = Math.hypot(x, y, z);
  if (distance === 0) return [0, 0, 0];

  const scaled = scaleDistance(distance, mode) / distance;
  return [x * scaled, y * scaled, z * scaled];
}

/** Convert a body radius in kilometres to scene units. */
export function scaleRadius(radiusKm: number, mode: RadiusMode): number {
  if (mode === "true") {
    // Physically exact: the same units-per-AU that positions use.
    return (radiusKm / CONSTANTS.AU_KM) * UNITS_PER_AU;
  }
  return (
    EARTH_READABLE_RADIUS * Math.pow(radiusKm / EARTH_RADIUS_KM, RADIUS_COMPRESSION_EXPONENT)
  );
}

/**
 * How many times larger than life a body is being drawn.
 *
 * Surfaced in the UI so the exaggeration is never silent. Returns 1 in true mode.
 */
export function radiusExaggeration(radiusKm: number, mode: RadiusMode): number {
  if (mode === "true") return 1;
  return scaleRadius(radiusKm, mode) / scaleRadius(radiusKm, "true");
}

/**
 * The Sun in readable mode.
 *
 * Applying the same power law to the Sun gives a sphere that swallows Mercury's
 * orbit, so it gets its own smaller multiplier. Physically indefensible, visually
 * necessary, and reported as such.
 */
export function sunRadius(mode: RadiusMode): number {
  if (mode === "true") return (CONSTANTS.SUN_RADIUS_KM / CONSTANTS.AU_KM) * UNITS_PER_AU;
  return 0.85;
}
