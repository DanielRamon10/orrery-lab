/**
 * Orbital elements to 3D positions, in the browser.
 *
 * A port of `orrery/ephemeris.py`; see that module for the derivation. In short:
 * propagate the elements to the date, solve Kepler's equation, place the body on
 * a flat ellipse, then rotate that ellipse into the ecliptic frame with
 * `R = Rz(Omega) . Rx(i) . Rz(omega)`.
 *
 * All output is in the **heliocentric ecliptic J2000** frame, positions in AU and
 * velocities in AU/day --- the same frame and units as the Python package, which
 * is what lets `ephemeris.test.ts` compare the two directly. Converting to
 * three.js's Y-up world happens in the scene layer, not here.
 */

import { BODIES, CONSTANTS, type BodyElements } from "../data/elements.generated";
import { meanMotion, solveKepler } from "./kepler";

const DEG_TO_RAD = Math.PI / 180;
const JULIAN_CENTURY_DAYS = 36525;

export type Vector3Tuple = readonly [number, number, number];

export interface State {
  /** Heliocentric position in AU, ecliptic J2000. */
  readonly position: Vector3Tuple;
  /** Heliocentric velocity in AU/day, same frame. */
  readonly velocity: Vector3Tuple;
  /** Distance from the Sun, in AU. */
  readonly distanceAu: number;
  /** Orbital speed, in km/s. */
  readonly speedKmPerSecond: number;
}

/** Elements propagated to a date, with angles already in radians. */
export interface PropagatedElements {
  readonly semiMajorAxisAu: number;
  readonly eccentricity: number;
  readonly inclination: number;
  readonly node: number;
  readonly argumentOfPerihelion: number;
  readonly meanAnomaly: number;
}

const BODIES_BY_ID = new Map<string, BodyElements>(BODIES.map((body) => [body.id, body]));

/** Look up a body by its lower-case id, e.g. `"mars"`. */
export function getBody(id: string): BodyElements {
  const body = BODIES_BY_ID.get(id);
  if (!body) throw new Error(`unknown body "${id}"`);
  return body;
}

/**
 * Julian Date of a JavaScript `Date`.
 *
 * `Date.getTime()` counts milliseconds from the Unix epoch, which is itself at a
 * known Julian Date, so this is one subtraction rather than calendar arithmetic.
 */
export function julianDateFromDate(moment: Date): number {
  return moment.getTime() / 86_400_000 + 2_440_587.5;
}

/**
 * Inverse of {@link julianDateFromDate}.
 *
 * Rounded to the nearest millisecond deliberately. A Julian Date near 2.46e6 has
 * about 10 microseconds of float64 resolution left, so the division in
 * {@link julianDateFromDate} is not exactly reversible; without the rounding a
 * round trip can land one millisecond off, and `Date` counts whole milliseconds
 * anyway.
 */
export function dateFromJulianDate(jd: number): Date {
  return new Date(Math.round((jd - 2_440_587.5) * 86_400_000));
}

/**
 * Propagate a body's elements to a Julian Date.
 *
 * The two conversions at the end are the only subtlety in using JPL's table: the
 * published columns are *longitudes* measured from the reference direction, while
 * the geometry below needs the argument of perihelion (`omega_bar - Omega`) and
 * the mean anomaly (`L - omega_bar`).
 */
export function propagateElements(body: BodyElements, jd: number): PropagatedElements {
  const centuries = (jd - CONSTANTS.J2000_JD) / JULIAN_CENTURY_DAYS;

  const meanLongitude = body.meanLongitudeDeg + body.meanLongitudeRate * centuries;
  const perihelion =
    body.longitudeOfPerihelionDeg + body.longitudeOfPerihelionRate * centuries;
  const node =
    body.longitudeOfAscendingNodeDeg + body.longitudeOfAscendingNodeRate * centuries;

  return {
    semiMajorAxisAu: body.semiMajorAxisAu + body.semiMajorAxisRate * centuries,
    eccentricity: body.eccentricity + body.eccentricityRate * centuries,
    inclination: (body.inclinationDeg + body.inclinationRate * centuries) * DEG_TO_RAD,
    node: node * DEG_TO_RAD,
    argumentOfPerihelion: (perihelion - node) * DEG_TO_RAD,
    meanAnomaly: (meanLongitude - perihelion) * DEG_TO_RAD,
  };
}

/**
 * Turn propagated elements into a full state vector.
 *
 * The velocity comes from differentiating the position with respect to time. That
 * introduces `dE/dt`, which follows from differentiating Kepler's equation itself:
 * `M = E - e sin E` gives `n = (1 - e cos E) dE/dt`.
 */
export function stateFromElements(elements: PropagatedElements): State {
  const {
    semiMajorAxisAu: a,
    eccentricity: e,
    inclination,
    node,
    argumentOfPerihelion,
    meanAnomaly,
  } = elements;

  const eccentricAnomaly = solveKepler(meanAnomaly, e);
  const cosE = Math.cos(eccentricAnomaly);
  const sinE = Math.sin(eccentricAnomaly);
  const sqrtOneMinusESquared = Math.sqrt(1 - e * e);

  // Position and velocity in the flat orbital (perifocal) plane.
  const xPerifocal = a * (cosE - e);
  const yPerifocal = a * sqrtOneMinusESquared * sinE;

  const eccentricRate = meanMotion(a, CONSTANTS.GM_SUN) / (1 - e * cosE);
  const vxPerifocal = -a * sinE * eccentricRate;
  const vyPerifocal = a * sqrtOneMinusESquared * cosE * eccentricRate;

  // Rz(node) . Rx(inclination) . Rz(argumentOfPerihelion), written out so it can
  // be checked line by line against the published form.
  const cosI = Math.cos(inclination);
  const sinI = Math.sin(inclination);
  const cosNode = Math.cos(node);
  const sinNode = Math.sin(node);
  const cosArg = Math.cos(argumentOfPerihelion);
  const sinArg = Math.sin(argumentOfPerihelion);

  const m00 = cosArg * cosNode - sinArg * sinNode * cosI;
  const m01 = -sinArg * cosNode - cosArg * sinNode * cosI;
  const m10 = cosArg * sinNode + sinArg * cosNode * cosI;
  const m11 = -sinArg * sinNode + cosArg * cosNode * cosI;
  const m20 = sinArg * sinI;
  const m21 = cosArg * sinI;

  const position: Vector3Tuple = [
    m00 * xPerifocal + m01 * yPerifocal,
    m10 * xPerifocal + m11 * yPerifocal,
    m20 * xPerifocal + m21 * yPerifocal,
  ];
  const velocity: Vector3Tuple = [
    m00 * vxPerifocal + m01 * vyPerifocal,
    m10 * vxPerifocal + m11 * vyPerifocal,
    m20 * vxPerifocal + m21 * vyPerifocal,
  ];

  const distanceAu = Math.hypot(...position);
  const speedAuPerDay = Math.hypot(...velocity);

  return {
    position,
    velocity,
    distanceAu,
    speedKmPerSecond: (speedAuPerDay * CONSTANTS.AU_KM) / 86_400,
  };
}

/** Heliocentric state of a named body at a Julian Date. */
export function bodyState(id: string, jd: number): State {
  return stateFromElements(propagateElements(getBody(id), jd));
}

/**
 * Sample one full closed orbit, for drawing the trajectory line.
 *
 * The elements are frozen at `jd` and only the mean anomaly is swept through a
 * revolution, which yields a closed ellipse --- the orbit as it stands on that
 * date. Letting the elements drift as well would draw a slowly opening spiral.
 */
export function orbitPath(id: string, jd: number, samples = 512): Vector3Tuple[] {
  const elements = propagateElements(getBody(id), jd);
  const path: Vector3Tuple[] = [];

  for (let index = 0; index <= samples; index += 1) {
    const meanAnomaly = (index / samples) * 2 * Math.PI;
    path.push(stateFromElements({ ...elements, meanAnomaly }).position);
  }
  return path;
}

/**
 * Map an ecliptic vector into three.js world space.
 *
 * three.js treats **+Y** as up, while the ecliptic frame treats **+Z** as its
 * pole. The permutation below swaps them while preserving handedness, so the
 * planets orbit in the scene's horizontal plane and the small out-of-plane
 * component (the inclination) reads as height.
 */
export function eclipticToWorld([x, y, z]: Vector3Tuple): [number, number, number] {
  return [x, z, -y];
}
