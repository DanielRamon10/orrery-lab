// GENERATED FILE --- do not edit by hand.
//
// Produced by scripts/export_web_data.py from orrery/elements.py and
// orrery/constants.py. Re-run that script after changing either one.
//
// Angles are in degrees and distances in astronomical units, matching
// JPL's published table; the conversion to radians happens in
// src/lib/ephemeris.ts so these numbers stay auditable against the source.

export interface BodyElements {
  /** Lower-case key, e.g. `mars`. */
  readonly id: string;
  /** Display name. */
  readonly name: string;
  /** True for the eight planets; false for Pluto. */
  readonly isPlanet: boolean;

  // Keplerian elements at J2000.0.
  readonly semiMajorAxisAu: number;
  readonly eccentricity: number;
  readonly inclinationDeg: number;
  readonly meanLongitudeDeg: number;
  readonly longitudeOfPerihelionDeg: number;
  readonly longitudeOfAscendingNodeDeg: number;

  // Rates of change, per Julian century.
  readonly semiMajorAxisRate: number;
  readonly eccentricityRate: number;
  readonly inclinationRate: number;
  readonly meanLongitudeRate: number;
  readonly longitudeOfPerihelionRate: number;
  readonly longitudeOfAscendingNodeRate: number;

  // Rendering and read-out metadata.
  /** Mean radius in kilometres. */
  readonly radiusKm: number;
  /** Sidereal rotation period in days; negative means retrograde. */
  readonly rotationPeriodDays: number;
  /** Axial tilt to the orbital plane, in degrees. */
  readonly axialTiltDeg: number;
  /** Representational surface colour. */
  readonly colour: string;
  /** Orbital period in days, from Kepler's third law. */
  readonly periodDays: number;
}

export const BODIES: readonly BodyElements[] = [
  {
    id: "mercury",
    name: "Mercury",
    isPlanet: true,
    semiMajorAxisAu: 0.38709927,
    eccentricity: 0.20563593,
    inclinationDeg: 7.00497902,
    meanLongitudeDeg: 252.2503235,
    longitudeOfPerihelionDeg: 77.45779628,
    longitudeOfAscendingNodeDeg: 48.33076593,
    semiMajorAxisRate: 3.7e-07,
    eccentricityRate: 1.906e-05,
    inclinationRate: -0.00594749,
    meanLongitudeRate: 149472.67411175,
    longitudeOfPerihelionRate: 0.16047689,
    longitudeOfAscendingNodeRate: -0.12534081,
    radiusKm: 2439.7,
    rotationPeriodDays: 58.646,
    axialTiltDeg: 0.034,
    colour: "#9a8f88",
    periodDays: 87.96946593127765,
  },
  {
    id: "venus",
    name: "Venus",
    isPlanet: true,
    semiMajorAxisAu: 0.72333566,
    eccentricity: 0.00677672,
    inclinationDeg: 3.39467605,
    meanLongitudeDeg: 181.9790995,
    longitudeOfPerihelionDeg: 131.60246718,
    longitudeOfAscendingNodeDeg: 76.67984255,
    semiMajorAxisRate: 3.9e-06,
    eccentricityRate: -4.107e-05,
    inclinationRate: -0.0007889,
    meanLongitudeRate: 58517.81538729,
    longitudeOfPerihelionRate: 0.00268329,
    longitudeOfAscendingNodeRate: -0.27769418,
    radiusKm: 6051.8,
    rotationPeriodDays: -243.025,
    axialTiltDeg: 177.36,
    colour: "#d9b98c",
    periodDays: 224.70267418246885,
  },
  {
    id: "earth",
    name: "Earth",
    isPlanet: true,
    semiMajorAxisAu: 1.00000261,
    eccentricity: 0.01671123,
    inclinationDeg: -1.531e-05,
    meanLongitudeDeg: 100.46457166,
    longitudeOfPerihelionDeg: 102.93768193,
    longitudeOfAscendingNodeDeg: 0.0,
    semiMajorAxisRate: 5.62e-06,
    eccentricityRate: -4.392e-05,
    inclinationRate: -0.01294668,
    meanLongitudeRate: 35999.37244981,
    longitudeOfPerihelionRate: 0.32327364,
    longitudeOfAscendingNodeRate: 0.0,
    radiusKm: 6371.0,
    rotationPeriodDays: 0.99726968,
    axialTiltDeg: 23.4393,
    colour: "#4a7fb5",
    periodDays: 365.25832830801806,
  },
  {
    id: "mars",
    name: "Mars",
    isPlanet: true,
    semiMajorAxisAu: 1.52371034,
    eccentricity: 0.0933941,
    inclinationDeg: 1.84969142,
    meanLongitudeDeg: -4.55343205,
    longitudeOfPerihelionDeg: -23.94362959,
    longitudeOfAscendingNodeDeg: 49.55953891,
    semiMajorAxisRate: 1.847e-05,
    eccentricityRate: 7.882e-05,
    inclinationRate: -0.00813131,
    meanLongitudeRate: 19140.30268499,
    longitudeOfPerihelionRate: 0.44441088,
    longitudeOfAscendingNodeRate: -0.29257343,
    radiusKm: 3389.5,
    rotationPeriodDays: 1.02595676,
    axialTiltDeg: 25.19,
    colour: "#c1593a",
    periodDays: 686.9925840073605,
  },
  {
    id: "jupiter",
    name: "Jupiter",
    isPlanet: true,
    semiMajorAxisAu: 5.202887,
    eccentricity: 0.04838624,
    inclinationDeg: 1.30439695,
    meanLongitudeDeg: 34.39644051,
    longitudeOfPerihelionDeg: 14.72847983,
    longitudeOfAscendingNodeDeg: 100.47390909,
    semiMajorAxisRate: -0.00011607,
    eccentricityRate: -0.00013253,
    inclinationRate: -0.00183714,
    meanLongitudeRate: 3034.74612775,
    longitudeOfPerihelionRate: 0.21252668,
    longitudeOfAscendingNodeRate: 0.20469106,
    radiusKm: 69911.0,
    rotationPeriodDays: 0.41354,
    axialTiltDeg: 3.13,
    colour: "#c9a27e",
    periodDays: 4334.759603064555,
  },
  {
    id: "saturn",
    name: "Saturn",
    isPlanet: true,
    semiMajorAxisAu: 9.53667594,
    eccentricity: 0.05386179,
    inclinationDeg: 2.48599187,
    meanLongitudeDeg: 49.95424423,
    longitudeOfPerihelionDeg: 92.59887831,
    longitudeOfAscendingNodeDeg: 113.66242448,
    semiMajorAxisRate: -0.0012506,
    eccentricityRate: -0.00050991,
    inclinationRate: 0.00193609,
    meanLongitudeRate: 1222.49362201,
    longitudeOfPerihelionRate: -0.41897216,
    longitudeOfAscendingNodeRate: -0.28867794,
    radiusKm: 58232.0,
    rotationPeriodDays: 0.44401,
    axialTiltDeg: 26.73,
    colour: "#d8c08a",
    periodDays: 10757.069262176126,
  },
  {
    id: "uranus",
    name: "Uranus",
    isPlanet: true,
    semiMajorAxisAu: 19.18916464,
    eccentricity: 0.04725744,
    inclinationDeg: 0.77263783,
    meanLongitudeDeg: 313.23810451,
    longitudeOfPerihelionDeg: 170.9542763,
    longitudeOfAscendingNodeDeg: 74.01692503,
    semiMajorAxisRate: -0.00196176,
    eccentricityRate: -4.397e-05,
    inclinationRate: -0.00242939,
    meanLongitudeRate: 428.48202785,
    longitudeOfPerihelionRate: 0.40805281,
    longitudeOfAscendingNodeRate: 0.04240589,
    radiusKm: 25362.0,
    rotationPeriodDays: -0.71833,
    axialTiltDeg: 97.77,
    colour: "#8ec5d1",
    periodDays: 30703.121445019293,
  },
  {
    id: "neptune",
    name: "Neptune",
    isPlanet: true,
    semiMajorAxisAu: 30.06992276,
    eccentricity: 0.00859048,
    inclinationDeg: 1.77004347,
    meanLongitudeDeg: -55.12002969,
    longitudeOfPerihelionDeg: 44.96476227,
    longitudeOfAscendingNodeDeg: 131.78422574,
    semiMajorAxisRate: 0.00026291,
    eccentricityRate: 5.105e-05,
    inclinationRate: 0.00035372,
    meanLongitudeRate: 218.45945325,
    longitudeOfPerihelionRate: -0.32241464,
    longitudeOfAscendingNodeRate: -0.00508664,
    radiusKm: 24622.0,
    rotationPeriodDays: 0.67125,
    axialTiltDeg: 28.32,
    colour: "#4a6fc4",
    periodDays: 60227.78559374613,
  },
  {
    id: "pluto",
    name: "Pluto",
    isPlanet: false,
    semiMajorAxisAu: 39.48211675,
    eccentricity: 0.2488273,
    inclinationDeg: 17.14001206,
    meanLongitudeDeg: 238.92903833,
    longitudeOfPerihelionDeg: 224.06891629,
    longitudeOfAscendingNodeDeg: 110.30393684,
    semiMajorAxisRate: -0.00031596,
    eccentricityRate: 5.17e-05,
    inclinationRate: 4.818e-05,
    meanLongitudeRate: 145.20780515,
    longitudeOfPerihelionRate: -0.04062942,
    longitudeOfAscendingNodeRate: -0.01183482,
    radiusKm: 1188.3,
    rotationPeriodDays: -6.3872,
    axialTiltDeg: 122.53,
    colour: "#b3a394",
    periodDays: 90614.78606946526,
  },
];

/** Physical constants the scene needs, kept in step with orrery/constants.py. */
export const CONSTANTS = {
  J2000_JD: 2451545.0,
  AU_KM: 149597870.7,
  GM_SUN: 0.00029591220828559115,
  SUN_RADIUS_KM: 695700.0,
  SUN_COLOUR: "#ffd27a",
  SUN_ROTATION_PERIOD_DAYS: 25.38,
} as const;
