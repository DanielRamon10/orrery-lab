/**
 * Live numbers for the selected body.
 *
 * This is the panel that makes the model a *model* rather than an animation: every
 * value here is computed from the same solver that places the sphere, so the number
 * and the picture cannot disagree.
 */

import { BODIES, CONSTANTS } from "../data/elements.generated";
import { bodyState, propagateElements } from "../lib/ephemeris";
import { radiusExaggeration } from "../lib/scale";
import { useSimulation, useSimulationTime } from "../state/simulation";

const RAD_TO_DEG = 180 / Math.PI;

function formatNumber(value: number, digits: number): string {
  return value.toLocaleString("en-GB", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function Readout() {
  const { selectedBodyId, radiusMode } = useSimulation();
  const { jd } = useSimulationTime();

  if (!selectedBodyId) {
    return (
      <div className="panel panel--readout panel--readout-empty">
        <p className="hint">Click a body to inspect it.</p>
      </div>
    );
  }

  const body = BODIES.find((candidate) => candidate.id === selectedBodyId);
  if (!body) return null;

  const state = bodyState(body.id, jd);
  const elements = propagateElements(body, jd);
  const exaggeration = radiusExaggeration(body.radiusKm, radiusMode);

  const lightMinutes = (state.distanceAu * CONSTANTS.AU_KM) / 299_792.458 / 60;

  const rows: readonly { readonly label: string; readonly value: string }[] = [
    { label: "Distance from Sun", value: `${formatNumber(state.distanceAu, 4)} AU` },
    {
      label: "",
      value: `${formatNumber((state.distanceAu * CONSTANTS.AU_KM) / 1e6, 2)} million km · ${formatNumber(lightMinutes, 1)} light-min`,
    },
    { label: "Orbital speed", value: `${formatNumber(state.speedKmPerSecond, 2)} km/s` },
    { label: "Orbital period", value: `${formatNumber(body.periodDays / 365.25, 3)} years` },
    { label: "Semi-major axis", value: `${formatNumber(elements.semiMajorAxisAu, 5)} AU` },
    { label: "Eccentricity", value: formatNumber(elements.eccentricity, 5) },
    {
      label: "Inclination",
      value: `${formatNumber(Math.abs(elements.inclination * RAD_TO_DEG), 3)}°`,
    },
    { label: "Mean radius", value: `${formatNumber(body.radiusKm, 0)} km` },
    {
      label: "Rotation period",
      value: `${formatNumber(Math.abs(body.rotationPeriodDays), 3)} d${
        body.rotationPeriodDays < 0 ? " (retrograde)" : ""
      }`,
    },
    { label: "Axial tilt", value: `${formatNumber(body.axialTiltDeg, 2)}°` },
  ];

  return (
    <div className="panel panel--readout">
      <header className="readout-header">
        <span className="readout-swatch" style={{ background: body.colour }} aria-hidden="true" />
        <h2>{body.name}</h2>
        {!body.isPlanet && <span className="tag">dwarf planet</span>}
      </header>

      <dl className="readout-grid">
        {rows.map((row, index) => (
          <div className="readout-row" key={`${row.label}-${index}`}>
            <dt>{row.label}</dt>
            <dd>{row.value}</dd>
          </div>
        ))}
      </dl>

      <p className="readout-note">
        {exaggeration > 1.5 ? (
          <>
            Drawn <strong>{formatNumber(exaggeration, 0)}×</strong> larger than life so it is
            visible. Switch the radius scale to <em>true</em> to see the honest size.
          </>
        ) : (
          <>Drawn at true scale — which is why it is barely a speck.</>
        )}
      </p>
    </div>
  );
}
