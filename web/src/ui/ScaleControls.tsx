/**
 * Exposes the two lies every solar-system diagram tells, as switches.
 *
 * See `lib/scale.ts` for why the distortion exists at all. The point of putting it
 * in the UI rather than burying it in a constant is that the viewer can *see* the
 * difference, which teaches the geometry better than any caption.
 */

import { useSimulation } from "../state/simulation";

const VIEW_PRESETS = [
  { view: "inner", label: "Inner" },
  { view: "outer", label: "Outer" },
  { view: "all", label: "All" },
] as const;

const STAR_MODES = [
  { mode: "sky", label: "Sky", hint: "Gaia stars as seen from here" },
  { mode: "galactic", label: "3D", hint: "their real positions, in parsecs" },
  { mode: "off", label: "Off", hint: "hide the star field" },
] as const;

export function ScaleControls() {
  const {
    distanceMode,
    radiusMode,
    setDistanceMode,
    setRadiusMode,
    requestView,
    selectedBodyId,
    starMode,
    setStarMode,
  } = useSimulation();

  return (
    <div className="panel panel--scale">
      <fieldset className="scale-group">
        <legend>View</legend>
        {VIEW_PRESETS.map((preset) => (
          <button
            key={preset.view}
            type="button"
            className="control control--compact"
            onClick={() => requestView(preset.view)}
          >
            {preset.label}
          </button>
        ))}
        <button
          type="button"
          className="control control--compact"
          onClick={() => requestView("follow")}
          disabled={!selectedBodyId}
          title={selectedBodyId ? undefined : "Select a body first"}
        >
          Focus
        </button>
      </fieldset>

      <fieldset className="scale-group">
        <legend>Distance</legend>
        <button
          type="button"
          className={`control control--compact${distanceMode === "compressed" ? " control--active" : ""}`}
          onClick={() => setDistanceMode("compressed")}
          aria-pressed={distanceMode === "compressed"}
        >
          Compressed
        </button>
        <button
          type="button"
          className={`control control--compact${distanceMode === "linear" ? " control--active" : ""}`}
          onClick={() => setDistanceMode("linear")}
          aria-pressed={distanceMode === "linear"}
        >
          True
        </button>
      </fieldset>

      <fieldset className="scale-group">
        <legend>Body size</legend>
        <button
          type="button"
          className={`control control--compact${radiusMode === "readable" ? " control--active" : ""}`}
          onClick={() => setRadiusMode("readable")}
          aria-pressed={radiusMode === "readable"}
        >
          Readable
        </button>
        <button
          type="button"
          className={`control control--compact${radiusMode === "true" ? " control--active" : ""}`}
          onClick={() => setRadiusMode("true")}
          aria-pressed={radiusMode === "true"}
        >
          True
        </button>
      </fieldset>

      <fieldset className="scale-group">
        <legend>Stars</legend>
        {STAR_MODES.map((preset) => (
          <button
            key={preset.mode}
            type="button"
            className={`control control--compact${starMode === preset.mode ? " control--active" : ""}`}
            onClick={() => setStarMode(preset.mode)}
            aria-pressed={starMode === preset.mode}
            title={preset.hint}
          >
            {preset.label}
          </button>
        ))}
      </fieldset>

      <p className="scale-note">
        {distanceMode === "compressed" ? "Orbits pulled inward by a power law." : "Orbits exactly proportional."}
        {" "}
        {radiusMode === "readable" ? "Bodies enlarged to be visible." : "Bodies at honest size."}
        {starMode === "sky" && " 9,000 real Gaia stars, where they actually are in the sky."}
        {starMode === "galactic" && " Gaia stars at their measured 3D distances — zoom out for the disc."}
      </p>
    </div>
  );
}
