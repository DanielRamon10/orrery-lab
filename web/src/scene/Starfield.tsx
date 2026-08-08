/**
 * Real stars, from Gaia DR3.
 *
 * Phase 2 used drei's procedural `<Stars>`: random points that read as a sky and mean
 * nothing. These are 9,000 actual Gaia sources at their measured directions,
 * brightnesses and colours, so the constellations in the scene are the constellations.
 *
 * Two modes, from the same rows:
 *
 * - **sky** — each star painted on a large sphere around the solar system, at constant
 *   screen size. This is the view from inside: what you would actually see looking up.
 * - **galactic** — the same stars at their measured 3D positions in parsecs, Sun at
 *   the origin. Pull the camera back and the flattened disc appears, because it is
 *   there in the data.
 *
 * Loading is handled by `lib/starCatalogue.ts`, which memoises one fetch shared with
 * the Hertzsprung-Russell panel so a selection there maps to these exact rows.
 */

import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import { AdditiveBlending, BufferAttribute, BufferGeometry, type Points } from "three";

import { useStarCatalogue } from "../lib/starCatalogue";
import { useSimulation } from "../state/simulation";

/** Radius of the sphere the sky is painted on, in scene units. */
const SKY_RADIUS = 900;

/** Scene units per parsec in galactic mode. */
const UNITS_PER_PARSEC = 0.5;

export type StarMode = "sky" | "galactic" | "off";

/**
 * Display brightness from apparent magnitude, in `[0.12, 1]`.
 *
 * Brightness is carried in the **colour**, not the point size. The obvious approach —
 * a per-vertex `size` attribute — silently does nothing: three.js's `PointsMaterial`
 * has one scalar size for the whole cloud and ignores any size attribute, so every
 * star renders identically and the sky comes out flat. Per-point sizing needs a
 * custom shader, which is a lot of machinery for an effect that additive blending
 * gives for free.
 *
 * Magnitudes are logarithmic and run backwards: each step of 1 is a factor of about
 * 2.512 in received light, and smaller means brighter. Applying that ratio literally
 * would make the faintest stars 50x dimmer than the brightest and leave most of the
 * sky invisible, so the exponent is softened. Ordering is preserved — Sirius is still
 * the brightest thing up there.
 */
function brightnessFromMagnitude(magnitude: number, brightest: number): number {
  return Math.max(0.12, Math.min(1, Math.pow(2.512, -0.42 * (magnitude - brightest))));
}

interface StarfieldProps {
  readonly mode: StarMode;
}

export function Starfield({ mode }: StarfieldProps) {
  const catalogue = useStarCatalogue();
  const { starSelection } = useSimulation();
  const pointsRef = useRef<Points>(null);

  const geometry = useMemo(() => {
    if (!catalogue || mode === "off") return null;

    // Galactic mode can only show stars whose parallax was good enough to invert, so
    // the two modes hold different numbers of points — see orrery/gaia.py for why
    // some are deliberately missing rather than guessed at.
    const indices: number[] = [];
    for (let index = 0; index < catalogue.count; index += 1) {
      if (mode === "sky" || catalogue.galactic[index] !== null) indices.push(index);
    }

    const positions = new Float32Array(indices.length * 3);
    const colours = new Float32Array(indices.length * 3);
    const brightest = Math.min(...catalogue.magnitude);

    /** Is this star inside the box dragged on the HR diagram? */
    const isSelected = (index: number) => {
      if (!starSelection) return true;
      const colour = catalogue.bpRp[index];
      const magnitude = catalogue.absoluteG[index];
      if (colour === null || magnitude === null) return false;
      return (
        colour >= starSelection.colour[0] &&
        colour <= starSelection.colour[1] &&
        magnitude >= starSelection.magnitude[0] &&
        magnitude <= starSelection.magnitude[1]
      );
    };

    indices.forEach((source, target) => {
      if (mode === "sky") {
        const [x, y, z] = catalogue.direction[source];
        // Equatorial to the scene's Y-up world, matching lib/ephemeris.ts.
        positions[target * 3] = x * SKY_RADIUS;
        positions[target * 3 + 1] = z * SKY_RADIUS;
        positions[target * 3 + 2] = -y * SKY_RADIUS;
      } else {
        const [x, y, z] = catalogue.galactic[source]!;
        positions[target * 3] = x * UNITS_PER_PARSEC;
        positions[target * 3 + 1] = z * UNITS_PER_PARSEC;
        positions[target * 3 + 2] = -y * UNITS_PER_PARSEC;
      }

      const [red, green, blue] = catalogue.colour[source];
      let brightness = brightnessFromMagnitude(catalogue.magnitude[source], brightest);

      // A selection dims everything else rather than hiding it, so the highlighted
      // population is seen *in context* — which is the point of linking the two views.
      if (starSelection) brightness *= isSelected(source) ? 1.6 : 0.08;

      colours[target * 3] = Math.min(1, red * brightness);
      colours[target * 3 + 1] = Math.min(1, green * brightness);
      colours[target * 3 + 2] = Math.min(1, blue * brightness);
    });

    const built = new BufferGeometry();
    built.setAttribute("position", new BufferAttribute(positions, 3));
    built.setAttribute("color", new BufferAttribute(colours, 3));
    return built;
  }, [catalogue, mode, starSelection]);

  // In sky mode the sphere is a backdrop, so it rides with the camera and never gets
  // closer no matter how far you travel — the same reason real constellations do not
  // change shape when you drive across a country.
  useFrame(({ camera }) => {
    if (mode === "sky" && pointsRef.current) {
      pointsRef.current.position.copy(camera.position);
    }
  });

  useEffect(() => () => geometry?.dispose(), [geometry]);

  if (!geometry) return null;

  return (
    <points ref={pointsRef} geometry={geometry} frustumCulled={false}>
      <pointsMaterial
        vertexColors
        // Constant screen size for the sky; real perspective for the 3D view, where
        // shrinking with distance is the depth cue that makes the disc readable.
        sizeAttenuation={mode === "galactic"}
        size={mode === "galactic" ? 1.6 : 1.7}
        // Additive so overlapping stars brighten rather than occluding one another,
        // which is how light actually behaves.
        blending={AdditiveBlending}
        depthWrite={false}
        transparent
      />
    </points>
  );
}
