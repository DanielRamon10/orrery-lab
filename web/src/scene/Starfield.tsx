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
 * The catalogue is loaded with `fetch` rather than imported. A 600 kB JSON `import`
 * would be inlined into the main bundle by Vite and block first paint; fetching it
 * keeps it a separate cacheable asset and lets the solar system render immediately
 * while the sky arrives a moment later.
 */

import { useFrame } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import { AdditiveBlending, BufferAttribute, BufferGeometry, type Points } from "three";

import starsUrl from "../data/stars.generated.json?url";

/** Radius of the sphere the sky is painted on, in scene units. */
const SKY_RADIUS = 900;

/** Scene units per parsec in galactic mode. */
const UNITS_PER_PARSEC = 0.5;

export type StarMode = "sky" | "galactic" | "off";

interface StarCatalogue {
  readonly count: number;
  readonly direction: number[][];
  readonly magnitude: number[];
  readonly colour: number[][];
  readonly galactic: (number[] | null)[];
}

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

function useStarCatalogue(): StarCatalogue | null {
  const [catalogue, setCatalogue] = useState<StarCatalogue | null>(null);

  useEffect(() => {
    let cancelled = false;

    fetch(starsUrl)
      .then((response) => {
        if (!response.ok) throw new Error(`star catalogue: HTTP ${response.status}`);
        return response.json();
      })
      .then((data: StarCatalogue) => {
        if (!cancelled) setCatalogue(data);
      })
      .catch((error) => {
        // A missing star field is a degraded scene, not a broken one: the solar
        // system is the subject and it renders fine without a sky.
        console.warn("could not load the Gaia star field:", error);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return catalogue;
}

interface StarfieldProps {
  readonly mode: StarMode;
}

export function Starfield({ mode }: StarfieldProps) {
  const catalogue = useStarCatalogue();
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
      const brightness = brightnessFromMagnitude(catalogue.magnitude[source], brightest);
      colours[target * 3] = red * brightness;
      colours[target * 3 + 1] = green * brightness;
      colours[target * 3 + 2] = blue * brightness;
    });

    const built = new BufferGeometry();
    built.setAttribute("position", new BufferAttribute(positions, 3));
    built.setAttribute("color", new BufferAttribute(colours, 3));
    return built;
  }, [catalogue, mode]);

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
