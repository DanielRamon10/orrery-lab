/**
 * Moves the camera to fit a chosen band of the solar system.
 *
 * Framing is not a cosmetic detail here: the compressed-distance scene spans about
 * 85 units out to Pluto's aphelion, and the linear one spans 490. A single hardcoded
 * camera position cannot serve both, and the wrong one leaves half the planets off
 * screen. So the distances below are *derived* from the same scale functions the
 * bodies use, and recomputed whenever the distance mode changes.
 *
 * The move is eased rather than snapped, because a cut makes it genuinely hard to
 * tell whether you zoomed out or the model changed size.
 */

import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useRef } from "react";
import { Vector3 } from "three";

import { bodyState, eclipticToWorld, getBody } from "../lib/ephemeris";
import { scaleDistance, scalePosition } from "../lib/scale";
import { useSimulation } from "../state/simulation";

/**
 * The slice of OrbitControls this component touches.
 *
 * Typed structurally rather than imported from `three-stdlib`, which is a
 * transitive dependency of drei and not something this project declares directly.
 */
interface OrbitControlsLike {
  target: Vector3;
  update: () => void;
}

/** Vertical field of view of the scene camera, in degrees. Must match App.tsx. */
const FOV_DEG = 45;

/** Extra room around the fitted extent, so nothing sits flush against the edge. */
const FIT_MARGIN = 1.1;

/** How high above the ecliptic to sit, as an angle in degrees. */
const ELEVATION_DEG = 28;

/** Fraction of the remaining distance covered per frame at 60 Hz. */
const EASING = 0.075;

/** Aphelion of the body that bounds each view band, in AU. */
const BAND_EDGE_AU = {
  inner: 1.666, // Mars
  outer: 30.33, // Neptune
  all: 49.31, // Pluto
} as const;

export function CameraDirector() {
  const { viewRequest, distanceMode, selectedBodyId, clock } = useSimulation();
  const camera = useThree((state) => state.camera);
  const controls = useThree((state) => state.controls) as OrbitControlsLike | null;

  const goalPosition = useRef<Vector3 | null>(null);
  const goalTarget = useRef<Vector3 | null>(null);

  useEffect(() => {
    const elevation = (ELEVATION_DEG * Math.PI) / 180;

    let target = new Vector3(0, 0, 0);
    let radius: number;

    if (viewRequest.view === "follow" && selectedBodyId) {
      const body = getBody(selectedBodyId);
      const [x, y, z] = scalePosition(
        eclipticToWorld(bodyState(body.id, clock.jd).position),
        distanceMode,
      );
      target = new Vector3(x, y, z);
      // Close enough to see the sphere, far enough not to clip into it.
      radius = Math.max(scaleDistance(0.06, distanceMode), 1.2);
    } else {
      const band = viewRequest.view === "follow" ? "all" : viewRequest.view;
      radius = scaleDistance(BAND_EDGE_AU[band], distanceMode);
    }

    // Fit a tilted *disc*, not a sphere.
    //
    // The solar system is very nearly flat, so seen from `ELEVATION_DEG` above the
    // plane it projects to an ellipse: full width `radius`, but only
    // `radius * sin(elevation)` tall. Fitting a sphere of that radius — the naive
    // choice — pushes the camera roughly twice as far as needed and leaves the
    // model marooned in the middle of an empty frame.
    //
    // Width and height are checked against their own frustum limits and the
    // binding one wins. On a typical landscape window the width binds; on a
    // portrait phone the height does.
    const halfFovTangent = Math.tan((FOV_DEG * Math.PI) / 360);
    const aspect = "aspect" in camera ? camera.aspect : 1;

    const halfWidth = radius;
    // The vertical extent is the foreshortened disc plus a little slack for the
    // genuinely out-of-plane bodies, Pluto above all.
    const halfHeight = radius * Math.max(Math.sin(elevation), 0.34);

    const distanceForWidth = halfWidth / (halfFovTangent * aspect);
    const distanceForHeight = halfHeight / halfFovTangent;
    const distance = FIT_MARGIN * Math.max(distanceForWidth, distanceForHeight);

    goalTarget.current = target;
    goalPosition.current = new Vector3(
      target.x,
      target.y + distance * Math.sin(elevation),
      target.z + distance * Math.cos(elevation),
    );
    // `clock` and `camera` are mutable objects, intentionally not dependencies:
    // this should fire on an explicit request, not on every tick of time.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewRequest, distanceMode, selectedBodyId]);

  useFrame(() => {
    const position = goalPosition.current;
    const target = goalTarget.current;
    if (!position || !target) return;

    camera.position.lerp(position, EASING);
    if (controls) {
      controls.target.lerp(target, EASING);
      controls.update();
    }

    // Stop steering once the camera is close enough that further easing would be
    // invisible; otherwise this would fight the user's own dragging forever.
    if (camera.position.distanceTo(position) < position.length() * 0.002) {
      goalPosition.current = null;
      goalTarget.current = null;
    }
  });

  return null;
}
