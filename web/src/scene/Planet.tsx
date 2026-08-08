/**
 * One body: its sphere, its spin, its label, and its click target.
 *
 * The position is written straight to the three.js object inside `useFrame`, never
 * through React state — see `state/simulation.tsx` for why. That makes this
 * component render only when something a person changed actually changes.
 */

import { Html } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { useMemo, useRef, useState } from "react";
import type { Group, Mesh } from "three";

import type { BodyElements } from "../data/elements.generated";
import { bodyState, eclipticToWorld } from "../lib/ephemeris";
import { scalePosition, scaleRadius, type DistanceMode, type RadiusMode } from "../lib/scale";
import { useSimulation } from "../state/simulation";

const DEG_TO_RAD = Math.PI / 180;

/**
 * Minimum orbit-radius-to-viewing-distance ratio for a label to appear.
 *
 * 0.085 is tuned so that framing the whole system leaves Jupiter outward labelled
 * and the inner four unlabelled, which is exactly where the pile-up was.
 */
const LABEL_VISIBILITY_RATIO = 0.085;

interface PlanetProps {
  readonly body: BodyElements;
  readonly distanceMode: DistanceMode;
  readonly radiusMode: RadiusMode;
  readonly selected: boolean;
}

export function Planet({ body, distanceMode, radiusMode, selected }: PlanetProps) {
  const groupRef = useRef<Group>(null);
  const meshRef = useRef<Mesh>(null);
  const { clock, setSelectedBodyId } = useSimulation();
  const camera = useThree((state) => state.camera);

  /**
   * Labels declutter themselves as you pull back.
   *
   * Zoomed out to the whole system, the four inner planets are a knot a few pixels
   * wide and their labels pile into an unreadable stack. So a label only shows once
   * its orbit is a large enough fraction of the viewing distance to be visually
   * separate — which means the giants stay labelled from far away and the inner
   * planets earn their labels as you zoom in. The selected body is always labelled,
   * because that one was asked for explicitly.
   */
  const [labelVisible, setLabelVisible] = useState(true);

  const radius = scaleRadius(body.radiusKm, radiusMode);

  // In true-scale mode the spheres are sub-pixel, so the click target has to be a
  // separate invisible sphere or the body becomes impossible to select.
  const hitRadius = useMemo(() => Math.max(radius, 0.22), [radius]);

  useFrame(() => {
    const group = groupRef.current;
    if (!group) return;

    const { position } = bodyState(body.id, clock.jd);
    const [x, y, z] = scalePosition(eclipticToWorld(position), distanceMode);
    group.position.set(x, y, z);

    // Ratio of this orbit's radius to how far away the viewer is. Compared as a
    // ratio rather than an absolute size so the rule holds in both distance modes,
    // whose scene units differ by a factor of about fifty.
    const orbitRadius = Math.hypot(x, y, z);
    const viewingDistance = Math.max(camera.position.length(), 1e-6);
    const shouldShowLabel = selected || orbitRadius / viewingDistance > LABEL_VISIBILITY_RATIO;

    // Guarded so this only re-renders on the rare frame that crosses the threshold.
    if (shouldShowLabel !== labelVisible) setLabelVisible(shouldShowLabel);

    // Axial spin. Retrograde rotators (Venus, Uranus, Pluto) carry a negative
    // period in the table, which makes this turn the other way for free.
    const mesh = meshRef.current;
    if (mesh && body.rotationPeriodDays !== 0) {
      mesh.rotation.y = ((clock.jd % body.rotationPeriodDays) / body.rotationPeriodDays) * Math.PI * 2;
    }
  });

  return (
    <group ref={groupRef}>
      <mesh
        ref={meshRef}
        // Axial tilt, applied about the x-axis before the spin about y.
        rotation={[body.axialTiltDeg * DEG_TO_RAD, 0, 0]}
      >
        <sphereGeometry args={[radius, 32, 24]} />
        <meshStandardMaterial
          color={body.colour}
          roughness={0.85}
          metalness={0.05}
          // A touch of self-illumination so the night side is not pure black at
          // Neptune's distance, where the Sun delivers almost no light.
          emissive={body.colour}
          emissiveIntensity={0.12}
        />
      </mesh>

      {/* Invisible, larger click target. */}
      <mesh
        onClick={(event) => {
          event.stopPropagation();
          setSelectedBodyId(body.id);
        }}
        onPointerOver={() => {
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={() => {
          document.body.style.cursor = "auto";
        }}
      >
        <sphereGeometry args={[hitRadius, 12, 8]} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} />
      </mesh>

      {selected && (
        <mesh>
          <sphereGeometry args={[hitRadius * 1.35, 24, 16]} />
          <meshBasicMaterial color="#ffffff" transparent opacity={0.1} depthWrite={false} />
        </mesh>
      )}

      {labelVisible && (
        <Html
          // No distanceFactor: labels behave like a HUD and stay legible whether
          // you are parked next to Mercury or looking in from beyond Neptune.
          position={[0, hitRadius + 0.12, 0]}
          center
          style={{ pointerEvents: "none", userSelect: "none" }}
        >
          <span className={`body-label${selected ? " body-label--selected" : ""}`}>
            {body.name}
          </span>
        </Html>
      )}
    </group>
  );
}
