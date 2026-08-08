/**
 * The scene graph: the Sun, the bodies, their orbits, and the lighting.
 *
 * The only thing this component does on the hot path is advance the clock. Every
 * body positions itself from that clock inside its own `useFrame`.
 */

import { OrbitControls } from "@react-three/drei";
import { useFrame } from "@react-three/fiber";
import { useRef } from "react";
import { AdditiveBlending, BackSide, type Mesh } from "three";

import { BODIES, CONSTANTS } from "../data/elements.generated";
import { sunRadius } from "../lib/scale";
import { useSimulation, useSimulationTime } from "../state/simulation";
import { CameraDirector } from "./CameraDirector";
import { OrbitLine } from "./OrbitLine";
import { Planet } from "./Planet";
import { Starfield } from "./Starfield";

/**
 * Advances the simulation clock, once per frame, for the whole scene.
 *
 * Uses the frame's own delta rather than a fixed increment so the simulated rate
 * stays the same on a 60 Hz laptop and a 144 Hz monitor.
 */
function ClockDriver() {
  const { clock } = useSimulation();

  useFrame((_, delta) => {
    if (!clock.playing) return;
    // Guard against the huge delta a backgrounded tab produces on return, which
    // would otherwise teleport the simulation by years in a single frame.
    clock.jd += Math.min(delta, 0.1) * clock.daysPerSecond;
  });

  return null;
}

function Sun() {
  const { radiusMode } = useSimulation();
  const { clock } = useSimulation();
  const meshRef = useRef<Mesh>(null);
  const radius = sunRadius(radiusMode);

  useFrame(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    mesh.rotation.y =
      ((clock.jd % CONSTANTS.SUN_ROTATION_PERIOD_DAYS) / CONSTANTS.SUN_ROTATION_PERIOD_DAYS) *
      Math.PI *
      2;
  });

  return (
    <group>
      {/* The Sun is the only light source, which is what makes the far side of
          each planet fall dark and the outer system look as dim as it is. */}
      <pointLight intensity={520} distance={0} decay={2} color="#fff4e0" />

      <mesh ref={meshRef}>
        <sphereGeometry args={[radius, 48, 32]} />
        <meshBasicMaterial color={CONSTANTS.SUN_COLOUR} />
      </mesh>

      {/* Corona, as three nested shells with **additive** blending.
          Plain transparency made these read as flat grey discs with a visible
          rim: a semi-transparent dark-ish shell over black *darkens* nothing and
          just shows its own silhouette. Additive blending accumulates light
          instead, so the shells sum into a glow that fades out at the edge. */}
      {[
        [1.35, 0.3],
        [1.9, 0.13],
        [3.1, 0.05],
      ].map(([factor, opacity]) => (
        <mesh key={factor}>
          <sphereGeometry args={[radius * factor, 32, 24]} />
          <meshBasicMaterial
            color={CONSTANTS.SUN_COLOUR}
            transparent
            opacity={opacity}
            blending={AdditiveBlending}
            depthWrite={false}
          />
        </mesh>
      ))}
    </group>
  );
}

export function Orrery() {
  const { distanceMode, radiusMode, selectedBodyId, setSelectedBodyId, starMode } =
    useSimulation();
  // Throttled: only the orbit lines need this, and only to pick an epoch bucket.
  const { jd } = useSimulationTime();

  return (
    <>
      <ClockDriver />

      {/* Just enough ambient light to keep night sides from being pure black. */}
      <ambientLight intensity={0.08} />

      {/* Real Gaia DR3 sources, replacing the procedural sky of phase 2. */}
      <Starfield mode={starMode} />

      <Sun />

      {BODIES.map((body) => (
        <OrbitLine
          key={`orbit-${body.id}`}
          bodyId={body.id}
          jd={jd}
          distanceMode={distanceMode}
          colour={body.colour}
          highlighted={body.id === selectedBodyId}
          dashed={!body.isPlanet}
        />
      ))}

      {BODIES.map((body) => (
        <Planet
          key={body.id}
          body={body}
          distanceMode={distanceMode}
          radiusMode={radiusMode}
          selected={body.id === selectedBodyId}
        />
      ))}

      {/* Clicking empty space clears the selection. A huge inside-out sphere is
          the simplest catch-all that still lets ray-casting reach the bodies,
          since they sit in front of it. */}
      <mesh onClick={() => setSelectedBodyId(null)} visible={false}>
        <sphereGeometry args={[2000, 8, 6]} />
        <meshBasicMaterial side={BackSide} />
      </mesh>

      <OrbitControls
        enablePan
        enableDamping
        dampingFactor={0.08}
        minDistance={0.4}
        // Wide enough for linear mode, where Pluto's aphelion alone is ~490 units.
        maxDistance={1600}
        makeDefault
      />

      {/* After OrbitControls, so `useThree(state => state.controls)` is populated. */}
      <CameraDirector />
    </>
  );
}
