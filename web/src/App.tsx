/**
 * Page shell: the canvas fills the viewport, the panels float over it.
 */

import { Canvas } from "@react-three/fiber";
import { Suspense } from "react";

import { Orrery } from "./scene/Orrery";
import { SimulationProvider } from "./state/SimulationProvider";
import { Readout } from "./ui/Readout";
import { ScaleControls } from "./ui/ScaleControls";
import { TimeControls } from "./ui/TimeControls";

export default function App() {
  return (
    <SimulationProvider>
      <div className="app">
        <Canvas
          // Roughly the "All" framing, so the first painted frame is already in
          // the right place; CameraDirector then eases it to the exact fit. The
          // 45-degree fov here must match FOV_DEG in CameraDirector.
          camera={{ position: [0, 108, 204], fov: 45, near: 0.01, far: 8000 }}
          dpr={[1, 2]}
          gl={{ antialias: true }}
        >
          <Suspense fallback={null}>
            <Orrery />
          </Suspense>
        </Canvas>

        <header className="masthead">
          <h1>orrery-lab</h1>
          <p>
            Positions solved from JPL orbital elements — Kepler&rsquo;s equation, not a
            canned animation.
          </p>
        </header>

        <div className="dock dock--right">
          <ScaleControls />
          <Readout />
        </div>

        <div className="dock dock--bottom">
          <TimeControls />
        </div>

        <footer className="credit">
          Drag to orbit · scroll to zoom · click a body to inspect
        </footer>
      </div>
    </SimulationProvider>
  );
}
