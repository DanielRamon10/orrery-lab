/**
 * Shared simulation state: the clock, the scale modes, the selection, the camera.
 *
 * Why the clock is not React state
 * --------------------------------
 * The scene advances every animation frame. If the current date lived in
 * `useState`, every frame would re-render the whole component tree sixty times a
 * second just to move nine spheres — React would become the bottleneck rather
 * than the GPU.
 *
 * So the clock lives in a **mutable ref**. `useFrame` handlers read it directly and
 * write straight to the three.js objects, bypassing React entirely on the hot path.
 * The UI, which only needs to show a date and a speed, subscribes through
 * {@link useSimulationTime}, which re-renders at a fixed low rate instead.
 *
 * The scale modes, the selection and the camera requests *are* ordinary state: they
 * change when a person clicks something, which is many orders of magnitude slower
 * than a frame.
 *
 * This module holds only types, constants, the context object and hooks. The
 * provider component lives in `SimulationProvider.tsx` so that neither file mixes
 * component and non-component exports, which would break Vite's fast refresh.
 */

import { createContext, useContext, useEffect, useState } from "react";

import type { StarMode } from "../scene/Starfield";
import type { DistanceMode, RadiusMode } from "../lib/scale";

/** How often the date read-out refreshes, in milliseconds. */
const UI_REFRESH_INTERVAL_MS = 100;

/** Preset speeds, in simulated days per real second. */
export const SPEED_PRESETS: readonly { readonly label: string; readonly daysPerSecond: number }[] =
  [
    { label: "1 h/s", daysPerSecond: 1 / 24 },
    { label: "1 d/s", daysPerSecond: 1 },
    { label: "1 wk/s", daysPerSecond: 7 },
    { label: "1 mo/s", daysPerSecond: 30.44 },
    { label: "1 yr/s", daysPerSecond: 365.25 },
    { label: "10 yr/s", daysPerSecond: 3652.5 },
  ];

/** The mutable clock. Written by the animation loop, read by everything. */
export interface SimulationClock {
  /** Current Julian Date. */
  jd: number;
  playing: boolean;
  /** Simulated days per real second; negative runs time backwards. */
  daysPerSecond: number;
}

/**
 * A framing request from the UI to the camera.
 *
 * `inner` / `outer` / `all` fit a distance band; `follow` re-centres on whatever
 * body is selected. Carries a monotonic `nonce` so that asking for the same view
 * twice still triggers a move — without it, clicking "All" after zooming in by
 * hand would be a no-op.
 */
export interface ViewRequest {
  readonly view: "inner" | "outer" | "all" | "follow";
  readonly nonce: number;
}

/**
 * A box dragged on the Hertzsprung-Russell diagram.
 *
 * Both bounds are inclusive. Stars inside it are highlighted in the 3D scene and the
 * rest are dimmed rather than hidden, so a population is seen in its context.
 */
export interface StarSelection {
  /** Colour index BP-RP, `[low, high]`. */
  readonly colour: readonly [number, number];
  /** Absolute magnitude, `[bright, faint]` — remember the scale runs backwards. */
  readonly magnitude: readonly [number, number];
}

export interface SimulationValue {
  readonly clock: SimulationClock;
  readonly distanceMode: DistanceMode;
  readonly radiusMode: RadiusMode;
  readonly selectedBodyId: string | null;
  readonly viewRequest: ViewRequest;
  readonly requestView: (view: ViewRequest["view"]) => void;
  /** How the Gaia star field is drawn, or whether it is drawn at all. */
  readonly starMode: StarMode;
  readonly setStarMode: (mode: StarMode) => void;
  /** The region of the HR diagram currently highlighted, if any. */
  readonly starSelection: StarSelection | null;
  readonly setStarSelection: (selection: StarSelection | null) => void;
  readonly setDistanceMode: (mode: DistanceMode) => void;
  readonly setRadiusMode: (mode: RadiusMode) => void;
  readonly setSelectedBodyId: (id: string | null) => void;
  /** Jump the clock to a Julian Date. */
  readonly seek: (jd: number) => void;
  /** Jump back to the present moment. */
  readonly seekToNow: () => void;
  readonly setPlaying: (playing: boolean) => void;
  readonly setDaysPerSecond: (daysPerSecond: number) => void;
}

export const SimulationContext = createContext<SimulationValue | null>(null);

export function useSimulation(): SimulationValue {
  const value = useContext(SimulationContext);
  if (!value) throw new Error("useSimulation must be used inside a SimulationProvider");
  return value;
}

/**
 * Subscribe to the clock for display purposes.
 *
 * Polls at {@link UI_REFRESH_INTERVAL_MS} rather than every frame. Ten updates a
 * second is past the point where a changing date reads as smooth, and it keeps the
 * React tree off the animation loop's critical path.
 */
export function useSimulationTime(): { jd: number; playing: boolean; daysPerSecond: number } {
  const { clock } = useSimulation();
  const [snapshot, setSnapshot] = useState({
    jd: clock.jd,
    playing: clock.playing,
    daysPerSecond: clock.daysPerSecond,
  });

  useEffect(() => {
    const handle = window.setInterval(() => {
      setSnapshot((previous) =>
        previous.jd === clock.jd &&
        previous.playing === clock.playing &&
        previous.daysPerSecond === clock.daysPerSecond
          ? previous // identical snapshot: skip the re-render entirely
          : { jd: clock.jd, playing: clock.playing, daysPerSecond: clock.daysPerSecond },
      );
    }, UI_REFRESH_INTERVAL_MS);

    return () => window.clearInterval(handle);
  }, [clock]);

  return snapshot;
}
