/**
 * Owns the simulation state and publishes it through the context.
 *
 * Kept apart from `simulation.ts` so that this file exports a component and that
 * one exports only hooks, types and constants — the split Vite's fast refresh
 * needs to reload either without discarding the other.
 */

import { useCallback, useMemo, useRef, useState, type ReactNode } from "react";

import { julianDateFromDate } from "../lib/ephemeris";
import type { DistanceMode, RadiusMode } from "../lib/scale";
import {
  SimulationContext,
  type SimulationClock,
  type SimulationValue,
  type ViewRequest,
} from "./simulation";

export function SimulationProvider({ children }: { children: ReactNode }) {
  // One object for the whole session; mutated in place, never replaced.
  const clockRef = useRef<SimulationClock>({
    jd: julianDateFromDate(new Date()),
    playing: true,
    daysPerSecond: 7,
  });

  const [distanceMode, setDistanceMode] = useState<DistanceMode>("compressed");
  const [radiusMode, setRadiusMode] = useState<RadiusMode>("readable");
  const [selectedBodyId, setSelectedBodyId] = useState<string | null>("earth");
  const [viewRequest, setViewRequest] = useState<ViewRequest>({ view: "all", nonce: 0 });

  const requestView = useCallback((view: ViewRequest["view"]) => {
    setViewRequest((previous) => ({ view, nonce: previous.nonce + 1 }));
  }, []);

  const seek = useCallback((jd: number) => {
    clockRef.current.jd = jd;
  }, []);

  const seekToNow = useCallback(() => {
    clockRef.current.jd = julianDateFromDate(new Date());
  }, []);

  const setPlaying = useCallback((playing: boolean) => {
    clockRef.current.playing = playing;
  }, []);

  const setDaysPerSecond = useCallback((daysPerSecond: number) => {
    clockRef.current.daysPerSecond = daysPerSecond;
  }, []);

  const value = useMemo<SimulationValue>(
    () => ({
      clock: clockRef.current,
      distanceMode,
      radiusMode,
      selectedBodyId,
      viewRequest,
      requestView,
      setDistanceMode,
      setRadiusMode,
      setSelectedBodyId,
      seek,
      seekToNow,
      setPlaying,
      setDaysPerSecond,
    }),
    [
      distanceMode,
      radiusMode,
      selectedBodyId,
      viewRequest,
      requestView,
      seek,
      seekToNow,
      setPlaying,
      setDaysPerSecond,
    ],
  );

  return <SimulationContext.Provider value={value}>{children}</SimulationContext.Provider>;
}
