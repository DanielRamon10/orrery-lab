/**
 * The elliptical path a body traces, drawn as a closed line.
 *
 * Recomputed lazily. An orbit's *shape* drifts only through the secular rates in
 * the element table, which move it by a fraction of a degree per century — utterly
 * invisible frame to frame. So the geometry is memoised against a coarse epoch
 * bucket instead of the live clock: scrubbing across centuries redraws the ellipse,
 * while playing forward at normal speed never touches it.
 */

import { Line } from "@react-three/drei";
import { useMemo } from "react";

import { eclipticToWorld, orbitPath } from "../lib/ephemeris";
import { scalePosition, type DistanceMode } from "../lib/scale";

/** Orbits are re-derived once per decade of simulated time. */
const EPOCH_BUCKET_DAYS = 3652.5;

interface OrbitLineProps {
  readonly bodyId: string;
  readonly jd: number;
  readonly distanceMode: DistanceMode;
  readonly colour: string;
  readonly highlighted: boolean;
  /** Pluto is drawn differently, matching the convention in the Python figure. */
  readonly dashed?: boolean;
}

export function OrbitLine({
  bodyId,
  jd,
  distanceMode,
  colour,
  highlighted,
  dashed = false,
}: OrbitLineProps) {
  const epochBucket = Math.round(jd / EPOCH_BUCKET_DAYS);

  const points = useMemo(() => {
    const epoch = epochBucket * EPOCH_BUCKET_DAYS;
    // 512 samples keeps even Mercury's eccentric ellipse visually smooth.
    return orbitPath(bodyId, epoch, 512).map((point) =>
      scalePosition(eclipticToWorld(point), distanceMode),
    );
  }, [bodyId, epochBucket, distanceMode]);

  return (
    <Line
      points={points}
      color={colour}
      lineWidth={highlighted ? 1.8 : 1}
      transparent
      opacity={highlighted ? 0.95 : 0.4}
      dashed={dashed}
      dashSize={0.5}
      gapSize={0.35}
    />
  );
}
