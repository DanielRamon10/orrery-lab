/**
 * Loading the Gaia star catalogue, once, for everyone who needs it.
 *
 * Two places consume these rows: the 3D star field and the Hertzsprung–Russell panel.
 * They need the **same** rows — a box dragged on the diagram only highlights the right
 * stars in space if the indices line up — so the fetch is memoised at module scope and
 * the second consumer joins the first request instead of issuing its own.
 *
 * Fetched rather than imported. A 700 kB JSON `import` would be inlined into the main
 * bundle by Vite and block first paint; fetching keeps it a separate cacheable asset
 * and lets the solar system render immediately while the sky arrives a moment later.
 *
 * Lives apart from the components so that neither file mixes component and
 * non-component exports, which breaks Vite's fast refresh.
 */

import { useEffect, useState } from "react";

import starsUrl from "../data/stars.generated.json?url";

export interface StarCatalogue {
  readonly count: number;
  /** Unit vectors in equatorial J2000 — the direction to each star from here. */
  readonly direction: number[][];
  /** Apparent Gaia G magnitude. Smaller is brighter. */
  readonly magnitude: number[];
  /** Approximate display RGB, in `[0, 1]`. */
  readonly colour: number[][];
  /**
   * 3D position in parsecs, Sun at the origin, galactic axes.
   *
   * `null` where the parallax was too noisy to invert. Those stars are deliberately
   * absent rather than guessed at — see `orrery/gaia.py` for why inverting a marginal
   * parallax is worse than admitting you cannot.
   */
  readonly galactic: (number[] | null)[];
  /** Colour index BP−RP: the HR diagram's x-axis. */
  readonly bpRp: (number | null)[];
  /** Absolute magnitude: the HR diagram's y-axis. Needs a distance, so often null. */
  readonly absoluteG: (number | null)[];
}

let cataloguePromise: Promise<StarCatalogue> | null = null;

function loadCatalogue(): Promise<StarCatalogue> {
  cataloguePromise ??= fetch(starsUrl).then((response) => {
    if (!response.ok) throw new Error(`star catalogue: HTTP ${response.status}`);
    return response.json() as Promise<StarCatalogue>;
  });
  return cataloguePromise;
}

/** The catalogue, or `null` until it arrives. */
export function useStarCatalogue(): StarCatalogue | null {
  const [catalogue, setCatalogue] = useState<StarCatalogue | null>(null);

  useEffect(() => {
    let cancelled = false;

    loadCatalogue()
      .then((data) => {
        if (!cancelled) setCatalogue(data);
      })
      .catch((error) => {
        // A missing star field is a degraded scene, not a broken one: the solar system
        // is the subject and it renders perfectly well without a sky.
        console.warn("could not load the Gaia star field:", error);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return catalogue;
}
