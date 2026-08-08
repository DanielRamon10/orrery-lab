import react from "@vitejs/plugin-react";
// From "vitest/config", not "vite": it is the same defineConfig with the `test`
// key added to the type, so the block below type-checks under `tsc -b`.
import { defineConfig } from "vitest/config";

/**
 * `base` has to match the GitHub Pages sub-path, because the site is served from
 * `https://<user>.github.io/<repo>/` rather than from a domain root. Without it
 * every asset URL would resolve one level too high and the page would load blank.
 *
 * It is overridable so a fork under a different repository name still builds:
 *   VITE_BASE=/my-fork/ npm run build
 */
const base = process.env.VITE_BASE ?? "/orrery-lab/";

export default defineConfig({
  base,
  plugins: [react()],
  test: {
    // The suite is pure numerics against the Python reference, so it needs no DOM.
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
