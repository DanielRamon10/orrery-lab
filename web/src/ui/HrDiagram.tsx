/**
 * The Hertzsprung–Russell diagram, drawn from the same rows the 3D scene renders.
 *
 * Colour against intrinsic brightness — the plot that revealed how stars live and die.
 * Because it and the star field come from the same 9,000 Gaia sources, a selection here
 * is a selection *there*: drag a box around the red giants and the red giants light up
 * in space, still where they actually are.
 *
 * That link is the whole reason the panel exists. An HR diagram on its own is a
 * textbook figure; an HR diagram wired to the positions turns "these stars are a
 * distinct population" into something you can see the shape of.
 *
 * Drawn to a `<canvas>` rather than as SVG: 9,000 marks is far past the point where one
 * DOM node each is sensible, and the plot is a density, not a set of interactive
 * elements.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import type { StarCatalogue } from "../lib/starCatalogue";
import { useSimulation, type StarSelection } from "../state/simulation";

/** Axis bounds. Generous enough to hold the whole sample without clipping. */
const COLOUR_RANGE = [-0.4, 3.0] as const;
const MAGNITUDE_RANGE = [-6, 10] as const;

const WIDTH = 258;
const HEIGHT = 210;
const PADDING = { left: 30, right: 8, top: 8, bottom: 26 };

interface Box {
  readonly x0: number;
  readonly y0: number;
  readonly x1: number;
  readonly y1: number;
}

function toPixel(colour: number, magnitude: number): [number, number] {
  const plotWidth = WIDTH - PADDING.left - PADDING.right;
  const plotHeight = HEIGHT - PADDING.top - PADDING.bottom;

  const x =
    PADDING.left +
    ((colour - COLOUR_RANGE[0]) / (COLOUR_RANGE[1] - COLOUR_RANGE[0])) * plotWidth;
  // Magnitudes run backwards — brighter is a smaller number — so the axis is inverted
  // and bright stars sit at the top, as every published HR diagram has them.
  const y =
    PADDING.top +
    ((magnitude - MAGNITUDE_RANGE[0]) / (MAGNITUDE_RANGE[1] - MAGNITUDE_RANGE[0])) *
      plotHeight;
  return [x, y];
}

function fromPixel(x: number, y: number): [number, number] {
  const plotWidth = WIDTH - PADDING.left - PADDING.right;
  const plotHeight = HEIGHT - PADDING.top - PADDING.bottom;

  const colour =
    COLOUR_RANGE[0] +
    ((x - PADDING.left) / plotWidth) * (COLOUR_RANGE[1] - COLOUR_RANGE[0]);
  const magnitude =
    MAGNITUDE_RANGE[0] +
    ((y - PADDING.top) / plotHeight) * (MAGNITUDE_RANGE[1] - MAGNITUDE_RANGE[0]);
  return [colour, magnitude];
}

export function HrDiagram({ catalogue }: { catalogue: StarCatalogue | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { starSelection, setStarSelection, starMode } = useSimulation();
  const [dragging, setDragging] = useState<Box | null>(null);

  // Draw the stars once per catalogue; the selection overlay is drawn on top each time
  // it changes, which is cheap because the underlying scatter is cached as an image.
  const scatterRef = useRef<ImageData | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !catalogue) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    const ratio = window.devicePixelRatio || 1;
    canvas.width = WIDTH * ratio;
    canvas.height = HEIGHT * ratio;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, WIDTH, HEIGHT);

    // Axes.
    context.strokeStyle = "rgba(255,255,255,0.18)";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(PADDING.left, PADDING.top);
    context.lineTo(PADDING.left, HEIGHT - PADDING.bottom);
    context.lineTo(WIDTH - PADDING.right, HEIGHT - PADDING.bottom);
    context.stroke();

    // Additive-ish blending so the main sequence's density reads without the
    // individual points needing to be large.
    context.globalCompositeOperation = "lighter";
    for (let index = 0; index < catalogue.count; index += 1) {
      const colour = catalogue.bpRp[index];
      const magnitude = catalogue.absoluteG[index];
      if (colour === null || magnitude === null) continue;

      const [x, y] = toPixel(colour, magnitude);
      if (x < PADDING.left || x > WIDTH - PADDING.right) continue;
      if (y < PADDING.top || y > HEIGHT - PADDING.bottom) continue;

      context.fillStyle = "rgba(120, 170, 240, 0.30)";
      context.fillRect(x, y, 1.6, 1.6);
    }
    context.globalCompositeOperation = "source-over";

    scatterRef.current = context.getImageData(0, 0, canvas.width, canvas.height);
  }, [catalogue]);

  // Selection overlay.
  useEffect(() => {
    const canvas = canvasRef.current;
    const scatter = scatterRef.current;
    if (!canvas || !scatter) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    const ratio = window.devicePixelRatio || 1;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.putImageData(scatter, 0, 0);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);

    const box = dragging ?? boxFromSelection(starSelection);
    if (!box) return;

    context.strokeStyle = "#ffffff";
    context.lineWidth = 1;
    context.setLineDash([3, 3]);
    context.strokeRect(
      Math.min(box.x0, box.x1),
      Math.min(box.y0, box.y1),
      Math.abs(box.x1 - box.x0),
      Math.abs(box.y1 - box.y0),
    );
    context.setLineDash([]);
    context.fillStyle = "rgba(255,255,255,0.07)";
    context.fillRect(
      Math.min(box.x0, box.x1),
      Math.min(box.y0, box.y1),
      Math.abs(box.x1 - box.x0),
      Math.abs(box.y1 - box.y0),
    );
  }, [dragging, starSelection, catalogue]);

  const pointerPosition = useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top] as const;
  }, []);

  if (!catalogue) {
    return (
      <div className="panel panel--hr">
        <h3>Hertzsprung–Russell</h3>
        <p className="hint">Loading the Gaia sample…</p>
      </div>
    );
  }

  return (
    <div className="panel panel--hr">
      <header className="hr-header">
        <h3>Hertzsprung–Russell</h3>
        {starSelection && (
          <button type="button" className="control control--compact" onClick={() => setStarSelection(null)}>
            clear
          </button>
        )}
      </header>

      <canvas
        ref={canvasRef}
        className="hr-canvas"
        style={{ width: WIDTH, height: HEIGHT }}
        onPointerDown={(event) => {
          const [x, y] = pointerPosition(event);
          event.currentTarget.setPointerCapture(event.pointerId);
          setDragging({ x0: x, y0: y, x1: x, y1: y });
        }}
        onPointerMove={(event) => {
          if (!dragging) return;
          const [x, y] = pointerPosition(event);
          setDragging({ ...dragging, x1: x, y1: y });
        }}
        onPointerUp={() => {
          if (!dragging) return;
          const dragged =
            Math.abs(dragging.x1 - dragging.x0) > 4 && Math.abs(dragging.y1 - dragging.y0) > 4;

          if (dragged) {
            const [c0, m0] = fromPixel(dragging.x0, dragging.y0);
            const [c1, m1] = fromPixel(dragging.x1, dragging.y1);
            setStarSelection({
              colour: [Math.min(c0, c1), Math.max(c0, c1)],
              magnitude: [Math.min(m0, m1), Math.max(m0, m1)],
            });
          } else {
            // A click without a drag clears, which is the obvious way to undo.
            setStarSelection(null);
          }
          setDragging(null);
        }}
      />

      <p className="hr-note">
        {starSelection ? (
          <>
            Selected BP−RP {starSelection.colour[0].toFixed(2)}–{starSelection.colour[1].toFixed(2)},
            M<sub>G</sub> {starSelection.magnitude[0].toFixed(1)}–{starSelection.magnitude[1].toFixed(1)}.
            {starMode === "off" && " Turn the star field on to see them."}
          </>
        ) : (
          <>Drag a box to light those stars up in the scene. The diagonal is the main sequence; the cloud above it is red giants.</>
        )}
      </p>
    </div>
  );
}

function boxFromSelection(selection: StarSelection | null): Box | null {
  if (!selection) return null;
  const [x0, y0] = toPixel(selection.colour[0], selection.magnitude[0]);
  const [x1, y1] = toPixel(selection.colour[1], selection.magnitude[1]);
  return { x0, y0, x1, y1 };
}
