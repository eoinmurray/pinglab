import { useMemo, useRef, useState } from "react";

type Pt = { x: number; y: number };

const W = 360;
const H = 240;
// Quiver grid extends to the full SVG extent, not the unit-bump extent.
// World half-extent matches the larger axis so arrows fill the whole frame.
const SIGMA2 = 0.6;
// Peak / valley positions are pinned to integer cell offsets from the
// origin in the quiver grid below, so the +/− markers land exactly on a
// cell centre — no off-by-half-a-cell drift.
const PITCH = 30;                       // px between quiver arrow centres
const SCALE = 75;                       // px per world unit
const CELL_W = PITCH / SCALE;           // world units per cell
const PEAK = { x: -2 * CELL_W, y: -1 * CELL_W };   // (-0.80, -0.40)
const VALL = { x:  2 * CELL_W, y:  1 * CELL_W };   // ( 0.80,  0.40)

function f(x: number, y: number): number {
  const dpx = x - PEAK.x;
  const dpy = y - PEAK.y;
  const dvx = x - VALL.x;
  const dvy = y - VALL.y;
  return (
    Math.exp(-(dpx * dpx + dpy * dpy) / SIGMA2)
    - Math.exp(-(dvx * dvx + dvy * dvy) / SIGMA2)
  );
}

function grad(x: number, y: number): Pt {
  const dpx = x - PEAK.x;
  const dpy = y - PEAK.y;
  const dvx = x - VALL.x;
  const dvy = y - VALL.y;
  const ep = Math.exp(-(dpx * dpx + dpy * dpy) / SIGMA2);
  const ev = Math.exp(-(dvx * dvx + dvy * dvy) / SIGMA2);
  return {
    x: (-2 * dpx * ep + 2 * dvx * ev) / SIGMA2,
    y: (-2 * dpy * ep + 2 * dvy * ev) / SIGMA2,
  };
}

function arrowPath(x1: number, y1: number, x2: number, y2: number, head: number) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy);
  if (len < 1) return null;
  const ux = dx / len;
  const uy = dy / len;
  const bx = x2 - ux * head;
  const by = y2 - uy * head;
  const px = -uy;
  const py = ux;
  const h = head * 0.55;
  return {
    shaft: { x1, y1, x2: bx, y2: by },
    head: `${x2},${y2} ${bx + px * h},${by + py * h} ${bx - px * h},${by - py * h}`,
  };
}

const HALF_X = (W / 2) / SCALE;
const HALF_Y = (H / 2) / SCALE;

export default function Gradient() {
  const cx = W / 2;
  const cy = H / 2;
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [p, setP] = useState<Pt>({ x: 0.2, y: -0.6 });
  const dragRef = useRef(false);

  const toScreen = (v: Pt) => ({ x: cx + v.x * SCALE, y: cy - v.y * SCALE });

  // Quiver: spans the entire visible canvas with a fixed screen-pixel pitch.
  // Grid is centred on (cx, cy) so the origin sits on a cell, and peak /
  // valley land on cell centres at integer offsets — markers align with
  // arrows cleanly. Every cell renders a direction arrow regardless of
  // magnitude; length is sqrt-scaled with a floor and opacity tracks
  // |∇f| so flat regions dim out instead of vanishing.
  const { quiver, gmax } = useMemo(() => {
    const halfNX = Math.floor((W / 2) / PITCH);
    const halfNY = Math.floor((H / 2) / PITCH);
    // Cells that coincide with the peak / valley markers — skip the
    // arrows there so the dots stand alone (∇f = 0 at those points
    // anyway, so we'd only be drawing the minimum-length floor arrow).
    const skipCells = new Set([
      `${-2},${1}`,   // peak: world (-0.80, -0.40)
      `${ 2},${-1}`,  // valley: world ( 0.80,  0.40)
    ]);
    const pts: { sx: number; sy: number; gx: number; gy: number }[] = [];
    let gmax = 0;
    for (let jy = -halfNY; jy <= halfNY; jy++) {
      for (let ix = -halfNX; ix <= halfNX; ix++) {
        if (skipCells.has(`${ix},${jy}`)) continue;
        const sx = cx + ix * PITCH;
        const sy = cy + jy * PITCH;
        const xw = (sx - cx) / SCALE;
        const yw = -(sy - cy) / SCALE;
        const g = grad(xw, yw);
        const m = Math.hypot(g.x, g.y);
        if (m > gmax) gmax = m;
        pts.push({ sx, sy, gx: g.x, gy: g.y });
      }
    }
    const cellSize = PITCH;
    const MIN_FRAC = 0.32;               // shortest arrow as a fraction of cellSize
    const MAX_FRAC = 0.78;               // longest arrow as a fraction of cellSize
    const quiver = pts.map(({ sx, sy, gx, gy }) => {
      const m = Math.hypot(gx, gy);
      const t = m / (gmax || 1);
      const arrowLen = (MIN_FRAC + (MAX_FRAC - MIN_FRAC) * Math.sqrt(t)) * cellSize;
      // Direction: at very small magnitudes, the numerical direction is
      // unstable, so fall back to a tiny default rather than randomising.
      const ux = m > 1e-6 ? gx / m : 0;
      const uy = m > 1e-6 ? gy / m : 0;
      const arrow = arrowPath(
        sx - ux * arrowLen * 0.5,
        sy + uy * arrowLen * 0.5,
        sx + ux * arrowLen * 0.5,
        sy - uy * arrowLen * 0.5,
        4,
      );
      // Opacity: faint where the gradient is flat, full where steep.
      const op = 0.18 + 0.62 * Math.sqrt(t);
      return arrow ? { ...arrow, op } : null;
    });
    return { quiver, gmax };
  }, []);

  const peakS = toScreen(PEAK);
  const vallS = toScreen(VALL);

  const pS = toScreen(p);
  const gp = grad(p.x, p.y);
  const gpMag = Math.hypot(gp.x, gp.y);
  const showLen = Math.min(80, (gpMag / (gmax || 1)) * 100);
  const ux = gpMag > 0 ? gp.x / gpMag : 0;
  const uy = gpMag > 0 ? gp.y / gpMag : 0;
  const big = arrowPath(pS.x, pS.y, pS.x + ux * showLen, pS.y - uy * showLen, 9);

  const handlePointer = (e: React.PointerEvent) => {
    if (!dragRef.current || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const vbX = ((e.clientX - rect.left) / rect.width) * W;
    const vbY = ((e.clientY - rect.top) / rect.height) * H;
    // Clamp probe inside the visible world to keep the handle on-canvas.
    const wx = Math.max(-HALF_X, Math.min(HALF_X, (vbX - cx) / SCALE));
    const wy = Math.max(-HALF_Y, Math.min(HALF_Y, -(vbY - cy) / SCALE));
    setP({ x: wx, y: wy });
  };

  return (
    <figure style={{ width: "100%", margin: "0 0 16px 0" }}>
      <svg
        ref={svgRef}
        width={W}
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        onPointerMove={handlePointer}
        onPointerUp={() => { dragRef.current = false; }}
        onPointerLeave={() => { dragRef.current = false; }}
        style={{ touchAction: "none", display: "block", width: "100%", height: "auto", maxWidth: 720 }}
        role="img"
        aria-label="Gradient demo"
      >
        <rect x={0.5} y={0.5} width={W - 1} height={H - 1}
              fill="none" stroke="currentColor" strokeWidth={1} opacity={0.3} />

        <line x1={0} y1={cy} x2={W} y2={cy} stroke="currentColor" strokeWidth={0.5} opacity={0.2} />
        <line x1={cx} y1={0} x2={cx} y2={H} stroke="currentColor" strokeWidth={0.5} opacity={0.2} />

        <g opacity={0.85}>
          <circle cx={peakS.x} cy={peakS.y} r={3.5} fill="currentColor" />
          <text x={peakS.x + 8} y={peakS.y - 6}
                fontSize={13} fontFamily="ui-monospace, monospace" fontWeight={600}
                fill="currentColor">+</text>
          <circle cx={vallS.x} cy={vallS.y} r={3.5} fill="none" stroke="currentColor" strokeWidth={1.4} />
          <text x={vallS.x + 8} y={vallS.y - 6}
                fontSize={13} fontFamily="ui-monospace, monospace" fontWeight={600}
                fill="currentColor">−</text>
        </g>

        {quiver.map((q, i) => q && (
          <g key={i} opacity={q.op}>
            <line x1={q.shaft.x1} y1={q.shaft.y1} x2={q.shaft.x2} y2={q.shaft.y2}
                  stroke="currentColor" strokeWidth={1} strokeLinecap="round" />
            <polygon points={q.head} fill="currentColor" />
          </g>
        ))}

        {big && (
          <g>
            <line x1={big.shaft.x1} y1={big.shaft.y1} x2={big.shaft.x2} y2={big.shaft.y2}
                  stroke="currentColor" strokeWidth={2} strokeLinecap="round" />
            <polygon points={big.head} fill="currentColor" />
          </g>
        )}

        <circle cx={pS.x} cy={pS.y} r={9}
                fill="currentColor" fillOpacity={0.1} stroke="currentColor" strokeWidth={1.5}
                onPointerDown={(e) => { (e.target as Element).setPointerCapture(e.pointerId); dragRef.current = true; }}
                style={{ cursor: "grab" }} />

        <text x={W - 8} y={H - 8} textAnchor="end"
              fontSize={11} fontFamily="ui-monospace, monospace"
              fill="currentColor" opacity={0.85}>
          f = {f(p.x, p.y).toFixed(2)}    |∇f| = {gpMag.toFixed(2)}
        </text>
      </svg>
      <figcaption style={{ fontSize: "0.85rem", opacity: 0.7, marginTop: 6 }}>
        Small arrows are ∇<em>f</em> sampled on a grid covering the whole frame, for a two-bump
        scalar field with a peak (filled dot, <em>+</em>) and a valley (hollow dot, <em>−</em>).
        Every arrow points the way <em>f</em> increases fastest at that point. Drag the handle to
        place a probe; the big arrow is ∇<em>f</em> there, and the readout shows <em>f</em> and
        |∇<em>f</em>|. ∇<em>f</em> always points uphill — toward the peak, away from the valley —
        and shrinks to zero at the extrema themselves.
      </figcaption>
    </figure>
  );
}
