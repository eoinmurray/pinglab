import { useMemo, useRef, useState } from "react";

type Pt = { x: number; y: number };

const W = 360;
const H = 240;
const PITCH = 30;
const SCALE = 75;
const CELL_W = PITCH / SCALE;

// A vector field with an explicit source and sink. Near the source the
// field radiates outward (positive divergence); near the sink it
// converges (negative divergence). The same cell-aligned positions as
// the gradient demo so the two are visually comparable.
const SIGMA2 = 0.6;
const SRC = { x: -2 * CELL_W, y: -1 * CELL_W };   // (-0.80, -0.40)
const SNK = { x:  2 * CELL_W, y:  1 * CELL_W };   // ( 0.80,  0.40)

// F(x,y) = exp(-d²_src/σ²) (r - src) − exp(-d²_snk/σ²) (r - snk)
function field(x: number, y: number): Pt {
  const dsx = x - SRC.x;
  const dsy = y - SRC.y;
  const dnx = x - SNK.x;
  const dny = y - SNK.y;
  const es = Math.exp(-(dsx * dsx + dsy * dsy) / SIGMA2);
  const en = Math.exp(-(dnx * dnx + dny * dny) / SIGMA2);
  return {
    x: es * dsx - en * dnx,
    y: es * dsy - en * dny,
  };
}

// Closed form: ∇·F = 2·exp(-d²/σ²)·(1 - d²/σ²) for each term.
function divergence(x: number, y: number): number {
  const dsx = x - SRC.x;
  const dsy = y - SRC.y;
  const dnx = x - SNK.x;
  const dny = y - SNK.y;
  const r2s = (dsx * dsx + dsy * dsy) / SIGMA2;
  const r2n = (dnx * dnx + dny * dny) / SIGMA2;
  const es = Math.exp(-r2s);
  const en = Math.exp(-r2n);
  return 2 * es * (1 - r2s) - 2 * en * (1 - r2n);
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

export default function Divergence() {
  const cx = W / 2;
  const cy = H / 2;
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [p, setP] = useState<Pt>({ x: 0, y: 0 });
  const dragRef = useRef(false);

  const toScreen = (v: Pt) => ({ x: cx + v.x * SCALE, y: cy - v.y * SCALE });

  // Quiver: same grid layout as Gradient — cells centred on (cx, cy),
  // skipping the two cells that coincide with the source / sink markers.
  const { quiver, fmax } = useMemo(() => {
    const halfNX = Math.floor((W / 2) / PITCH);
    const halfNY = Math.floor((H / 2) / PITCH);
    const skip = new Set([`${-2},${1}`, `${2},${-1}`]);
    const pts: { sx: number; sy: number; fx: number; fy: number }[] = [];
    let fmax = 0;
    for (let jy = -halfNY; jy <= halfNY; jy++) {
      for (let ix = -halfNX; ix <= halfNX; ix++) {
        if (skip.has(`${ix},${jy}`)) continue;
        const sx = cx + ix * PITCH;
        const sy = cy + jy * PITCH;
        const xw = (sx - cx) / SCALE;
        const yw = -(sy - cy) / SCALE;
        const F = field(xw, yw);
        const m = Math.hypot(F.x, F.y);
        if (m > fmax) fmax = m;
        pts.push({ sx, sy, fx: F.x, fy: F.y });
      }
    }
    const MIN_FRAC = 0.32;
    const MAX_FRAC = 0.78;
    const quiver = pts.map(({ sx, sy, fx, fy }) => {
      const m = Math.hypot(fx, fy);
      const t = m / (fmax || 1);
      const arrowLen = (MIN_FRAC + (MAX_FRAC - MIN_FRAC) * Math.sqrt(t)) * PITCH;
      const ux = m > 1e-6 ? fx / m : 0;
      const uy = m > 1e-6 ? fy / m : 0;
      const arrow = arrowPath(
        sx - ux * arrowLen * 0.5,
        sy + uy * arrowLen * 0.5,
        sx + ux * arrowLen * 0.5,
        sy - uy * arrowLen * 0.5,
        4,
      );
      const op = 0.18 + 0.62 * Math.sqrt(t);
      return arrow ? { ...arrow, op } : null;
    });
    return { quiver, fmax };
  }, []);

  const srcS = toScreen(SRC);
  const snkS = toScreen(SNK);

  const pS = toScreen(p);
  const div = divergence(p.x, p.y);

  // Probe rings: inner ring is the imagined test volume. A second ring
  // sits OUTSIDE for positive divergence (the volume is being pushed
  // outward, net outflow) and INSIDE for negative divergence (the
  // volume is being squeezed inward, net inflow). Distance from the
  // inner ring scales with |∇·F|.
  const ringR = 18;
  const offset = Math.min(14, Math.abs(div) * 7);
  const outRingR = div >= 0 ? ringR + offset : ringR - offset;
  const divPos = div > 0;

  const handlePointer = (e: React.PointerEvent) => {
    if (!dragRef.current || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const vbX = ((e.clientX - rect.left) / rect.width) * W;
    const vbY = ((e.clientY - rect.top) / rect.height) * H;
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
        aria-label="Divergence demo"
      >
        <rect x={0.5} y={0.5} width={W - 1} height={H - 1}
              fill="none" stroke="currentColor" strokeWidth={1} opacity={0.3} />

        <line x1={0} y1={cy} x2={W} y2={cy} stroke="currentColor" strokeWidth={0.5} opacity={0.2} />
        <line x1={cx} y1={0} x2={cx} y2={H} stroke="currentColor" strokeWidth={0.5} opacity={0.2} />

        <g opacity={0.85}>
          <circle cx={srcS.x} cy={srcS.y} r={3.5} fill="currentColor" />
          <text x={srcS.x + 8} y={srcS.y - 6}
                fontSize={13} fontFamily="ui-monospace, monospace" fontWeight={600}
                fill="currentColor">+</text>
          <circle cx={snkS.x} cy={snkS.y} r={3.5} fill="none" stroke="currentColor" strokeWidth={1.4} />
          <text x={snkS.x + 8} y={snkS.y - 6}
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

        {/* Probe: inner ring marks the test volume; outer ring grows
            with |div|, dashed outward if positive, dashed inward if
            negative (drawn the same — sign is read from the readout). */}
        <circle cx={pS.x} cy={pS.y} r={ringR}
                fill="none" stroke="currentColor" strokeWidth={1.4} opacity={0.85} />
        {Math.abs(div) > 0.02 && outRingR > 2 && (
          <circle cx={pS.x} cy={pS.y} r={outRingR}
                  fill="none" stroke="currentColor"
                  strokeWidth={1.4}
                  strokeDasharray={divPos ? "5 4" : "2 3"}
                  opacity={0.65} />
        )}

        <circle cx={pS.x} cy={pS.y} r={4}
                fill="currentColor" fillOpacity={0.85}
                onPointerDown={(e) => { (e.target as Element).setPointerCapture(e.pointerId); dragRef.current = true; }}
                style={{ cursor: "grab" }} />

        <text x={W - 8} y={H - 8} textAnchor="end"
              fontSize={11} fontFamily="ui-monospace, monospace"
              fill="currentColor" opacity={0.85}>
          ∇·F = {div.toFixed(2)}
        </text>
      </svg>
      <figcaption style={{ fontSize: "0.85rem", opacity: 0.7, marginTop: 6 }}>
        Arrows show a vector field <em>F</em> with a source (filled dot, <em>+</em>) and a sink
        (hollow dot, <em>−</em>) — flow radiates out of one and converges into the other. Drag the
        probe to read ∇·<em>F</em>. The inner ring is the imagined test volume; the dashed ring sits
        <em>outside</em> it when divergence is positive (the volume is being pushed outward), and
        <em>inside</em> it when negative (the volume is being squeezed inward). The gap between
        the two rings scales with |∇·<em>F</em>|. Halfway between source and sink — and far from
        both — divergence is zero and only the inner ring is drawn.
      </figcaption>
    </figure>
  );
}
