import { useMemo, useRef, useState } from "react";

type Pt = { x: number; y: number };

const W = 360;
const H = 240;
const PITCH = 30;
const SCALE = 75;
const CELL_W = PITCH / SCALE;

// Two vortices: a counter-clockwise (positive curl) one on the left and
// a clockwise (negative curl) one on the right. Same cell-aligned
// positions as the divergence demo so the two are visually comparable.
const SIGMA2 = 0.6;
const CCW = { x: -2 * CELL_W, y: -1 * CELL_W };   // (-0.80, -0.40)
const CW  = { x:  2 * CELL_W, y:  1 * CELL_W };   // ( 0.80,  0.40)

// F(x,y) = CCW vortex - CW vortex (each smeared by a Gaussian).
// CCW vortex around p:  F = (-Δy, Δx) · exp(-d²/σ²)
function field(x: number, y: number): Pt {
  const dax = x - CCW.x;
  const day = y - CCW.y;
  const dbx = x - CW.x;
  const dby = y - CW.y;
  const ea = Math.exp(-(dax * dax + day * day) / SIGMA2);
  const eb = Math.exp(-(dbx * dbx + dby * dby) / SIGMA2);
  return {
    x: ea * (-day) + eb * (dby),
    y: ea * (dax)  + eb * (-dbx),
  };
}

// Closed form: ∇×F (z-component) = 2 e^(-d²/σ²) (1 - d²/σ²) per term,
// with the second term negated because that vortex spins the other way.
function curl(x: number, y: number): number {
  const dax = x - CCW.x;
  const day = y - CCW.y;
  const dbx = x - CW.x;
  const dby = y - CW.y;
  const r2a = (dax * dax + day * day) / SIGMA2;
  const r2b = (dbx * dbx + dby * dby) / SIGMA2;
  return 2 * Math.exp(-r2a) * (1 - r2a) - 2 * Math.exp(-r2b) * (1 - r2b);
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

export default function Curl() {
  const cx = W / 2;
  const cy = H / 2;
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [p, setP] = useState<Pt>({ x: 0, y: 0 });
  const dragRef = useRef(false);

  const toScreen = (v: Pt) => ({ x: cx + v.x * SCALE, y: cy - v.y * SCALE });

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
      // Screen y is flipped vs world y, so map +y world → -y screen.
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

  const ccwS = toScreen(CCW);
  const cwS = toScreen(CW);

  const pS = toScreen(p);
  const c = curl(p.x, p.y);

  // Probe paddle-wheel: a small arc inside the probe ring, sweep length
  // and direction encode |curl| and sign. SVG arc path.
  const ringR = 18;
  const arcR = 11;
  const cMax = 2.05; // approximate peak |curl| in this field
  const sweep = Math.min(1, Math.abs(c) / cMax);          // 0..1
  const sweepDeg = sweep * 280;                            // up to ~280° arc
  const ccwArc = c > 0;
  // Arc geometry (SVG y is down, so visually-CCW corresponds to SVG's
  // sweep-flag = 0 when going from start to end via the long way).
  const startAng = -90;                                    // top of circle
  const endAng = ccwArc ? startAng - sweepDeg : startAng + sweepDeg;
  const sx1 = pS.x + arcR * Math.cos(startAng * Math.PI / 180);
  const sy1 = pS.y + arcR * Math.sin(startAng * Math.PI / 180);
  const sx2 = pS.x + arcR * Math.cos(endAng * Math.PI / 180);
  const sy2 = pS.y + arcR * Math.sin(endAng * Math.PI / 180);
  const largeArc = sweepDeg > 180 ? 1 : 0;
  const sweepFlag = ccwArc ? 0 : 1;
  const arcD = `M ${sx1} ${sy1} A ${arcR} ${arcR} 0 ${largeArc} ${sweepFlag} ${sx2} ${sy2}`;
  // Arrowhead at the end of the arc, tangent direction.
  const tangentAngle = ccwArc ? endAng - 90 : endAng + 90;
  const tangX = Math.cos(tangentAngle * Math.PI / 180);
  const tangY = Math.sin(tangentAngle * Math.PI / 180);
  const headSize = 6;
  const headTipX = sx2;
  const headTipY = sy2;
  const headBaseX = sx2 - tangX * headSize;
  const headBaseY = sy2 - tangY * headSize;
  const perpX = -tangY;
  const perpY = tangX;
  const arrowHead =
    `${headTipX},${headTipY} ` +
    `${headBaseX + perpX * headSize * 0.55},${headBaseY + perpY * headSize * 0.55} ` +
    `${headBaseX - perpX * headSize * 0.55},${headBaseY - perpY * headSize * 0.55}`;

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
        aria-label="Curl demo"
      >
        <rect x={0.5} y={0.5} width={W - 1} height={H - 1}
              fill="none" stroke="currentColor" strokeWidth={1} opacity={0.3} />

        <line x1={0} y1={cy} x2={W} y2={cy} stroke="currentColor" strokeWidth={0.5} opacity={0.2} />
        <line x1={cx} y1={0} x2={cx} y2={H} stroke="currentColor" strokeWidth={0.5} opacity={0.2} />

        <g opacity={0.85}>
          <circle cx={ccwS.x} cy={ccwS.y} r={3.5} fill="currentColor" />
          <text x={ccwS.x + 8} y={ccwS.y - 6}
                fontSize={13} fontFamily="ui-monospace, monospace" fontWeight={600}
                fill="currentColor">↺</text>
          <circle cx={cwS.x} cy={cwS.y} r={3.5} fill="none" stroke="currentColor" strokeWidth={1.4} />
          <text x={cwS.x + 8} y={cwS.y - 6}
                fontSize={13} fontFamily="ui-monospace, monospace" fontWeight={600}
                fill="currentColor">↻</text>
        </g>

        {quiver.map((q, i) => q && (
          <g key={i} opacity={q.op}>
            <line x1={q.shaft.x1} y1={q.shaft.y1} x2={q.shaft.x2} y2={q.shaft.y2}
                  stroke="currentColor" strokeWidth={1} strokeLinecap="round" />
            <polygon points={q.head} fill="currentColor" />
          </g>
        ))}

        {/* Probe ring + paddle-wheel arc (rotation indicator) */}
        <circle cx={pS.x} cy={pS.y} r={ringR}
                fill="none" stroke="currentColor" strokeWidth={1.4} opacity={0.85} />
        {Math.abs(c) > 0.05 && (
          <g opacity={0.85}>
            <path d={arcD} fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" />
            <polygon points={arrowHead} fill="currentColor" />
          </g>
        )}

        <circle cx={pS.x} cy={pS.y} r={3}
                fill="currentColor" fillOpacity={0.85}
                onPointerDown={(e) => { (e.target as Element).setPointerCapture(e.pointerId); dragRef.current = true; }}
                style={{ cursor: "grab" }} />

        <text x={W - 8} y={H - 8} textAnchor="end"
              fontSize={11} fontFamily="ui-monospace, monospace"
              fill="currentColor" opacity={0.85}>
          (∇×F)_z = {c.toFixed(2)}
        </text>
      </svg>
      <figcaption style={{ fontSize: "0.85rem", opacity: 0.7, marginTop: 6 }}>
        Arrows show a vector field <em>F</em> with two vortices — a counter-clockwise one
        (filled dot, ↺) and a clockwise one (hollow dot, ↻). Drag the probe to read (∇×<em>F</em>)<sub>z</sub>
        at that point; the curved arrow inside the probe ring is a "paddle wheel" that would spin
        the same way (counter-clockwise for positive curl, clockwise for negative), with arc length
        scaling with |∇×<em>F</em>|. Between the vortices, and far from both, the field still flows
        but doesn't twist — curl is zero.
      </figcaption>
    </figure>
  );
}
