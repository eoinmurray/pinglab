import { useMemo, useRef, useState } from "react";

type Pt = { x: number; y: number };

const W = 360;
const H = 240;
const PITCH = 30;
const SCALE = 75;
const CELL_W = PITCH / SCALE;

// Same source-sink field as the divergence demo so the two pages line up.
const SIGMA2 = 0.6;
const SRC = { x: -2 * CELL_W, y: -1 * CELL_W };
const SNK = { x:  2 * CELL_W, y:  1 * CELL_W };
const REGION_R = 0.55;                       // world units

function field(x: number, y: number): Pt {
  const dsx = x - SRC.x;
  const dsy = y - SRC.y;
  const dnx = x - SNK.x;
  const dny = y - SNK.y;
  const es = Math.exp(-(dsx * dsx + dsy * dsy) / SIGMA2);
  const en = Math.exp(-(dnx * dnx + dny * dny) / SIGMA2);
  return { x: es * dsx - en * dnx, y: es * dsy - en * dny };
}

function divergence(x: number, y: number): number {
  const dsx = x - SRC.x;
  const dsy = y - SRC.y;
  const dnx = x - SNK.x;
  const dny = y - SNK.y;
  const r2s = (dsx * dsx + dsy * dsy) / SIGMA2;
  const r2n = (dnx * dnx + dny * dny) / SIGMA2;
  return 2 * Math.exp(-r2s) * (1 - r2s) - 2 * Math.exp(-r2n) * (1 - r2n);
}

// ∫∫_R (∇·F) dA over disk centred at p with radius R, polar grid.
function integrateDiv(p: Pt, R: number): number {
  const NR = 12;
  const NA = 28;
  const dr = R / NR;
  const dAng = (2 * Math.PI) / NA;
  let sum = 0;
  for (let i = 0; i < NR; i++) {
    const r = (i + 0.5) * dr;
    for (let j = 0; j < NA; j++) {
      const ang = (j + 0.5) * dAng;
      const x = p.x + r * Math.cos(ang);
      const y = p.y + r * Math.sin(ang);
      const dA = r * dr * dAng;
      sum += divergence(x, y) * dA;
    }
  }
  return sum;
}

// ∮_∂R F·n dℓ around the boundary, outward normal.
function integrateFlux(p: Pt, R: number): number {
  const NB = 64;
  const dAng = (2 * Math.PI) / NB;
  const dl = R * dAng;
  let sum = 0;
  for (let j = 0; j < NB; j++) {
    const ang = (j + 0.5) * dAng;
    const x = p.x + R * Math.cos(ang);
    const y = p.y + R * Math.sin(ang);
    const F = field(x, y);
    const nx = Math.cos(ang);
    const ny = Math.sin(ang);
    sum += (F.x * nx + F.y * ny) * dl;
  }
  return sum;
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

export default function DivergenceTheorem() {
  const cx = W / 2;
  const cy = H / 2;
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [p, setP] = useState<Pt>({ x: -CELL_W * 2, y: -CELL_W * 1 });
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
    const MIN_FRAC = 0.30;
    const MAX_FRAC = 0.74;
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
      const op = 0.14 + 0.50 * Math.sqrt(t);
      return arrow ? { ...arrow, op } : null;
    });
    return { quiver, fmax };
  }, []);

  const srcS = toScreen(SRC);
  const snkS = toScreen(SNK);
  const pS = toScreen(p);
  const regionRpx = REGION_R * SCALE;

  // Boundary flux indicators: short outward / inward arrows sampled around
  // the region's edge so the reader can SEE the local F·n contributions
  // that the integral on the right sums up.
  const boundary = useMemo(() => {
    const N = 18;
    let maxFN = 0;
    const samples: { ang: number; fn: number }[] = [];
    for (let i = 0; i < N; i++) {
      const ang = (i + 0.5) * (2 * Math.PI) / N;
      const wx = p.x + REGION_R * Math.cos(ang);
      const wy = p.y + REGION_R * Math.sin(ang);
      const F = field(wx, wy);
      const fn = F.x * Math.cos(ang) + F.y * Math.sin(ang);
      if (Math.abs(fn) > maxFN) maxFN = Math.abs(fn);
      samples.push({ ang, fn });
    }
    return samples.map(({ ang, fn }) => {
      const wxBase = p.x + REGION_R * Math.cos(ang);
      const wyBase = p.y + REGION_R * Math.sin(ang);
      const baseS = toScreen({ x: wxBase, y: wyBase });
      const len = Math.min(14, Math.abs(fn) / (maxFN || 1) * 14);
      const sign = fn >= 0 ? 1 : -1;
      // SVG-space normal: outward radial direction. Note y-flip.
      const nxS = Math.cos(ang);
      const nyS = -Math.sin(ang);
      const tipS = {
        x: baseS.x + sign * len * nxS,
        y: baseS.y + sign * len * nyS,
      };
      return arrowPath(baseS.x, baseS.y, tipS.x, tipS.y, 5);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p.x, p.y]);

  const lhs = integrateDiv(p, REGION_R);
  const rhs = integrateFlux(p, REGION_R);

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
        aria-label="Divergence theorem demo"
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

        {/* Region boundary */}
        <circle cx={pS.x} cy={pS.y} r={regionRpx}
                fill="currentColor" fillOpacity={0.06}
                stroke="currentColor" strokeWidth={1.6} strokeOpacity={0.85} />

        {/* Boundary flux indicators */}
        {boundary.map((q, i) => q && (
          <g key={`b-${i}`}>
            <line x1={q.shaft.x1} y1={q.shaft.y1} x2={q.shaft.x2} y2={q.shaft.y2}
                  stroke="currentColor" strokeWidth={1.4} strokeLinecap="round" />
            <polygon points={q.head} fill="currentColor" />
          </g>
        ))}

        {/* Drag handle at the region centre */}
        <circle cx={pS.x} cy={pS.y} r={6}
                fill="currentColor" fillOpacity={0.15}
                stroke="currentColor" strokeWidth={1.4}
                onPointerDown={(e) => { (e.target as Element).setPointerCapture(e.pointerId); dragRef.current = true; }}
                style={{ cursor: "grab" }} />

        <text x={8} y={H - 24} fontSize={11} fontFamily="ui-monospace, monospace"
              fill="currentColor" opacity={0.85}>
          ∫∫ ∇·F dA = {lhs.toFixed(2)}
        </text>
        <text x={8} y={H - 8} fontSize={11} fontFamily="ui-monospace, monospace"
              fill="currentColor" opacity={0.85}>
          ∮ F·n dℓ = {rhs.toFixed(2)}
        </text>
        <text x={W - 8} y={H - 8} textAnchor="end"
              fontSize={11} fontFamily="ui-monospace, monospace"
              fill="currentColor" opacity={0.6}>
          difference = {(lhs - rhs).toFixed(3)}
        </text>
      </svg>
      <figcaption style={{ fontSize: "0.85rem", opacity: 0.7, marginTop: 6 }}>
        Drag the disc-shaped region <em>R</em>. The small arrows on the boundary show <em>F</em>·<em>n</em> —
        local flux out of (outward arrow) or into (inward arrow) <em>R</em>. The two readouts are the
        two sides of the theorem: ∫∫<sub>R</sub> ∇·<em>F</em> d<em>A</em> on top, ∮<sub>∂R</sub> <em>F</em>·<em>n</em> d<em>ℓ</em>
        on the bottom. They stay equal regardless of where you place the region: enclose only the source
        (both positive), only the sink (both negative), both (cancel), neither (both ~ zero).
      </figcaption>
    </figure>
  );
}
