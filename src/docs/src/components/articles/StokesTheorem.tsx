import { useMemo, useRef, useState } from "react";

type Pt = { x: number; y: number };

const W = 360;
const H = 240;
const PITCH = 30;
const SCALE = 75;
const CELL_W = PITCH / SCALE;

// Same two-vortex field as the curl demo so the two pages line up.
const SIGMA2 = 0.6;
const CCW = { x: -2 * CELL_W, y: -1 * CELL_W };
const CW  = { x:  2 * CELL_W, y:  1 * CELL_W };
const REGION_R = 0.55;                       // world units

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

function curl(x: number, y: number): number {
  const dax = x - CCW.x;
  const day = y - CCW.y;
  const dbx = x - CW.x;
  const dby = y - CW.y;
  const r2a = (dax * dax + day * day) / SIGMA2;
  const r2b = (dbx * dbx + dby * dby) / SIGMA2;
  return 2 * Math.exp(-r2a) * (1 - r2a) - 2 * Math.exp(-r2b) * (1 - r2b);
}

// ∫∫_R (∇×F)_z dA over disk centred at p with radius R.
function integrateCurl(p: Pt, R: number): number {
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
      sum += curl(x, y) * dA;
    }
  }
  return sum;
}

// ∮_∂R F·dℓ — tangential circulation, CCW orientation chosen by Stokes.
function integrateCirc(p: Pt, R: number): number {
  const NB = 64;
  const dAng = (2 * Math.PI) / NB;
  const dl = R * dAng;
  let sum = 0;
  for (let j = 0; j < NB; j++) {
    const ang = (j + 0.5) * dAng;
    const x = p.x + R * Math.cos(ang);
    const y = p.y + R * Math.sin(ang);
    const F = field(x, y);
    // CCW tangent in world coords: (-sin, cos)
    const tx = -Math.sin(ang);
    const ty =  Math.cos(ang);
    sum += (F.x * tx + F.y * ty) * dl;
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

export default function StokesTheorem() {
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

  const ccwS = toScreen(CCW);
  const cwS = toScreen(CW);
  const pS = toScreen(p);
  const regionRpx = REGION_R * SCALE;

  // Boundary circulation indicators: short tangential arrows showing F·t
  // sampled around ∂R. Direction along the boundary picks the CCW sense
  // when F aligns with it (positive contribution) or the CW sense when
  // it opposes (negative contribution). Length scales with |F·t|.
  const boundary = useMemo(() => {
    const N = 18;
    const samples: { ang: number; ft: number }[] = [];
    let maxFT = 0;
    for (let i = 0; i < N; i++) {
      const ang = (i + 0.5) * (2 * Math.PI) / N;
      const wx = p.x + REGION_R * Math.cos(ang);
      const wy = p.y + REGION_R * Math.sin(ang);
      const F = field(wx, wy);
      const tx = -Math.sin(ang);
      const ty =  Math.cos(ang);
      const ft = F.x * tx + F.y * ty;
      if (Math.abs(ft) > maxFT) maxFT = Math.abs(ft);
      samples.push({ ang, ft });
    }
    return samples.map(({ ang, ft }) => {
      const wxBase = p.x + REGION_R * Math.cos(ang);
      const wyBase = p.y + REGION_R * Math.sin(ang);
      const baseS = toScreen({ x: wxBase, y: wyBase });
      const len = Math.min(14, Math.abs(ft) / (maxFT || 1) * 14);
      const sign = ft >= 0 ? 1 : -1;
      // World tangent (-sin, cos); SVG screen has y inverted.
      const txS = -Math.sin(ang);
      const tyS = -Math.cos(ang);
      const tipS = {
        x: baseS.x + sign * len * txS,
        y: baseS.y + sign * len * tyS,
      };
      return arrowPath(baseS.x, baseS.y, tipS.x, tipS.y, 5);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p.x, p.y]);

  const lhs = integrateCurl(p, REGION_R);
  const rhs = integrateCirc(p, REGION_R);

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
        aria-label="Stokes theorem demo"
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

        <circle cx={pS.x} cy={pS.y} r={regionRpx}
                fill="currentColor" fillOpacity={0.06}
                stroke="currentColor" strokeWidth={1.6} strokeOpacity={0.85} />

        {boundary.map((q, i) => q && (
          <g key={`b-${i}`}>
            <line x1={q.shaft.x1} y1={q.shaft.y1} x2={q.shaft.x2} y2={q.shaft.y2}
                  stroke="currentColor" strokeWidth={1.4} strokeLinecap="round" />
            <polygon points={q.head} fill="currentColor" />
          </g>
        ))}

        <circle cx={pS.x} cy={pS.y} r={6}
                fill="currentColor" fillOpacity={0.15}
                stroke="currentColor" strokeWidth={1.4}
                onPointerDown={(e) => { (e.target as Element).setPointerCapture(e.pointerId); dragRef.current = true; }}
                style={{ cursor: "grab" }} />

        <text x={8} y={H - 24} fontSize={11} fontFamily="ui-monospace, monospace"
              fill="currentColor" opacity={0.85}>
          ∫∫ (∇×F)_z dA = {lhs.toFixed(2)}
        </text>
        <text x={8} y={H - 8} fontSize={11} fontFamily="ui-monospace, monospace"
              fill="currentColor" opacity={0.85}>
          ∮ F·dℓ = {rhs.toFixed(2)}
        </text>
        <text x={W - 8} y={H - 8} textAnchor="end"
              fontSize={11} fontFamily="ui-monospace, monospace"
              fill="currentColor" opacity={0.6}>
          difference = {(lhs - rhs).toFixed(3)}
        </text>
      </svg>
      <figcaption style={{ fontSize: "0.85rem", opacity: 0.7, marginTop: 6 }}>
        Drag the disc-shaped surface <em>S</em>. The arrows on its boundary are the local
        tangential contributions <em>F</em>·d<em>ℓ</em> — going CCW around the disc when positive,
        CW when negative. The two readouts are the two sides of Stokes' theorem:
        ∫∫<sub>S</sub> (∇×<em>F</em>)<sub>z</sub> d<em>A</em> on top, ∮<sub>∂S</sub> <em>F</em>·d<em>ℓ</em>
        on the bottom. Enclose only the CCW vortex (both positive), only the CW one (both
        negative), both (cancel), or neither (both ~ zero).
      </figcaption>
    </figure>
  );
}
