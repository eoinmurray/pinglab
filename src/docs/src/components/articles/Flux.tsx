import { useMemo, useRef, useState } from "react";

type Pt = { x: number; y: number };

const W = 360;
const H = 240;
const PITCH = 30;
const SCALE = 75;
const CELL_W = PITCH / SCALE;

// Source-sink field, same as the divergence demos for continuity.
const SIGMA2 = 0.6;
const SRC = { x: -2 * CELL_W, y: -1 * CELL_W };
const SNK = { x:  2 * CELL_W, y:  1 * CELL_W };

function field(x: number, y: number): Pt {
  const dsx = x - SRC.x;
  const dsy = y - SRC.y;
  const dnx = x - SNK.x;
  const dny = y - SNK.y;
  const es = Math.exp(-(dsx * dsx + dsy * dsy) / SIGMA2);
  const en = Math.exp(-(dnx * dnx + dny * dny) / SIGMA2);
  return { x: es * dsx - en * dnx, y: es * dsy - en * dny };
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

export default function Flux() {
  const cx = W / 2;
  const cy = H / 2;
  const svgRef = useRef<SVGSVGElement | null>(null);
  // Horizontal segment between (but not on) source and sink, so neither
  // handle overlaps a marker. The line cuts the saddle band where the
  // field flows from source toward sink.
  const [a, setA] = useState<Pt>({ x: -0.5, y: 0.0 });
  const [b, setB] = useState<Pt>({ x:  0.5, y: 0.0 });
  const dragRef = useRef<null | "a" | "b">(null);

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
  const aS = toScreen(a);
  const bS = toScreen(b);

  // Tangent and (right-hand) normal of the segment in world coords.
  const dxW = b.x - a.x;
  const dyW = b.y - a.y;
  const lenW = Math.hypot(dxW, dyW);
  const tx = lenW > 0 ? dxW / lenW : 1;
  const ty = lenW > 0 ? dyW / lenW : 0;
  // Right-hand normal of (tx, ty) is (ty, -tx).
  const nx = ty;
  const ny = -tx;

  // Integrate ∫ F·n dℓ along the segment.
  const N_QUAD = 40;
  let flux = 0;
  for (let i = 0; i < N_QUAD; i++) {
    const t = (i + 0.5) / N_QUAD;
    const x = a.x + t * dxW;
    const y = a.y + t * dyW;
    const F = field(x, y);
    flux += (F.x * nx + F.y * ny) * (lenW / N_QUAD);
  }

  // Sample arrows along the segment showing F·n locally.
  const N_SAMPLES = 14;
  const samples: ReturnType<typeof arrowPath>[] = [];
  let maxFN = 0;
  const fnValues: { sx: number; sy: number; fn: number }[] = [];
  for (let i = 0; i < N_SAMPLES; i++) {
    const t = (i + 0.5) / N_SAMPLES;
    const xW = a.x + t * dxW;
    const yW = a.y + t * dyW;
    const F = field(xW, yW);
    const fn = F.x * nx + F.y * ny;
    if (Math.abs(fn) > maxFN) maxFN = Math.abs(fn);
    const s = toScreen({ x: xW, y: yW });
    fnValues.push({ sx: s.x, sy: s.y, fn });
  }
  for (const { sx, sy, fn } of fnValues) {
    const L = Math.min(14, Math.abs(fn) / (maxFN || 1) * 14);
    const sign = fn >= 0 ? 1 : -1;
    // Screen normal: world (nx, ny) → screen (nx, -ny).
    const nxS = nx;
    const nyS = -ny;
    samples.push(arrowPath(
      sx, sy,
      sx + sign * L * nxS,
      sy + sign * L * nyS,
      5,
    ));
  }

  // Big normal vector indicator at the midpoint, length proportional
  // to segment length so it looks proportionate.
  const midS = toScreen({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
  const nIndicator = arrowPath(
    midS.x, midS.y,
    midS.x + nx * 28,
    midS.y - ny * 28,
    7,
  );

  const handlePointer = (e: React.PointerEvent) => {
    if (!dragRef.current || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const vbX = ((e.clientX - rect.left) / rect.width) * W;
    const vbY = ((e.clientY - rect.top) / rect.height) * H;
    const wx = Math.max(-HALF_X, Math.min(HALF_X, (vbX - cx) / SCALE));
    const wy = Math.max(-HALF_Y, Math.min(HALF_Y, -(vbY - cy) / SCALE));
    const p: Pt = { x: wx, y: wy };
    if (dragRef.current === "a") setA(p);
    else setB(p);
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
        onPointerUp={() => { dragRef.current = null; }}
        onPointerLeave={() => { dragRef.current = null; }}
        style={{ touchAction: "none", display: "block", width: "100%", height: "auto", maxWidth: 720 }}
        role="img"
        aria-label="Flux demo"
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

        {/* The segment */}
        <line x1={aS.x} y1={aS.y} x2={bS.x} y2={bS.y}
              stroke="currentColor" strokeWidth={1.8} opacity={0.85} />

        {/* Sample F·n arrows along the segment */}
        {samples.map((q, i) => q && (
          <g key={`s-${i}`}>
            <line x1={q.shaft.x1} y1={q.shaft.y1} x2={q.shaft.x2} y2={q.shaft.y2}
                  stroke="currentColor" strokeWidth={1.4} strokeLinecap="round" />
            <polygon points={q.head} fill="currentColor" />
          </g>
        ))}

        {/* Normal direction indicator at the midpoint */}
        {nIndicator && (
          <g opacity={0.55}>
            <line x1={nIndicator.shaft.x1} y1={nIndicator.shaft.y1}
                  x2={nIndicator.shaft.x2} y2={nIndicator.shaft.y2}
                  stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" />
            <polygon points={nIndicator.head} fill="currentColor" />
            <text x={nIndicator.shaft.x2 + nx * 6} y={midS.y - ny * 38}
                  fontSize={11} fontFamily="ui-monospace, monospace"
                  fontStyle="italic" fill="currentColor">n</text>
          </g>
        )}

        {/* Endpoint handles */}
        <circle cx={aS.x} cy={aS.y} r={8}
                fill="currentColor" fillOpacity={0.12} stroke="currentColor" strokeWidth={1.4}
                onPointerDown={(e) => { (e.target as Element).setPointerCapture(e.pointerId); dragRef.current = "a"; }}
                style={{ cursor: "grab" }} />
        <circle cx={bS.x} cy={bS.y} r={8}
                fill="currentColor" fillOpacity={0.12} stroke="currentColor" strokeWidth={1.4}
                onPointerDown={(e) => { (e.target as Element).setPointerCapture(e.pointerId); dragRef.current = "b"; }}
                style={{ cursor: "grab" }} />

        <text x={W - 8} y={H - 8} textAnchor="end"
              fontSize={11} fontFamily="ui-monospace, monospace"
              fill="currentColor" opacity={0.85}>
          ∫ F·n dℓ = {flux.toFixed(2)}
        </text>
      </svg>
      <figcaption style={{ fontSize: "0.85rem", opacity: 0.7, marginTop: 6 }}>
        Drag either endpoint of the line segment through the field. The small arrows along the segment are
        the local <em>F</em>·<em>n</em> contributions — outward (along the chosen normal <em>n</em>) when
        positive, opposite when negative. The readout is their sum: the flux of <em>F</em> across the
        segment. Rotate the segment by swinging an endpoint and watch the sign flip when the normal swaps sides;
        line the segment up <em>along</em> the field and the flux drops to zero (nothing is crossing it).
      </figcaption>
    </figure>
  );
}
