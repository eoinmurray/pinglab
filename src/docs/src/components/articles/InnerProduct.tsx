import { useRef, useState } from "react";

type Vec = { x: number; y: number };

const W = 360;
const H = 240;
const SCALE = 75;     // px per world unit (matches CrossProduct + Gradient)

function dot(a: Vec, b: Vec) {
  return a.x * b.x + a.y * b.y;
}

function Arrow({
  x1, y1, x2, y2, opacity = 1,
}: { x1: number; y1: number; x2: number; y2: number; opacity?: number }) {
  const head = 8;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy);
  if (len < 1) return null;
  const ux = dx / len;
  const uy = dy / len;
  const baseX = x2 - ux * head;
  const baseY = y2 - uy * head;
  const px = -uy;
  const py = ux;
  const l1 = `${baseX + px * head * 0.55},${baseY + py * head * 0.55}`;
  const l2 = `${baseX - px * head * 0.55},${baseY - py * head * 0.55}`;
  return (
    <g opacity={opacity}>
      <line x1={x1} y1={y1} x2={baseX} y2={baseY}
            stroke="currentColor" strokeWidth={2} strokeLinecap="round" />
      <polygon points={`${x2},${y2} ${l1} ${l2}`} fill="currentColor" />
    </g>
  );
}

export default function InnerProduct() {
  const cx = W / 2;
  const cy = H / 2;
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [a, setA] = useState<Vec>({ x: 0.6, y: 0.8 });
  const [b, setB] = useState<Vec>({ x: 1.0, y: -0.4 });
  const dragRef = useRef<null | "a" | "b">(null);

  const toScreen = (v: Vec) => ({ x: cx + v.x * SCALE, y: cy - v.y * SCALE });
  const aS = toScreen(a);
  const bS = toScreen(b);

  const handlePointer = (e: React.PointerEvent) => {
    if (!dragRef.current || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const vbX = ((e.clientX - rect.left) / rect.width) * W;
    const vbY = ((e.clientY - rect.top) / rect.height) * H;
    const v: Vec = { x: (vbX - cx) / SCALE, y: -(vbY - cy) / SCALE };
    if (dragRef.current === "a") setA(v);
    else setB(v);
  };

  const d = dot(a, b);
  const mb2 = b.x * b.x + b.y * b.y;
  const projScale = mb2 > 0 ? d / mb2 : 0;
  const proj: Vec = { x: b.x * projScale, y: b.y * projScale };
  const projS = toScreen(proj);

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
        aria-label="Inner product demo"
      >
        <rect x={0.5} y={0.5} width={W - 1} height={H - 1}
              fill="none" stroke="currentColor" strokeWidth={1} opacity={0.3} />
        <line x1={0} y1={cy} x2={W} y2={cy}
              stroke="currentColor" strokeWidth={0.5} opacity={0.25} />
        <line x1={cx} y1={0} x2={cx} y2={H}
              stroke="currentColor" strokeWidth={0.5} opacity={0.25} />

        {mb2 > 0 && (
          <line x1={aS.x} y1={aS.y} x2={projS.x} y2={projS.y}
                stroke="currentColor" strokeWidth={1} strokeDasharray="3 3" opacity={0.45} />
        )}
        {mb2 > 0 && Math.abs(projScale) > 0.02 && (
          <Arrow x1={cx} y1={cy} x2={projS.x} y2={projS.y} opacity={0.45} />
        )}
        <Arrow x1={cx} y1={cy} x2={bS.x} y2={bS.y} />
        <Arrow x1={cx} y1={cy} x2={aS.x} y2={aS.y} />

        <circle cx={aS.x} cy={aS.y} r={9}
                fill="currentColor" fillOpacity={0.1} stroke="currentColor" strokeWidth={1.5}
                onPointerDown={(e) => { (e.target as Element).setPointerCapture(e.pointerId); dragRef.current = "a"; }}
                style={{ cursor: "grab" }} />
        <circle cx={bS.x} cy={bS.y} r={9}
                fill="currentColor" fillOpacity={0.1} stroke="currentColor" strokeWidth={1.5}
                onPointerDown={(e) => { (e.target as Element).setPointerCapture(e.pointerId); dragRef.current = "b"; }}
                style={{ cursor: "grab" }} />

        <text x={aS.x + 12} y={aS.y - 8}
              fontSize={13} fontFamily="ui-monospace, monospace"
              fontStyle="italic" fill="currentColor">a</text>
        <text x={bS.x + 12} y={bS.y - 8}
              fontSize={13} fontFamily="ui-monospace, monospace"
              fontStyle="italic" fill="currentColor">b</text>

        <text x={W - 8} y={H - 8} textAnchor="end"
              fontSize={11} fontFamily="ui-monospace, monospace"
              fill="currentColor" opacity={0.85}>
          a · b = {d.toFixed(2)}
        </text>
      </svg>
      <figcaption style={{ fontSize: "0.85rem", opacity: 0.7, marginTop: 6 }}>
        Drag either tip. The dashed segment drops a perpendicular from <em>a</em> onto the line of <em>b</em>;
        the grey arrow along <em>b</em> is the projection, and its signed length equals
        (<em>a</em> · <em>b</em>) / |<em>b</em>|. Make the two vectors perpendicular and the dot product
        goes to zero; rotate past 90° and it goes negative.
      </figcaption>
    </figure>
  );
}
