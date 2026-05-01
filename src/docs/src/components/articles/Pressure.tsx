import { useEffect, useRef, useState } from "react";
import { Group } from "@visx/group";
import { ParentSize } from "@visx/responsive";

type Particle = { x: number; y: number; vx: number; vy: number };
type MomEntry = { t: number; p: number };

const N = 40;
const RADIUS = 5;
const INITIAL_SPEED = 100;
const BOX_H = 240;
const WINDOW_S = 1.0;

function initParticles(w: number, h: number): Particle[] {
  const ps: Particle[] = [];
  const cols = Math.ceil(Math.sqrt((N * w) / h));
  const rows = Math.ceil(N / cols);
  const dx = w / (cols + 1);
  const dy = h / (rows + 1);
  let i = 0;
  for (let r = 0; r < rows && i < N; r++) {
    for (let c = 0; c < cols && i < N; c++, i++) {
      const a = Math.random() * Math.PI * 2;
      ps.push({
        x: dx * (c + 1),
        y: dy * (r + 1),
        vx: Math.cos(a) * INITIAL_SPEED,
        vy: Math.sin(a) * INITIAL_SPEED,
      });
    }
  }
  return ps;
}

function step(
  ps: Particle[],
  dt: number,
  w: number,
  h: number,
  log: MomEntry[],
  now: number,
) {
  for (const p of ps) {
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    let mom = 0;
    if (p.x < RADIUS) { p.x = RADIUS; mom += 2 * Math.abs(p.vx); p.vx = -p.vx; }
    else if (p.x > w - RADIUS) { p.x = w - RADIUS; mom += 2 * Math.abs(p.vx); p.vx = -p.vx; }
    if (p.y < RADIUS) { p.y = RADIUS; mom += 2 * Math.abs(p.vy); p.vy = -p.vy; }
    else if (p.y > h - RADIUS) { p.y = h - RADIUS; mom += 2 * Math.abs(p.vy); p.vy = -p.vy; }
    if (mom > 0) log.push({ t: now, p: mom });
  }
  const minD = 2 * RADIUS;
  for (let i = 0; i < ps.length; i++) {
    const a = ps[i];
    for (let j = i + 1; j < ps.length; j++) {
      const b = ps[j];
      const ddx = b.x - a.x;
      const ddy = b.y - a.y;
      const d2 = ddx * ddx + ddy * ddy;
      if (d2 === 0 || d2 > minD * minD) continue;
      const d = Math.sqrt(d2);
      const nx = ddx / d;
      const ny = ddy / d;
      const vn = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny;
      if (vn > 0) continue;
      a.vx += vn * nx; a.vy += vn * ny;
      b.vx -= vn * nx; b.vy -= vn * ny;
      const overlap = (minD - d) / 2;
      a.x -= nx * overlap; a.y -= ny * overlap;
      b.x += nx * overlap; b.y += ny * overlap;
    }
  }
  while (log.length > 0 && log[0].t < now - WINDOW_S) log.shift();
}

function Demo({
  fullW,
  widthFracRef,
  widthFrac,
  resetVersion,
}: {
  fullW: number;
  widthFracRef: { current: number };
  widthFrac: number;
  resetVersion: number;
}) {
  const psRef = useRef<Particle[] | null>(null);
  const momRef = useRef<MomEntry[]>([]);
  const startTRef = useRef(0);
  const lastTRef = useRef(0);
  const [, setTick] = useState(0);

  useEffect(() => {
    if (!fullW) return;
    const initialW = fullW * widthFracRef.current;
    psRef.current = initParticles(initialW, BOX_H);
    momRef.current = [];
    startTRef.current = 0;
    lastTRef.current = 0;

    let raf = 0;
    const loop = (t: number) => {
      if (startTRef.current === 0) startTRef.current = t;
      const dt = lastTRef.current === 0 ? 0 : Math.min((t - lastTRef.current) / 1000, 0.05);
      lastTRef.current = t;
      const now = (t - startTRef.current) / 1000;
      const boxW = fullW * widthFracRef.current;
      if (psRef.current) {
        for (const p of psRef.current) {
          if (p.x > boxW - RADIUS) p.x = boxW - RADIUS;
        }
        step(psRef.current, dt, boxW, BOX_H, momRef.current, now);
      }
      setTick((n) => n + 1);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [fullW, widthFracRef, resetVersion]);

  const boxW = fullW * widthFrac;
  const ps = psRef.current ?? [];
  const momSum = momRef.current.reduce((s, m) => s + m.p, 0);
  const elapsed = lastTRef.current === 0 ? 0 : (lastTRef.current - startTRef.current) / 1000;
  const window = Math.max(0.1, Math.min(WINDOW_S, elapsed));
  const perimeter = 2 * (boxW + BOX_H);
  const P = momSum / (perimeter * window);
  const V = boxW * BOX_H;
  const meanKE = ps.length === 0 ? 0 : ps.reduce((s, p) => s + 0.5 * (p.vx * p.vx + p.vy * p.vy), 0) / ps.length;
  const pvOverN = (P * V) / (ps.length || 1);

  return (
    <div>
      <svg width={fullW} height={BOX_H} role="img" aria-label="Pressure demo">
        <rect
          x={boxW + 0.5}
          y={0}
          width={fullW - boxW - 0.5}
          height={BOX_H}
          fill="currentColor"
          opacity={0.04}
        />
        <rect
          x={0.5}
          y={0.5}
          width={boxW - 1}
          height={BOX_H - 1}
          fill="none"
          stroke="currentColor"
          strokeWidth={1}
          opacity={0.4}
        />
        <line
          x1={boxW}
          y1={0}
          x2={boxW}
          y2={BOX_H}
          stroke="currentColor"
          strokeWidth={2}
          opacity={0.6}
        />
        <Group>
          {ps.map((p, i) => (
            <circle key={i} cx={p.x} cy={p.y} r={RADIUS} fill="currentColor" opacity={0.85} />
          ))}
        </Group>
      </svg>
      <div
        style={{
          marginTop: 8,
          fontFamily: "ui-monospace, monospace",
          fontSize: "0.85rem",
          opacity: 0.75,
          display: "flex",
          gap: 18,
          flexWrap: "wrap",
        }}
      >
        <span>P ≈ {P.toFixed(1)}</span>
        <span>V = {V.toFixed(0)}</span>
        <span>T ≈ {meanKE.toFixed(0)}</span>
        <span>PV/N ≈ {pvOverN.toFixed(0)}</span>
      </div>
    </div>
  );
}

const btn: React.CSSProperties = {
  padding: "4px 14px",
  border: "1px solid currentColor",
  borderRadius: 2,
  background: "transparent",
  cursor: "pointer",
  fontFamily: "inherit",
  fontSize: "0.85rem",
  opacity: 0.8,
};

export default function Pressure() {
  const [widthFrac, setWidthFrac] = useState(1.0);
  const widthFracRef = useRef(1.0);
  widthFracRef.current = widthFrac;
  const [resetVersion, setResetVersion] = useState(0);
  return (
    <div style={{ width: "100%" }}>
      <div style={{ display: "flex", gap: 12, marginBottom: 12, alignItems: "center", flexWrap: "wrap" }}>
        <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: "0.85rem", opacity: 0.8 }}>
          box width
          <input
            type="range"
            min={0.3}
            max={1.0}
            step={0.01}
            value={widthFrac}
            onChange={(e) => setWidthFrac(parseFloat(e.target.value))}
            style={{ width: 200 }}
          />
          <span style={{ fontFamily: "ui-monospace, monospace" }}>{widthFrac.toFixed(2)}</span>
        </label>
        <button onClick={() => setResetVersion((v) => v + 1)} style={btn}>reset</button>
      </div>
      <ParentSize>
        {({ width }) => (
          <Demo
            fullW={width}
            widthFracRef={widthFracRef}
            widthFrac={widthFrac}
            resetVersion={resetVersion}
          />
        )}
      </ParentSize>
    </div>
  );
}
