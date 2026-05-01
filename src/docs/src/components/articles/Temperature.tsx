import { useEffect, useRef, useState } from "react";
import { Group } from "@visx/group";
import { ParentSize } from "@visx/responsive";

type Particle = { x: number; y: number; vx: number; vy: number };

const N = 40;
const RADIUS = 5;
const INITIAL_SPEED = 80;
const HIST_BINS = 14;
const V_MAX = 3 * INITIAL_SPEED;

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

function step(ps: Particle[], dt: number, w: number, h: number) {
  for (const p of ps) {
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    if (p.x < RADIUS) { p.x = RADIUS; p.vx = -p.vx; }
    else if (p.x > w - RADIUS) { p.x = w - RADIUS; p.vx = -p.vx; }
    if (p.y < RADIUS) { p.y = RADIUS; p.vy = -p.vy; }
    else if (p.y > h - RADIUS) { p.y = h - RADIUS; p.vy = -p.vy; }
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
}

function speedHue(v: number) {
  const t = Math.min(1, v / (2 * INITIAL_SPEED));
  return 220 - 220 * t;
}

function Demo({ width, scaleSignal, resetVersion }: { width: number; scaleSignal: { v: number }; resetVersion: number }) {
  const boxH = Math.max(180, Math.round(width * 0.5));
  const histH = 100;
  const psRef = useRef<Particle[] | null>(null);
  const lastTRef = useRef(0);
  const [, setTick] = useState(0);

  useEffect(() => {
    if (!width) return;
    psRef.current = initParticles(width, boxH);
    lastTRef.current = 0;
    let raf = 0;
    const loop = (t: number) => {
      const dt = lastTRef.current === 0 ? 0 : Math.min((t - lastTRef.current) / 1000, 0.05);
      lastTRef.current = t;
      if (scaleSignal.v !== 1 && psRef.current) {
        for (const p of psRef.current) { p.vx *= scaleSignal.v; p.vy *= scaleSignal.v; }
        scaleSignal.v = 1;
      }
      if (psRef.current) step(psRef.current, dt, width, boxH);
      setTick((n) => n + 1);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [width, boxH, scaleSignal, resetVersion]);

  const ps = psRef.current ?? [];
  const meanKE = ps.length === 0
    ? 0
    : ps.reduce((s, p) => s + 0.5 * (p.vx * p.vx + p.vy * p.vy), 0) / ps.length;
  const T = Math.max(meanKE, 1);

  const counts = new Array(HIST_BINS).fill(0);
  for (const p of ps) {
    const v = Math.hypot(p.vx, p.vy);
    const idx = Math.min(HIST_BINS - 1, Math.floor((v / V_MAX) * HIST_BINS));
    counts[idx]++;
  }
  const maxC = Math.max(...counts, 1);
  const binW = V_MAX / HIST_BINS;

  const samples = 60;
  const mb: string[] = [];
  for (let i = 0; i <= samples; i++) {
    const v = (V_MAX * i) / samples;
    const f = (v / T) * Math.exp((-v * v) / (2 * T));
    const expected = f * binW * N;
    const x = (i / samples) * width;
    const y = histH - (expected / maxC) * histH;
    mb.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  }

  return (
    <div>
      <svg width={width} height={boxH} role="img" aria-label="Gas particles">
        <rect
          x={0.5}
          y={0.5}
          width={width - 1}
          height={boxH - 1}
          fill="none"
          stroke="currentColor"
          strokeWidth={1}
          opacity={0.4}
        />
        <Group>
          {ps.map((p, i) => {
            const v = Math.hypot(p.vx, p.vy);
            return (
              <circle
                key={i}
                cx={p.x}
                cy={p.y}
                r={RADIUS}
                fill={`hsl(${speedHue(v)}, 70%, 50%)`}
                opacity={0.9}
              />
            );
          })}
        </Group>
      </svg>
      <svg width={width} height={histH + 24} style={{ marginTop: 4 }}>
        <Group>
          {counts.map((c, i) => {
            const bw = width / HIST_BINS;
            const bh = (c / maxC) * histH;
            return (
              <rect
                key={i}
                x={i * bw + 1}
                y={histH - bh}
                width={bw - 2}
                height={bh}
                fill="currentColor"
                opacity={0.22}
              />
            );
          })}
          <polyline
            points={mb.join(" ")}
            fill="none"
            stroke="currentColor"
            strokeWidth={1.5}
            opacity={0.9}
          />
        </Group>
        <text
          x={4}
          y={histH + 16}
          fontSize={11}
          fontFamily="ui-monospace, monospace"
          fill="currentColor"
          opacity={0.7}
        >
          speed →    T = {T.toFixed(0)}    ⟨KE⟩ = {meanKE.toFixed(0)}
        </text>
      </svg>
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

export default function Temperature() {
  const scaleSignalRef = useRef({ v: 1 });
  const [resetVersion, setResetVersion] = useState(0);
  return (
    <div style={{ width: "100%" }}>
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <button onClick={() => { scaleSignalRef.current.v *= 1.2; }} style={btn}>heat +</button>
        <button onClick={() => { scaleSignalRef.current.v *= 0.83; }} style={btn}>cool −</button>
        <button onClick={() => setResetVersion((v) => v + 1)} style={btn}>reset</button>
      </div>
      <ParentSize>{({ width }) => <Demo width={width} scaleSignal={scaleSignalRef.current} resetVersion={resetVersion} />}</ParentSize>
    </div>
  );
}
