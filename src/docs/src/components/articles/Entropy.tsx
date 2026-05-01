import { useEffect, useRef, useState } from "react";
import { Group } from "@visx/group";
import { ParentSize } from "@visx/responsive";

type Particle = { x: number; y: number; vx: number; vy: number };
type EntropySample = { t: number; H: number };

const N = 80;
const RADIUS = 4;
const SPEED = 90;
const BOX_H = 220;
const CELLS_X = 8;
const CELLS_Y = 4;
const M = CELLS_X * CELLS_Y;
const H_MAX = Math.log(M);
const SAMPLE_HZ = 10;
const HISTORY_S = 25;

function initParticles(w: number, h: number, leftFrac: number): Particle[] {
  const ps: Particle[] = [];
  const halfW = w * leftFrac;
  const cols = Math.ceil(Math.sqrt((N * halfW) / h));
  const rows = Math.ceil(N / cols);
  const dx = halfW / (cols + 1);
  const dy = h / (rows + 1);
  let i = 0;
  for (let r = 0; r < rows && i < N; r++) {
    for (let c = 0; c < cols && i < N; c++, i++) {
      const a = Math.random() * Math.PI * 2;
      ps.push({
        x: dx * (c + 1),
        y: dy * (r + 1),
        vx: Math.cos(a) * SPEED,
        vy: Math.sin(a) * SPEED,
      });
    }
  }
  return ps;
}

function step(ps: Particle[], dt: number, w: number, h: number, partitionX: number | null) {
  for (const p of ps) {
    const xPrev = p.x;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    if (p.x < RADIUS) { p.x = RADIUS; p.vx = -p.vx; }
    else if (p.x > w - RADIUS) { p.x = w - RADIUS; p.vx = -p.vx; }
    if (p.y < RADIUS) { p.y = RADIUS; p.vy = -p.vy; }
    else if (p.y > h - RADIUS) { p.y = h - RADIUS; p.vy = -p.vy; }
    if (partitionX !== null) {
      if (xPrev <= partitionX && p.x > partitionX - RADIUS) {
        p.x = partitionX - RADIUS;
        p.vx = -Math.abs(p.vx);
      } else if (xPrev >= partitionX && p.x < partitionX + RADIUS) {
        p.x = partitionX + RADIUS;
        p.vx = Math.abs(p.vx);
      }
    }
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

function shannonH(ps: Particle[], w: number, h: number): number {
  const counts = new Array(M).fill(0);
  for (const p of ps) {
    const cx = Math.min(CELLS_X - 1, Math.floor((p.x / w) * CELLS_X));
    const cy = Math.min(CELLS_Y - 1, Math.floor((p.y / h) * CELLS_Y));
    counts[cy * CELLS_X + cx]++;
  }
  const total = ps.length;
  let H = 0;
  for (const c of counts) {
    if (c === 0) continue;
    const p = c / total;
    H -= p * Math.log(p);
  }
  return H;
}

function Demo({ width, releasedRef, released, resetVersion }: {
  width: number;
  releasedRef: { current: boolean };
  released: boolean;
  resetVersion: number;
}) {
  const psRef = useRef<Particle[] | null>(null);
  const samplesRef = useRef<EntropySample[]>([]);
  const startTRef = useRef(0);
  const lastTRef = useRef(0);
  const lastSampleTRef = useRef(0);
  const [, setTick] = useState(0);

  useEffect(() => {
    if (!width) return;
    psRef.current = initParticles(width, BOX_H, 0.5);
    samplesRef.current = [];
    startTRef.current = 0;
    lastTRef.current = 0;
    lastSampleTRef.current = 0;
    let raf = 0;
    const loop = (t: number) => {
      if (startTRef.current === 0) startTRef.current = t;
      const dt = lastTRef.current === 0 ? 0 : Math.min((t - lastTRef.current) / 1000, 0.05);
      lastTRef.current = t;
      const now = (t - startTRef.current) / 1000;
      const partitionX = releasedRef.current ? null : width / 2;
      if (psRef.current) {
        step(psRef.current, dt, width, BOX_H, partitionX);
        if (now - lastSampleTRef.current >= 1 / SAMPLE_HZ) {
          lastSampleTRef.current = now;
          const H = shannonH(psRef.current, width, BOX_H);
          samplesRef.current.push({ t: now, H });
          while (samplesRef.current.length > 0 && samplesRef.current[0].t < now - HISTORY_S) {
            samplesRef.current.shift();
          }
        }
      }
      setTick((n) => n + 1);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [width, releasedRef, resetVersion]);

  const ps = psRef.current ?? [];
  const samples = samplesRef.current;
  const Hnow = samples.length > 0 ? samples[samples.length - 1].H : 0;

  const plotH = 90;
  const tNow = lastTRef.current === 0 ? 0 : (lastTRef.current - startTRef.current) / 1000;
  const tMin = Math.max(0, tNow - HISTORY_S);
  const tSpan = Math.max(1, tNow - tMin);
  const points = samples.map((s) => {
    const x = ((s.t - tMin) / tSpan) * width;
    const y = plotH - (s.H / H_MAX) * plotH;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");

  return (
    <div>
      <svg width={width} height={BOX_H} role="img" aria-label="Entropy demo">
        <rect
          x={0.5}
          y={0.5}
          width={width - 1}
          height={BOX_H - 1}
          fill="none"
          stroke="currentColor"
          strokeWidth={1}
          opacity={0.4}
        />
        {!released && (
          <line
            x1={width / 2}
            y1={0}
            x2={width / 2}
            y2={BOX_H}
            stroke="currentColor"
            strokeWidth={2}
            opacity={0.5}
            strokeDasharray="4 4"
          />
        )}
        <Group opacity={0.15} stroke="currentColor">
          {Array.from({ length: CELLS_X - 1 }).map((_, i) => (
            <line key={`vx${i}`} x1={((i + 1) / CELLS_X) * width} y1={0} x2={((i + 1) / CELLS_X) * width} y2={BOX_H} />
          ))}
          {Array.from({ length: CELLS_Y - 1 }).map((_, i) => (
            <line key={`hy${i}`} x1={0} y1={((i + 1) / CELLS_Y) * BOX_H} x2={width} y2={((i + 1) / CELLS_Y) * BOX_H} />
          ))}
        </Group>
        <Group>
          {ps.map((p, i) => (
            <circle key={i} cx={p.x} cy={p.y} r={RADIUS} fill="currentColor" opacity={0.85} />
          ))}
        </Group>
      </svg>
      <svg width={width} height={plotH + 24} style={{ marginTop: 4 }}>
        <line
          x1={0}
          y1={0}
          x2={width}
          y2={0}
          stroke="currentColor"
          strokeDasharray="3 3"
          opacity={0.35}
        />
        <text x={4} y={10} fontSize={10} fontFamily="ui-monospace, monospace" fill="currentColor" opacity={0.6}>
          H_max = log({M}) = {H_MAX.toFixed(2)}
        </text>
        {points && (
          <polyline points={points} fill="none" stroke="currentColor" strokeWidth={1.5} opacity={0.9} />
        )}
        <text x={4} y={plotH + 16} fontSize={11} fontFamily="ui-monospace, monospace" fill="currentColor" opacity={0.7}>
          H(t) →    H = {Hnow.toFixed(2)}    H/H_max = {(Hnow / H_MAX).toFixed(2)}
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

export default function Entropy() {
  const [released, setReleased] = useState(false);
  const releasedRef = useRef(false);
  releasedRef.current = released;
  const [resetVersion, setResetVersion] = useState(0);
  return (
    <div style={{ width: "100%" }}>
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <button
          onClick={() => setReleased(true)}
          disabled={released}
          style={{ ...btn, opacity: released ? 0.4 : 0.8, cursor: released ? "default" : "pointer" }}
        >
          remove partition
        </button>
        <button
          onClick={() => {
            setReleased(false);
            setResetVersion((v) => v + 1);
          }}
          style={btn}
        >
          reset
        </button>
      </div>
      <ParentSize>
        {({ width }) => (
          <Demo width={width} releasedRef={releasedRef} released={released} resetVersion={resetVersion} />
        )}
      </ParentSize>
    </div>
  );
}
