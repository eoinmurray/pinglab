import { useEffect, useRef, useState } from "react";
import { Group } from "@visx/group";
import { ParentSize } from "@visx/responsive";

type Particle = { x: number; y: number; vx: number; vy: number };

const N = 24;
const RADIUS = 4;
const SPEED = 80;

function init(width: number, height: number): Particle[] {
  const ps: Particle[] = [];
  for (let i = 0; i < N; i++) {
    const angle = Math.random() * Math.PI * 2;
    ps.push({
      x: RADIUS + Math.random() * (width - 2 * RADIUS),
      y: RADIUS + Math.random() * (height - 2 * RADIUS),
      vx: Math.cos(angle) * SPEED,
      vy: Math.sin(angle) * SPEED,
    });
  }
  return ps;
}

function step(particles: Particle[], dt: number, width: number, height: number) {
  for (const p of particles) {
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    if (p.x < RADIUS) { p.x = RADIUS; p.vx = -p.vx; }
    else if (p.x > width - RADIUS) { p.x = width - RADIUS; p.vx = -p.vx; }
    if (p.y < RADIUS) { p.y = RADIUS; p.vy = -p.vy; }
    else if (p.y > height - RADIUS) { p.y = height - RADIUS; p.vy = -p.vy; }
  }
}

function Sim({ width, height, resetVersion }: { width: number; height: number; resetVersion: number }) {
  const particlesRef = useRef<Particle[] | null>(null);
  const lastTimeRef = useRef<number>(0);
  const [, setTick] = useState(0);

  useEffect(() => {
    if (width === 0 || height === 0) return;
    particlesRef.current = init(width, height);
    lastTimeRef.current = 0;

    let raf = 0;
    const loop = (t: number) => {
      const dt = lastTimeRef.current === 0 ? 0 : Math.min((t - lastTimeRef.current) / 1000, 0.05);
      lastTimeRef.current = t;
      if (particlesRef.current) step(particlesRef.current, dt, width, height);
      setTick((n) => n + 1);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [width, height, resetVersion]);

  const particles = particlesRef.current ?? [];

  return (
    <svg width={width} height={height} role="img" aria-label="Particles in a box">
      <rect x={0.5} y={0.5} width={width - 1} height={height - 1} fill="none" stroke="currentColor" strokeWidth={1} opacity={0.4} />
      <Group>
        {particles.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r={RADIUS} fill="currentColor" opacity={0.85} />
        ))}
      </Group>
    </svg>
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

export default function ParticlesInBox() {
  const [resetVersion, setResetVersion] = useState(0);
  return (
    <div style={{ width: "100%" }}>
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <button onClick={() => setResetVersion((v) => v + 1)} style={btn}>reset</button>
      </div>
      <div style={{ width: "100%", aspectRatio: "16 / 9" }}>
        <ParentSize>{({ width, height }) => <Sim width={width} height={height} resetVersion={resetVersion} />}</ParentSize>
      </div>
    </div>
  );
}
