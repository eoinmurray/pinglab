import { useEffect, useRef } from "react";

const RAMP = " .`'\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
const PALETTE_LIGHT = {
  bg: "#fafafa",
  bgChar: "#d8d8dc",
  e: "#52525b",
  i: "#e57373",
};
const PALETTE_DARK = {
  bg: "#15140f",
  bgChar: "#3a3833",
  e: "#c8c4b8",
  i: "#c75050",
};

function currentPalette() {
  if (typeof document === "undefined") return PALETTE_LIGHT;
  return document.documentElement.dataset.theme === "dark"
    ? PALETTE_DARK
    : PALETTE_LIGHT;
}

type Props = {
  /** Gamma-band frequency in Hz of the E volley. */
  gammaHz?: number;
  /** Phase lag (radians) of I behind E. */
  iLagRad?: number;
  /** Fraction of vertical extent occupied by the I band. */
  iBand?: [number, number];
  /** Fraction of vertical extent occupied by the E band. */
  eBand?: [number, number];
  /** Background sparsity floor. */
  baseOpacity?: number;
  /** Font size px. */
  fontSize?: number;
  /** Frame throttle ms. */
  frameMs?: number;
  /** CSS height of the canvas host. Default 360px. */
  height?: string;
  /** Border radius CSS value. Default "8px". Pass "0" for an edge-to-edge banner. */
  borderRadius?: string;
  /** Outer margin CSS value. Default "1.5rem 0 2rem". Pass "0" for an edge-to-edge banner. */
  margin?: string;
};

export default function PingAsciiHero({
  gammaHz = 40,
  iLagRad = Math.PI / 3,
  eBand = [0.20, 0.60],
  iBand = [0.70, 0.80],
  baseOpacity = 0.07,
  fontSize = 12,
  frameMs = 50,
  height = "360px",
  borderRadius = "8px",
  margin = "1.5rem 0 2rem",
}: Props) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const host = hostRef.current;
    const canvas = canvasRef.current;
    if (!host || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let lastFrame = 0;
    let cols = 0, rows = 0, cellW = 0, cellH = 0, dpr = 1;
    const mouse = { x: -9999, y: -9999, inside: false };

    const resize = () => {
      const rect = host.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.font = `${fontSize}px JetBrains Mono, ui-monospace, monospace`;
      ctx.textBaseline = "top";
      const m = ctx.measureText("M");
      cellW = m.width || fontSize * 0.6;
      cellH = fontSize * 1.15;
      cols = Math.max(1, Math.floor(rect.width / cellW));
      rows = Math.max(1, Math.floor(rect.height / cellH));
    };

    const render = (t: number) => {
      if (t - lastFrame < frameMs) {
        raf = requestAnimationFrame(render);
        return;
      }
      lastFrame = t;
      if (cols === 0 || rows === 0) {
        resize();
        raf = requestAnimationFrame(render);
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const time = t * 0.001;
      const omega = 2 * Math.PI * gammaHz * 0.08;
      const ePhase = time * omega;
      const iPhase = ePhase - iLagRad;
      const rampMax = RAMP.length - 1;
      const palette = currentPalette();
      host.style.background = palette.bg;

      ctx.clearRect(0, 0, rect.width, rect.height);

      const marginRows = Math.round(eBand[0] * rows);
      const eHeight = Math.round((eBand[1] - eBand[0]) * rows);
      const iHeight = Math.round((iBand[1] - iBand[0]) * rows);
      const eYTop = marginRows;
      const eYBot = eYTop + eHeight;
      const iYBot = rows - marginRows;
      const iYTop = iYBot - iHeight;

      const cx = (mouse.x - rect.left) / cellW;
      const cy = (mouse.y - rect.top) / cellH;

      for (let y = 0; y < rows; y++) {
        const inE = y >= eYTop && y < eYBot;
        const inI = y >= iYTop && y < iYBot;
        const phase = inI ? iPhase : ePhase;
        const color = inI ? palette.i : palette.e;

        for (let x = 0; x < cols; x++) {
          let intensity: number;
          if (inE || inI) {
            const k = (x / cols) * (Math.PI * 2 * 6);
            const travelling = Math.cos(phase - k);
            const volley = Math.max(0, travelling);
            const sharp = Math.pow(volley, 4);
            const jitter = 0.08 * Math.sin(x * 1.7 + y * 0.9 + time * 3);
            intensity = sharp + jitter;
          } else {
            intensity = 0.05 + 0.05 * Math.sin(x * 0.3 + y * 0.5 + time);
          }

          if (mouse.inside) {
            const dx = x - cx;
            const dy = (y - cy) * 1.6;
            const d2 = dx * dx + dy * dy;
            intensity += 0.9 * Math.exp(-d2 / 60);
          }

          intensity = Math.max(0, Math.min(1, intensity));
          const ch = RAMP[Math.floor(intensity * rampMax)];
          if (ch === " ") continue;

          let alpha = baseOpacity + (1 - baseOpacity) * intensity;
          if (alpha > 1) alpha = 1;
          if (alpha <= 0.02) continue;

          ctx.globalAlpha = alpha;
          ctx.fillStyle = (inE || inI) ? color : palette.bgChar;
          ctx.fillText(ch, x * cellW, y * cellH);
        }
      }
      ctx.globalAlpha = 1;
      raf = requestAnimationFrame(render);
    };

    const onMove = (e: MouseEvent) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
      const r = canvas.getBoundingClientRect();
      const margin = 24;
      mouse.inside =
        e.clientX >= r.left - margin && e.clientX <= r.right + margin &&
        e.clientY >= r.top - margin && e.clientY <= r.bottom + margin;
    };

    const ro = new ResizeObserver(resize);
    ro.observe(host);
    resize();
    window.addEventListener("mousemove", onMove, { passive: true });
    raf = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener("mousemove", onMove);
    };
  }, [gammaHz, iLagRad, eBand, iBand, baseOpacity, fontSize, frameMs]);

  return (
    <div
      ref={hostRef}
      style={{
        position: "relative",
        width: "100%",
        height,
        background: "var(--paper, #fafafa)",
        borderRadius,
        overflow: "hidden",
        margin,
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
      />
    </div>
  );
}
