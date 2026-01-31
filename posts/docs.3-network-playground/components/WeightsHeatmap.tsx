import { useEffect, useMemo, useRef } from "react";

type WeightsHeatmapProps = {
  width: number;
  height: number;
  matrix: number[][];
};

export default function WeightsHeatmap({ width, height, matrix }: WeightsHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rows = matrix.length;
  const cols = rows ? matrix[0].length : 0;
  const scale = useMemo(() => {
    const values: number[] = [];
    for (let i = 0; i < rows; i += 1) {
      const row = matrix[i] || [];
      for (let j = 0; j < row.length; j += 1) {
        const v = row[j];
        if (Number.isFinite(v)) values.push(v as number);
      }
    }
    if (!values.length) {
      return { min: 0, max: 1 };
    }
    values.sort((a, b) => a - b);
    const lo = Math.floor(values.length * 0.01);
    const hi = Math.floor(values.length * 0.99);
    const min = values[Math.max(0, Math.min(lo, values.length - 1))] ?? 0;
    const max = values[Math.max(0, Math.min(hi, values.length - 1))] ?? 1;
    if (max <= min) {
      return { min, max };
    }
    return { min, max };
  }, [matrix, rows]);

  if (!rows || !cols || width <= 0 || height <= 0) {
    return (
      <div className="flex h-full w-full items-center justify-center text-xs text-slate-500 dark:text-zinc-400">
        No weights
      </div>
    );
  }

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    canvas.width = Math.max(1, Math.floor(width));
    canvas.height = Math.max(1, Math.floor(height));
    const image = ctx.createImageData(cols, rows);
    for (let i = 0; i < rows; i += 1) {
      for (let j = 0; j < cols; j += 1) {
        const v = matrix[i][j] ?? 0;
        const range = scale.max - scale.min;
        const raw = range <= 1e-12 ? 0 : (v - scale.min) / range;
        const t = Math.max(0, Math.min(1, raw));
        const gamma = 0.35;
        const eased = Math.pow(t, gamma);
        const shade = Math.round(20 + eased * 235);
        const idx = (i * cols + j) * 4;
        image.data[idx] = shade;
        image.data[idx + 1] = shade;
        image.data[idx + 2] = shade;
        image.data[idx + 3] = 255;
      }
    }
    ctx.imageSmoothingEnabled = false;
    ctx.putImageData(image, 0, 0);
    ctx.drawImage(canvas, 0, 0, cols, rows, 0, 0, canvas.width, canvas.height);
  }, [width, height, rows, cols, matrix, scale]);

  return (
    <canvas ref={canvasRef} width={width} height={height} role="img" aria-label="Weights heatmap" />
  );
}
