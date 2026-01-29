import { AxisBottom, AxisLeft } from "@visx/axis";
import type { ScaleLinear } from "d3-scale";
import { useEffect, useMemo, useRef } from "react";

type SpikesResponse = {
  times: number[];
  ids: number[];
  types: number[];
};

type RasterPlotProps = {
  width: number;
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
  innerWidth: number;
  innerHeight: number;
  xScale: ScaleLinear<number, number>;
  yScale: ScaleLinear<number, number>;
  spikes: SpikesResponse | null;
};

export default function RasterPlot({
  width,
  height,
  margin,
  innerWidth,
  innerHeight,
  xScale,
  yScale,
  spikes,
}: RasterPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const baseColor = useMemo(() => {
    if (typeof window === "undefined") {
      return "#000";
    }
    return getComputedStyle(document.documentElement).color || "#000";
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !spikes) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(innerWidth * dpr);
    canvas.height = Math.floor(innerHeight * dpr);
    canvas.style.width = `${innerWidth}px`;
    canvas.style.height = `${innerHeight}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, innerWidth, innerHeight);

    const { times, ids, types } = spikes;
    ctx.fillStyle = baseColor;
    for (let i = 0; i < times.length; i += 1) {
      if (types[i] !== 0) continue;
      const x = xScale(times[i]) ?? 0;
      const y = yScale(ids[i]) ?? 0;
      ctx.fillRect(x, y, 1.2, 1.2);
    }
    ctx.fillStyle = "#d9480f";
    for (let i = 0; i < times.length; i += 1) {
      if (types[i] !== 1) continue;
      const x = xScale(times[i]) ?? 0;
      const y = yScale(ids[i]) ?? 0;
      ctx.fillRect(x, y, 1.2, 1.2);
    }
  }, [spikes, innerWidth, innerHeight, xScale, yScale, baseColor]);

  return (
    <div className="relative shrink-0" style={{ width, height }}>
      <div className="absolute left-0 top-0" style={{ transform: `translate(${margin.left}px, ${margin.top}px)` }}>
        <canvas ref={canvasRef} />
      </div>
      <svg width={width} height={height} role="img" aria-label="Raster plot" className="block text-black dark:text-zinc-100">
        <g transform={`translate(${margin.left},${margin.top})`}>
          <rect x={0} y={0} width={innerWidth} height={innerHeight} fill="none" />
          <AxisBottom
            top={innerHeight}
            scale={xScale}
            stroke="currentColor"
            tickStroke="currentColor"
            tickLabelProps={() => ({
              fill: "currentColor",
              fontSize: 11,
              textAnchor: "middle",
            })}
            label="Time (ms)"
            labelProps={{
              fill: "currentColor",
              fontSize: 12,
              textAnchor: "middle",
            }}
          />
          <AxisLeft
            scale={yScale}
            stroke="currentColor"
            tickStroke="currentColor"
            tickFormat={() => ""}
          />
        </g>
      </svg>
    </div>
  );
}
