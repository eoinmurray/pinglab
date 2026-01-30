import { AxisBottom, AxisLeft } from "@visx/axis";
import { LinePath } from "@visx/shape";
import { scaleLinear } from "@visx/scale";
import { useMemo } from "react";

type WeightsHistogramPlotProps = {
  width: number;
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
  innerWidth: number;
  innerHeight: number;
  bins: number[];
  counts: number[];
  color: string;
  label: string;
  yMax?: number;
};

export default function WeightsHistogramPlot({
  width,
  height,
  margin,
  innerWidth,
  innerHeight,
  bins,
  counts,
  color,
  label,
  yMax,
}: WeightsHistogramPlotProps) {
  const maxX = bins.length ? Math.max(...bins) : 1;
  const xScale = useMemo(() => {
    return scaleLinear({
      domain: [0, maxX],
      range: [0, innerWidth],
    });
  }, [maxX, innerWidth]);

  const maxCount = useMemo(() => {
    const localMax = Math.max(1, ...counts);
    if (yMax !== undefined && yMax > 0) {
      return Math.min(localMax, yMax);
    }
    return localMax;
  }, [counts, yMax]);

  const yScale = useMemo(() => {
    return scaleLinear({
      domain: [0, maxCount],
      range: [innerHeight, 0],
      nice: true,
    });
  }, [maxCount, innerHeight]);

  const series = useMemo(
    () => bins.map((b, i) => ({ x: b, y: counts[i] ?? 0 })),
    [bins, counts]
  );
  const seriesColor = color || "currentColor";

  return (
    <div className="relative" style={{ width, height }}>
      <svg width={width} height={height} role="img" aria-label="Weights histogram">
        <g transform={`translate(${margin.left},${margin.top})`}>
          <LinePath
            data={series}
            x={(d) => xScale(d.x) ?? 0}
            y={(d) => yScale(d.y) ?? 0}
            stroke={seriesColor}
            strokeWidth={1.2}
          />
          <AxisBottom
            top={innerHeight}
            scale={xScale}
            stroke="currentColor"
            tickStroke="currentColor"
            numTicks={4}
            tickLabelProps={() => ({
              fill: "currentColor",
              fontSize: 9,
              textAnchor: "middle",
            })}
            label="Weight"
            labelProps={{
              fill: "currentColor",
              fontSize: 10,
              textAnchor: "middle",
            }}
          />
          <AxisLeft
            scale={yScale}
            stroke="currentColor"
            tickStroke="currentColor"
            numTicks={4}
            tickLabelProps={() => ({
              fill: "currentColor",
              fontSize: 9,
              textAnchor: "end",
              dx: "-0.25em",
            })}
            label="Count"
            labelProps={{
              fill: "currentColor",
              fontSize: 10,
              textAnchor: "middle",
            }}
          />
        </g>
      </svg>
      <div
        style={{
          position: "absolute",
          top: 6,
          right: 8,
          display: "inline-flex",
          alignItems: "center",
          gap: "6px",
          fontSize: "10px",
          color: "currentColor",
        }}
      >
        <span style={{ width: 10, height: 10, backgroundColor: seriesColor }} />
        {label}
      </div>
    </div>
  );
}
