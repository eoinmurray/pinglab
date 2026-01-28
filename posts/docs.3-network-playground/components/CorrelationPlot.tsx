import { AxisBottom, AxisLeft } from "@visx/axis";
import { LinePath } from "@visx/shape";
import { scaleLinear } from "@visx/scale";
import { useMemo } from "react";

type CorrelationPlotProps = {
  width: number;
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
  innerWidth: number;
  innerHeight: number;
  lagsMs: number[];
  values: number[];
  xLabel: string;
  yLabel: string;
  color?: string;
  yMin?: number;
  yMax?: number;
};

export default function CorrelationPlot({
  width,
  height,
  margin,
  innerWidth,
  innerHeight,
  lagsMs,
  values,
  xLabel,
  yLabel,
  color = "#0f172a",
  yMin,
  yMax,
}: CorrelationPlotProps) {
  const xScale = useMemo(() => {
    if (!lagsMs.length) {
      return scaleLinear({ domain: [0, 1], range: [0, innerWidth] });
    }
    const minLag = Math.min(...lagsMs);
    const maxLag = Math.max(...lagsMs);
    return scaleLinear({
      domain: [minLag, maxLag],
      range: [0, innerWidth],
    });
  }, [lagsMs, innerWidth]);

  const yScale = useMemo(() => {
    if (!values.length) {
      return scaleLinear({ domain: [0, 1], range: [innerHeight, 0] });
    }
    const minVal = yMin ?? Math.min(...values);
    const maxVal = yMax ?? Math.max(...values);
    const pad = (maxVal - minVal) * 0.05;
    return scaleLinear({
      domain: [minVal - pad, maxVal + pad],
      range: [innerHeight, 0],
      nice: true,
    });
  }, [values, yMin, yMax, innerHeight]);

  const points = useMemo(
    () => lagsMs.map((t, i) => ({ t, v: values[i] ?? 0 })),
    [lagsMs, values]
  );

  return (
    <div className="relative" style={{ width, height }}>
      <svg width={width} height={height} role="img" aria-label={yLabel}>
        <g transform={`translate(${margin.left},${margin.top})`}>
          <LinePath
            data={points}
            x={(d) => xScale(d.t) ?? 0}
            y={(d) => yScale(d.v) ?? 0}
            stroke={color}
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
            label={xLabel}
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
            label={yLabel}
            labelProps={{
              fill: "currentColor",
              fontSize: 10,
              textAnchor: "middle",
            }}
          />
        </g>
      </svg>
    </div>
  );
}
