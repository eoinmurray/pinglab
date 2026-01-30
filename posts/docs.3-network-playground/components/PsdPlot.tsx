import { AxisBottom, AxisLeft } from "@visx/axis";
import { LinePath } from "@visx/shape";
import { scaleLinear } from "@visx/scale";
import { useMemo } from "react";

type PsdPlotProps = {
  width: number;
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
  innerWidth: number;
  innerHeight: number;
  freqsHz: number[];
  power: number[];
};

export default function PsdPlot({
  width,
  height,
  margin,
  innerWidth,
  innerHeight,
  freqsHz,
  power,
}: PsdPlotProps) {
  const maxFreq = freqsHz.length ? Math.max(...freqsHz) : 1;
  const maxPower = power.length ? Math.max(...power) : 1;

  const xScale = useMemo(
    () =>
      scaleLinear({
        domain: [0, maxFreq],
        range: [0, innerWidth],
      }),
    [maxFreq, innerWidth]
  );

  const yScale = useMemo(
    () =>
      scaleLinear({
        domain: [0, maxPower],
        range: [innerHeight, 0],
        nice: true,
      }),
    [maxPower, innerHeight]
  );

  const series = useMemo(
    () => freqsHz.map((f, i) => ({ f, p: power[i] ?? 0 })),
    [freqsHz, power]
  );

  return (
    <div className="relative" style={{ width, height }}>
      <svg width={width} height={height} role="img" aria-label="PSD">
        <g transform={`translate(${margin.left},${margin.top})`}>
          <LinePath
            data={series}
            x={(d) => xScale(d.f) ?? 0}
            y={(d) => yScale(d.p) ?? 0}
            stroke="currentColor"
            strokeWidth={1.2}
          />
          <AxisBottom
            top={innerHeight}
            scale={xScale}
            stroke="currentColor"
            tickStroke="currentColor"
            tickLabelProps={() => ({
              fill: "currentColor",
              fontSize: 10,
              textAnchor: "middle",
            })}
            label="Frequency (Hz)"
            labelProps={{
              fill: "currentColor",
              fontSize: 11,
              textAnchor: "middle",
            }}
          />
          <AxisLeft
            scale={yScale}
            stroke="currentColor"
            tickStroke="currentColor"
            tickLabelProps={() => ({
              fill: "currentColor",
              fontSize: 10,
              textAnchor: "end",
              dx: "-0.25em",
            })}
            label="PSD"
            labelProps={{
              fill: "currentColor",
              fontSize: 11,
              textAnchor: "middle",
            }}
          />
        </g>
      </svg>
    </div>
  );
}
