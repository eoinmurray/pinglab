import { AxisBottom, AxisLeft } from "@visx/axis";
import { LinePath } from "@visx/shape";
import { scaleLinear } from "@visx/scale";
import { useMemo } from "react";

type InputTracePlotProps = {
  width: number;
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
  innerWidth: number;
  innerHeight: number;
  tMs: number[];
  inputE: number[];
  inputI: number[];
  maxTMs?: number;
};

export default function InputTracePlot({
  width,
  height,
  margin,
  innerWidth,
  innerHeight,
  tMs,
  inputE,
  inputI,
  maxTMs,
}: InputTracePlotProps) {
  const xScale = useMemo(() => {
    const inferredMax = tMs.length ? Math.max(...tMs) : 1;
    const maxT = maxTMs ?? inferredMax;
    return scaleLinear({
      domain: [0, maxT],
      range: [0, innerWidth],
    });
  }, [tMs, innerWidth, maxTMs]);

  const yScale = useMemo(() => {
    const maxE = inputE.length ? Math.max(...inputE) : 0;
    const maxI = inputI.length ? Math.max(...inputI) : 0;
    const maxVal = Math.max(maxE, maxI, 1e-6);
    return scaleLinear({
      domain: [0, maxVal],
      range: [innerHeight, 0],
      nice: true,
    });
  }, [inputE, inputI, innerHeight]);

  const pointsE = useMemo(() => {
    return tMs.map((t, i) => ({ t, v: inputE[i] ?? 0 }));
  }, [tMs, inputE]);
  const pointsI = useMemo(() => {
    return tMs.map((t, i) => ({ t, v: inputI[i] ?? 0 }));
  }, [tMs, inputI]);

  return (
    <div className="relative" style={{ width, height }}>
      <svg width={width} height={height} role="img" aria-label="Input current">
        <g transform={`translate(${margin.left},${margin.top})`}>
          <LinePath
            data={pointsE}
            x={(d) => xScale(d.t) ?? 0}
            y={(d) => yScale(d.v) ?? 0}
            stroke="currentColor"
            strokeWidth={0.9}
          />
          <LinePath
            data={pointsI}
            x={(d) => xScale(d.t) ?? 0}
            y={(d) => yScale(d.v) ?? 0}
            stroke="#dc2626"
            strokeWidth={0.9}
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
            label="Time (ms)"
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
            label="Input"
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
