import { AxisBottom, AxisLeft } from "@visx/axis";
import { LinePath } from "@visx/shape";
import { scaleLinear } from "@visx/scale";
import { useMemo } from "react";

type PopulationRatePlotProps = {
  width: number;
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
  innerWidth: number;
  innerHeight: number;
  tMs: number[];
  rateHzE: number[];
  rateHzI?: number[];
  maxTMs?: number;
};

export default function PopulationRatePlot({
  width,
  height,
  margin,
  innerWidth,
  innerHeight,
  tMs,
  rateHzE,
  rateHzI,
  maxTMs,
}: PopulationRatePlotProps) {
  const xScale = useMemo(() => {
    const inferredMax = tMs.length ? Math.max(...tMs) : 1;
    const maxT = maxTMs ?? inferredMax;
    return scaleLinear({
      domain: [0, maxT],
      range: [0, innerWidth],
    });
  }, [tMs, innerWidth]);

  const yScale = useMemo(() => {
    const maxE = rateHzE.length ? Math.max(...rateHzE) : 0;
    const maxI = rateHzI?.length ? Math.max(...rateHzI) : 0;
    const maxRate = Math.max(maxE, maxI, 1);
    return scaleLinear({
      domain: [0, maxRate],
      range: [innerHeight, 0],
      nice: true,
    });
  }, [rateHzE, rateHzI, innerHeight]);

  const pointsE = useMemo(() => {
    return tMs.map((t, i) => ({ t, r: rateHzE[i] ?? 0 }));
  }, [tMs, rateHzE]);

  const pointsI = useMemo(() => {
    if (!rateHzI) {
      return [];
    }
    return tMs.map((t, i) => ({ t, r: rateHzI[i] ?? 0 }));
  }, [tMs, rateHzI]);

  return (
    <div className="relative" style={{ width, height }}>
      <svg width={width} height={height} role="img" aria-label="Population rate">
        <g transform={`translate(${margin.left},${margin.top})`}>
          <LinePath
            data={pointsE}
            x={(d) => xScale(d.t) ?? 0}
            y={(d) => yScale(d.r) ?? 0}
            stroke="currentColor"
            strokeWidth={1.4}
          />
          {pointsI.length ? (
            <LinePath
              data={pointsI}
              x={(d) => xScale(d.t) ?? 0}
              y={(d) => yScale(d.r) ?? 0}
              stroke="#dc2626"
              strokeWidth={1.4}
            />
          ) : null}
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
            label="Rate (Hz)"
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
