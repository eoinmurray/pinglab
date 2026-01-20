import { AxisBottom, AxisLeft } from "@visx/axis";
import { LinePath } from "@visx/shape";
import { scaleLinear } from "@visx/scale";
import { useMemo } from "react";

type MembranePotentialPlotProps = {
  width: number;
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
  innerWidth: number;
  innerHeight: number;
  tMs: number[];
  vE?: number[];
  vI?: number[];
  maxTMs?: number;
};

export default function MembranePotentialPlot({
  width,
  height,
  margin,
  innerWidth,
  innerHeight,
  tMs,
  vE,
  vI,
  maxTMs,
}: MembranePotentialPlotProps) {
  const xScale = useMemo(() => {
    const inferredMax = tMs.length ? Math.max(...tMs) : 1;
    const maxT = maxTMs ?? inferredMax;
    return scaleLinear({
      domain: [0, maxT],
      range: [0, innerWidth],
    });
  }, [tMs, innerWidth, maxTMs]);

  const yScale = useMemo(() => {
    const values = [...(vE ?? []), ...(vI ?? [])];
    const minV = values.length ? Math.min(...values) : -80;
    const maxV = values.length ? Math.max(...values) : 40;
    return scaleLinear({
      domain: [minV, maxV],
      range: [innerHeight, 0],
      nice: true,
    });
  }, [vE, vI, innerHeight]);

  const pointsE = useMemo(() => {
    if (!vE || vE.length !== tMs.length) {
      return [];
    }
    return tMs.map((t, i) => ({ t, v: vE[i] ?? 0 }));
  }, [tMs, vE]);
  const pointsI = useMemo(() => {
    if (!vI || vI.length !== tMs.length) {
      return [];
    }
    return tMs.map((t, i) => ({ t, v: vI[i] ?? 0 }));
  }, [tMs, vI]);

  return (
    <div className="relative" style={{ width, height }}>
      <svg width={width} height={height} role="img" aria-label="Membrane potential">
        <g transform={`translate(${margin.left},${margin.top})`}>
          {pointsE.length ? (
            <LinePath
              data={pointsE}
              x={(d) => xScale(d.t) ?? 0}
              y={(d) => yScale(d.v) ?? 0}
              stroke="currentColor"
              strokeWidth={1.2}
            />
          ) : null}
          {pointsI.length ? (
            <LinePath
              data={pointsI}
              x={(d) => xScale(d.t) ?? 0}
              y={(d) => yScale(d.v) ?? 0}
              stroke="#dc2626"
              strokeWidth={1.2}
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
            label="Membrane (mV)"
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
