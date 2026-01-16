import { LinePath } from "@visx/shape";
import { scaleLinear } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";

type Margin = {
  top: number;
  right: number;
  bottom: number;
  left: number;
};

type VoltagePlotProps = {
  times: number[];
  voltages: number[];
  T: number;
  threshold: number | null;
  yDomain: [number, number];
  width: number;
  height: number;
  margin: Margin;
};

export default function VoltagePlot({
  times,
  voltages,
  T,
  threshold,
  yDomain,
  width,
  height,
  margin,
}: VoltagePlotProps) {
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const xScale = scaleLinear({
    domain: [0, T],
    range: [0, innerWidth],
  });
  const yScale = scaleLinear({
    domain: yDomain,
    range: [innerHeight, 0],
  });

  return (
    <div className="h-[200px] rounded-xl border border-black bg-white p-2 dark:border-zinc-100 dark:bg-black">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        height="100%"
        role="img"
        aria-label="LIF voltage trace"
        className="block text-black dark:text-zinc-100"
      >
        <g transform={`translate(${margin.left},${margin.top})`}>
          <rect x={0} y={0} width={innerWidth} height={innerHeight} fill="none" rx={12} />
          <LinePath
            data={times}
            x={(d) => xScale(d) ?? 0}
            y={(d, i) => yScale(voltages[i]) ?? 0}
            stroke="currentColor"
            strokeWidth={1}
          />
          {threshold !== null && (
            <LinePath
              data={[0, T]}
              x={(d) => xScale(d) ?? 0}
              y={() => yScale(threshold) ?? 0}
              stroke="currentColor"
              strokeWidth={1.4}
              strokeDasharray="6 6"
            />
          )}
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
            tickLabelProps={() => ({
              fill: "currentColor",
              fontSize: 11,
              textAnchor: "end",
              dx: "-0.25em",
            })}
            label="Membrane Potential (mV)"
            labelProps={{
              fill: "currentColor",
              fontSize: 12,
              textAnchor: "middle",
            }}
          />
        </g>
      </svg>
    </div>
  );
}
