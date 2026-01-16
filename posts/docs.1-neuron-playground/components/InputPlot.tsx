import { LinePath } from "@visx/shape";
import { scaleLinear } from "@visx/scale";
import { AxisBottom, AxisLeft } from "@visx/axis";

type Margin = {
  top: number;
  right: number;
  bottom: number;
  left: number;
};

type InputPlotProps = {
  times: number[];
  input: number[];
  T: number;
  width: number;
  height: number;
  margin: Margin;
  yDomain: [number, number];
};

export default function InputPlot({
  times,
  input,
  T,
  width,
  height,
  margin,
  yDomain,
}: InputPlotProps) {
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const xScale = scaleLinear({
    domain: [0, T],
    range: [0, innerWidth],
  });
  const inputScale = scaleLinear({
    domain: yDomain,
    range: [innerHeight, 0],
  });

  return (
    <div className="h-[120px] rounded-xl border border-black bg-white p-2 dark:border-zinc-100 dark:bg-black">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        height="100%"
        role="img"
        aria-label="Input current trace"
        className="block text-black dark:text-zinc-100"
      >
        <g transform={`translate(${margin.left},${margin.top})`}>
          <rect x={0} y={0} width={innerWidth} height={innerHeight} fill="none" rx={12} />
          <LinePath
            data={times}
            x={(d) => xScale(d) ?? 0}
            y={(d, i) => inputScale(input[i]) ?? 0}
            stroke="currentColor"
            strokeWidth={1.8}
          />
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
            scale={inputScale}
            stroke="currentColor"
            tickStroke="currentColor"
            tickLabelProps={() => ({
              fill: "currentColor",
              fontSize: 11,
              textAnchor: "end",
              dx: "-0.25em",
            })}
            label="I_ext"
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
