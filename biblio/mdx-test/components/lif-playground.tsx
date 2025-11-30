import { useState, useMemo } from 'react'
import { LinePath } from '@visx/shape'
import { scaleLinear } from '@visx/scale'
import { AxisLeft, AxisBottom } from '@visx/axis'
import { Grid } from '@visx/grid'
import { Group } from '@visx/group'

const width = 600
const height = 400
const margin = { top: 20, right: 20, bottom: 50, left: 60 }

const innerWidth = width - margin.left - margin.right
const innerHeight = height - margin.top - margin.bottom

type Point = { t: number; V: number }

function simulateLIF(
  I: number,
  tau: number,
  Vrest: number,
  Vthresh: number,
  dt: number,
  duration: number
): Point[] {
  const points: Point[] = []
  let V = Vrest

  for (let t = 0; t < duration; t += dt) {
    // LIF dynamics: tau * dV/dt = -(V - Vrest) + R*I
    // Simplified with R=1: dV/dt = (-(V - Vrest) + I) / tau
    const dV = (-(V - Vrest) + I) / tau
    V += dV * dt

    // Spike and reset
    if (V >= Vthresh) {
      points.push({ t, V: Vthresh + 0.5 }) // spike peak
      V = Vrest
    }

    points.push({ t, V })
  }

  return points
}

export function LIFPlayground() {
  const [current, setCurrent] = useState(1.5)
  const [tau, setTau] = useState(10)

  const Vrest = -70
  const Vthresh = -55
  const dt = 0.1
  const duration = 100

  const data = useMemo(
    () => simulateLIF(current, tau, Vrest, Vthresh, dt, duration),
    [current, tau]
  )

  const xScale = scaleLinear({
    domain: [0, duration],
    range: [0, innerWidth],
  })

  const yScale = scaleLinear({
    domain: [-75, -50],
    range: [innerHeight, 0],
  })

  return (
    <div className="space-y-4">
      <div className="flex gap-6">
        <label className="flex flex-col gap-1">
          <span className="text-sm text-zinc-400">Input Current: {current.toFixed(1)}</span>
          <input
            type="range"
            min="0"
            max="3"
            step="0.1"
            value={current}
            onChange={(e) => setCurrent(parseFloat(e.target.value))}
            className="w-40"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-sm text-zinc-400">Tau (ms): {tau}</span>
          <input
            type="range"
            min="5"
            max="30"
            step="1"
            value={tau}
            onChange={(e) => setTau(parseInt(e.target.value))}
            className="w-40"
          />
        </label>
      </div>

      <svg width={width} height={height}>
        <Group left={margin.left} top={margin.top}>
          <Grid
            xScale={xScale}
            yScale={yScale}
            width={innerWidth}
            height={innerHeight}
            stroke="rgba(255,255,255,0.1)"
          />

          {/* Threshold line */}
          <line
            x1={0}
            x2={innerWidth}
            y1={yScale(Vthresh)}
            y2={yScale(Vthresh)}
            stroke="rgba(239, 68, 68, 0.5)"
            strokeDasharray="4,4"
          />

          <LinePath
            data={data}
            x={(d) => xScale(d.t)}
            y={(d) => yScale(d.V)}
            stroke="#3b82f6"
            strokeWidth={1.5}
          />

          <AxisLeft
            scale={yScale}
            stroke="#71717a"
            tickStroke="#71717a"
            tickLabelProps={{ fill: '#a1a1aa', fontSize: 11 }}
            label="Membrane Potential (mV)"
            labelProps={{ fill: '#a1a1aa', fontSize: 12, textAnchor: 'middle' }}
          />

          <AxisBottom
            scale={xScale}
            top={innerHeight}
            stroke="#71717a"
            tickStroke="#71717a"
            tickLabelProps={{ fill: '#a1a1aa', fontSize: 11 }}
            label="Time (ms)"
            labelProps={{ fill: '#a1a1aa', fontSize: 12, textAnchor: 'middle' }}
          />
        </Group>
      </svg>
    </div>
  )
}
