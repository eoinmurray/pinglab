import { useMemo } from 'react'
import { ParentSize } from '@visx/responsive'
import { scaleLinear } from '@visx/scale'
import { useSimulationContext } from '@/context/simulation-context'
import { formatAxisTick } from '@/lib/axis-format'

type Point = {
  timestep: number
  rate: number
}

const TIMESTEP_MAX = 200

type PlotPopulationRateTraceProps = {
  layerLabel?: string
}

function toPath(points: Point[], x: (v: number) => number, y: (v: number) => number) {
  if (points.length === 0) return ''
  return points
    .map((point, i) => `${i === 0 ? 'M' : 'L'} ${x(point.timestep)} ${y(point.rate)}`)
    .join(' ')
}

export default function PlotPopulationRateTrace({
  layerLabel,
}: PlotPopulationRateTraceProps) {
  const { runData, parameters } = useSimulationContext()
  const fallbackLabels = useMemo(() => {
    const labels = ['L1']
    if (Math.round(parameters.config.N_I.value) > 0) labels.push('I')
    return labels
  }, [parameters.config.N_I.value])
  const layerLabels = runData?.layer_labels?.length ? runData.layer_labels : fallbackLabels
  const activeLayer =
    layerLabel && layerLabels.includes(layerLabel) ? layerLabel : (layerLabels[0] ?? 'L1')

  const pointsE = useMemo(() => {
    if (!runData || !runData.population_rate_t_ms.length) return []
    return runData.population_rate_t_ms.map((timestep, idx) => ({
      timestep,
      rate: runData.population_rate_hz_E?.[idx] ?? 0,
    }))
  }, [runData])

  const pointsI = useMemo(() => {
    if (!runData || !runData.population_rate_t_ms.length) return []
    return runData.population_rate_t_ms.map((timestep, idx) => ({
      timestep,
      rate: runData.population_rate_hz_I?.[idx] ?? 0,
    }))
  }, [runData])

  return (
    <div className="relative flex h-full w-full min-h-0 min-w-0 overflow-hidden rounded-md border bg-background">
      <ParentSize className="h-full w-full min-h-0 min-w-0">
        {({ width, height }) => {
          if (width < 20 || height < 20) return null

          const margin = { top: 20, right: 12, bottom: 30, left: 40 }
          const minTimestep = pointsE[0]?.timestep ?? 0
          const maxTimestep = pointsE[pointsE.length - 1]?.timestep ?? TIMESTEP_MAX
          const maxRate = Math.max(
            1,
            ...pointsE.map((point) => point.rate),
            ...pointsI.map((point) => point.rate)
          )
          const xScale = scaleLinear<number>({
            domain: [minTimestep, maxTimestep],
            range: [margin.left, width - margin.right],
          })
          const yScale = scaleLinear<number>({
            domain: [0, maxRate * 1.05],
            range: [height - margin.bottom, margin.top],
          })
          const xTicks = [
            minTimestep,
            minTimestep + (maxTimestep - minTimestep) * 0.25,
            minTimestep + (maxTimestep - minTimestep) * 0.5,
            minTimestep + (maxTimestep - minTimestep) * 0.75,
            maxTimestep,
          ]
          const yTicks = [0, maxRate * 0.33, maxRate * 0.66, maxRate]
          const axisColor = 'var(--muted-foreground)'
          const pathE = toPath(pointsE, xScale, yScale)
          const pathI = toPath(pointsI, xScale, yScale)

          return (
            <svg width={width} height={height} className="block h-full w-full">
              <rect x={0} y={0} width={width} height={height} fill="var(--background)" />

              <text x={margin.left} y={14} fontSize={11} fill={axisColor}>
                Population rate (active layer selector: {activeLayer})
              </text>

              <line
                x1={margin.left}
                x2={width - margin.right}
                y1={height - margin.bottom}
                y2={height - margin.bottom}
                stroke={axisColor}
                strokeWidth={1}
              />
              <line
                x1={margin.left}
                x2={margin.left}
                y1={margin.top}
                y2={height - margin.bottom}
                stroke={axisColor}
                strokeWidth={1}
              />

              <path d={pathE} fill="none" stroke="var(--foreground)" strokeWidth={1.6} />
              {Math.round(parameters.config.N_I.value) > 0 ? (
                <path d={pathI} fill="none" stroke="#f97316" strokeWidth={1.4} />
              ) : null}

              <text x={width - margin.right - 90} y={14} fontSize={10} fill="var(--foreground)">
                E
              </text>
              {Math.round(parameters.config.N_I.value) > 0 ? (
                <text x={width - margin.right - 55} y={14} fontSize={10} fill="#f97316">
                  I
                </text>
              ) : null}

              {xTicks.map((tick) => (
                <text
                  key={`x-${tick}`}
                  x={xScale(tick)}
                  y={height - 8}
                  textAnchor="middle"
                  fontSize={9}
                  fill={axisColor}
                >
                  {Math.round(tick)}
                </text>
              ))}
              {yTicks.map((tick) => (
                <text
                  key={`y-${tick}`}
                  x={margin.left - 6}
                  y={yScale(tick) + 3}
                  textAnchor="end"
                  fontSize={9}
                  fill={axisColor}
                >
                  {formatAxisTick(tick)}
                </text>
              ))}
            </svg>
          )
        }}
      </ParentSize>
    </div>
  )
}
