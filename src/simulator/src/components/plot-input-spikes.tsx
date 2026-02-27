import { useMemo } from 'react'
import { ParentSize } from '@visx/responsive'
import { scaleLinear } from '@visx/scale'
import { useSimulationContext } from '@/context/simulation-context'
import { formatAxisTick } from '@/lib/axis-format'

type Point = {
  timestep: number
  value: number
}

type RasterPoint = {
  timestep: number
  id: number
}

function toLinePath(points: Point[], x: (v: number) => number, y: (v: number) => number) {
  if (points.length === 0) return ''
  return points
    .map((point, i) => `${i === 0 ? 'M' : 'L'} ${x(point.timestep)} ${y(point.value)}`)
    .join(' ')
}

function EnvelopePlot({ points }: { points: Point[] }) {
  return (
    <div className="min-h-0 overflow-hidden rounded-md border bg-background">
      <ParentSize className="h-full w-full min-h-0 min-w-0">
        {({ width, height }) => {
          if (width < 20 || height < 20) return null
          const margin = { top: 20, right: 12, bottom: 28, left: 40 }
          const minT = points[0]?.timestep ?? 0
          const maxT = points[points.length - 1]?.timestep ?? 1000
          const maxY = Math.max(1, ...points.map((p) => Math.max(0, p.value)))
          const xScale = scaleLinear<number>({
            domain: [minT, maxT],
            range: [margin.left, width - margin.right],
          })
          const yScale = scaleLinear<number>({
            domain: [0, maxY * 1.05],
            range: [height - margin.bottom, margin.top],
          })
          const xTicks = [
            minT,
            minT + (maxT - minT) * 0.25,
            minT + (maxT - minT) * 0.5,
            minT + (maxT - minT) * 0.75,
            maxT,
          ]
          const yTicks = [0, maxY * 0.5, maxY]
          const axisColor = 'var(--muted-foreground)'
          const path = toLinePath(points, xScale, yScale)

          return (
            <svg width={width} height={height} className="block h-full w-full">
              <rect x={0} y={0} width={width} height={height} fill="var(--background)" />
              <text x={margin.left} y={14} fontSize={11} fill={axisColor}>
                Generator Envelope (Hz)
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
              <path d={path} fill="none" stroke="#22c55e" strokeWidth={1.6} />
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

function RasterPlot({ points }: { points: RasterPoint[] }) {
  const maxId = Math.max(1, ...points.map((p) => p.id + 1))
  return (
    <div className="min-h-0 overflow-hidden rounded-md border bg-background">
      <ParentSize className="h-full w-full min-h-0 min-w-0">
        {({ width, height }) => {
          if (width < 20 || height < 20) return null
          const margin = { top: 20, right: 12, bottom: 28, left: 40 }
          const minT = points[0]?.timestep ?? 0
          const maxT = points[points.length - 1]?.timestep ?? 1000
          const xScale = scaleLinear<number>({
            domain: [minT, maxT],
            range: [margin.left, width - margin.right],
          })
          const yScale = scaleLinear<number>({
            domain: [-0.5, maxId - 0.5],
            range: [height - margin.bottom, margin.top],
          })
          const axisColor = 'var(--muted-foreground)'
          const xTicks = [
            minT,
            minT + (maxT - minT) * 0.25,
            minT + (maxT - minT) * 0.5,
            minT + (maxT - minT) * 0.75,
            maxT,
          ]
          const yTicks = [0, Math.floor(maxId * 0.5), Math.max(0, maxId - 1)]
          return (
            <svg width={width} height={height} className="block h-full w-full">
              <rect x={0} y={0} width={width} height={height} fill="var(--background)" />
              <text x={margin.left} y={14} fontSize={11} fill={axisColor}>
                Input Spike Raster
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
              {points.length > 0 ? (
                points.map((point, idx) => (
                  <circle
                    key={`r-${idx}`}
                    cx={xScale(point.timestep)}
                    cy={yScale(point.id)}
                    r={1.2}
                    fill="var(--foreground)"
                    opacity={0.9}
                  />
                ))
              ) : (
                <text
                  x={(margin.left + width - margin.right) * 0.5}
                  y={(margin.top + height - margin.bottom) * 0.5}
                  textAnchor="middle"
                  fontSize={10}
                  fill={axisColor}
                >
                  No input spike raster available
                </text>
              )}
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
                  {tick}
                </text>
              ))}
            </svg>
          )
        }}
      </ParentSize>
    </div>
  )
}

export default function PlotInputSpikes({ layerLabel }: { layerLabel?: string }) {
  const { runData, parameters } = useSimulationContext()
  const fallbackLabels = useMemo(() => {
    const labels = ['L1']
    if (Math.round(parameters.config.N_I.value) > 0) labels.push('I')
    return labels
  }, [parameters.config.N_I.value])
  const layerLabels = runData?.layer_labels?.length ? runData.layer_labels : fallbackLabels
  const activeLayer =
    layerLabel && layerLabels.includes(layerLabel) ? layerLabel : (layerLabels[0] ?? 'L1')
  const layerIdx = Math.max(0, layerLabels.indexOf(activeLayer))
  const tMs = runData?.input_t_ms ?? []

  const envelopePoints = useMemo(() => {
    if (!tMs.length) return []
    const layerEnvelope = runData?.input_envelope_hz_layers?.[layerIdx]
    const envelope = layerEnvelope && layerEnvelope.length ? layerEnvelope : runData?.input_envelope_hz ?? []
    return tMs.map((timestep, idx) => {
      const value = envelope[idx] ?? 0
      return { timestep, value }
    })
  }, [tMs, runData, layerIdx])

  const rasterPoints = useMemo(() => {
    const times = runData?.input_raw_raster_times_ms_layers?.[layerIdx] ?? []
    const ids = runData?.input_raw_raster_ids_layers?.[layerIdx] ?? []
    const n = Math.min(times.length, ids.length)
    return Array.from({ length: n }, (_, idx) => ({
      timestep: times[idx] ?? 0,
      id: ids[idx] ?? 0,
    })) as RasterPoint[]
  }, [runData, layerIdx])

  return (
    <div className="grid h-full min-h-0 grid-cols-1 gap-2 lg:grid-cols-2">
      <EnvelopePlot points={envelopePoints} />
      <RasterPlot points={rasterPoints} />
    </div>
  )
}
