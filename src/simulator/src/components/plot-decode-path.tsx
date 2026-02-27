import { useMemo } from 'react'
import { ParentSize } from '@visx/responsive'
import { scaleLinear } from '@visx/scale'
import { useSimulationContext } from '@/context/simulation-context'

type Point = {
  timestep: number
  value: number
}

type PlotDecodePathProps = {
  layerLabel?: string
}

function toLinePath(points: Point[], x: (v: number) => number, y: (v: number) => number) {
  if (points.length === 0) return ''
  return points
    .map((point, i) => `${i === 0 ? 'M' : 'L'} ${x(point.timestep)} ${y(point.value)}`)
    .join(' ')
}

function normalize(values: number[]): number[] {
  if (values.length === 0) return []
  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  for (const value of values) {
    if (value < min) min = value
    if (value > max) max = value
  }
  const span = max - min
  if (!Number.isFinite(span) || span <= 1e-12) {
    return values.map(() => 0)
  }
  return values.map((value) => (value - min) / span)
}

export default function PlotDecodePath({ layerLabel }: PlotDecodePathProps) {
  const { runData } = useSimulationContext()

  const series = useMemo(() => {
    if (!runData || !runData.population_rate_t_ms.length) {
      return {
        tMs: [] as number[],
        binnedNorm: [] as number[],
        lowPassNorm: [] as number[],
        envelopeNorm: [] as number[],
      }
    }
    const layerLabels = runData.layer_labels ?? []
    const activeLayer =
      layerLabel && layerLabels.includes(layerLabel) ? layerLabel : (layerLabels[0] ?? 'L1')
    const activeLayerIdx = Math.max(0, layerLabels.indexOf(activeLayer))
    const layerSeries = runData.population_rate_hz_layers?.[activeLayerIdx]
    const binnedRate =
      layerSeries && layerSeries.length > 0 ? layerSeries : runData.population_rate_hz_E
    const layerLowPassSeries = runData.decode_lowpass_hz_layers?.[activeLayerIdx]
    const lowPassRate =
      layerLowPassSeries && layerLowPassSeries.length > 0
        ? layerLowPassSeries
        : runData.decode_lowpass_hz_E

    const tMs = runData.population_rate_t_ms

    const layerEnvelopeSeries = runData.decode_envelope_hz_layers?.[activeLayerIdx]
    const envelope =
      layerEnvelopeSeries && layerEnvelopeSeries.length > 0
        ? layerEnvelopeSeries
        : runData.decode_envelope_hz

    const binnedNorm = normalize(binnedRate)
    const lowPassNorm = normalize(lowPassRate)
    const envelopeNorm = normalize(envelope)

    return {
      tMs,
      binnedNorm,
      lowPassNorm,
      envelopeNorm,
    }
  }, [runData, layerLabel])

  const binnedPoints: Point[] = series.tMs.map((timestep, idx) => ({
    timestep,
    value: series.binnedNorm[idx] ?? 0,
  }))
  const lowPassPoints: Point[] = series.tMs.map((timestep, idx) => ({
    timestep,
    value: series.lowPassNorm[idx] ?? 0,
  }))
  const envelopePoints: Point[] = series.tMs.map((timestep, idx) => ({
    timestep,
    value: series.envelopeNorm[idx] ?? 0,
  }))

  return (
    <div className="h-full min-h-0">
      <div className="relative h-full min-h-0 overflow-hidden rounded-md border bg-background">
        <ParentSize className="h-full w-full min-h-0 min-w-0">
          {({ width, height }) => {
            if (width < 20 || height < 20) return null
            const margin = { top: 24, right: 14, bottom: 30, left: 40 }
            const minT = series.tMs[0] ?? 0
            const maxT = series.tMs[series.tMs.length - 1] ?? 1000
            const xScale = scaleLinear<number>({
              domain: [minT, maxT],
              range: [margin.left, width - margin.right],
            })
            const yScale = scaleLinear<number>({
              domain: [0, 1],
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
            return (
              <svg width={width} height={height} className="block h-full w-full">
                <rect x={0} y={0} width={width} height={height} fill="var(--background)" />
                <text x={margin.left} y={14} fontSize={11} fill={axisColor}>
                  Decode Path (normalized)
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
                <path
                  d={toLinePath(binnedPoints, xScale, yScale)}
                  fill="none"
                  stroke="#64748b"
                  strokeWidth={1.2}
                />
                <path
                  d={toLinePath(lowPassPoints, xScale, yScale)}
                  fill="none"
                  stroke="#ef4444"
                  strokeWidth={1.8}
                />
                <path
                  d={toLinePath(envelopePoints, xScale, yScale)}
                  fill="none"
                  stroke="#22c55e"
                  strokeWidth={1.6}
                />
                <text x={margin.left + 8} y={26} fontSize={10} fill="#64748b">
                  binned pop rate
                </text>
                <text x={margin.left + 116} y={26} fontSize={10} fill="#ef4444">
                  low-pass &lt;10Hz
                </text>
                <text x={margin.left + 210} y={26} fontSize={10} fill="#22c55e">
                  envelope
                </text>
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
                {[0, 0.5, 1].map((tick) => (
                  <text
                    key={`y-${tick}`}
                    x={margin.left - 6}
                    y={yScale(tick) + 3}
                    textAnchor="end"
                    fontSize={9}
                    fill={axisColor}
                  >
                    {tick.toFixed(1)}
                  </text>
                ))}
              </svg>
            )
          }}
        </ParentSize>
      </div>
    </div>
  )
}
