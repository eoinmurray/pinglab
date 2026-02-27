import { useMemo } from 'react'
import { ParentSize } from '@visx/responsive'
import { scaleLinear } from '@visx/scale'
import { useSimulationContext } from '@/context/simulation-context'
import { formatAxisTick } from '@/lib/axis-format'
import { getTargetedInputOverlay } from '@/lib/targeted-input-overlay'

type Point = {
  timestep: number
  value: number
}

const TIMESTEP_MAX = 200

type PlotInputTraceProps = {
  layerLabel?: string
}

function toLinePath(points: Point[], x: (v: number) => number, y: (v: number) => number) {
  if (points.length === 0) return ''
  return points
    .map((point, i) => `${i === 0 ? 'M' : 'L'} ${x(point.timestep)} ${y(point.value)}`)
    .join(' ')
}

export default function PlotInputTrace({ layerLabel }: PlotInputTraceProps) {
  const { runData, parameters } = useSimulationContext()
  const targetedInputOverlay = getTargetedInputOverlay(parameters)
  const fallbackLabels = useMemo(() => {
    const labels = ['L1']
    if (Math.round(parameters.config.N_I.value) > 0) labels.push('I')
    return labels
  }, [parameters.config.N_I.value])
  const layerLabels = runData?.layer_labels?.length ? runData.layer_labels : fallbackLabels
  const activeLayer =
    layerLabel && layerLabels.includes(layerLabel) ? layerLabel : (layerLabels[0] ?? 'L1')
  const activeLayerIdx = Math.max(0, layerLabels.indexOf(activeLayer))

  const points = useMemo(() => {
    if (!runData || !runData.input_t_ms.length) return []
    const fromLayers = runData.input_mean_layers?.[activeLayerIdx]
    let values = fromLayers
    if (!values || values.length === 0) {
      values = activeLayer === 'I' ? runData.input_mean_I : runData.input_mean_E
    }
    return runData.input_t_ms.map((timestep, idx) => ({
      timestep,
      value: values?.[idx] ?? 0,
    }))
  }, [runData, activeLayer, activeLayerIdx])

  return (
    <div className="relative flex h-full w-full min-h-0 min-w-0 overflow-hidden rounded-md border bg-background">
      {targetedInputOverlay ? (
        <div
          className={`pointer-events-none absolute right-2 top-2 z-10 rounded border px-2 py-0.5 text-[10px] leading-tight ${
            targetedInputOverlay.emphasized
              ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-700'
              : 'border-border bg-background/90 text-muted-foreground'
          }`}
        >
          {targetedInputOverlay.text}
        </div>
      ) : null}
      <ParentSize className="h-full w-full min-h-0 min-w-0">
        {({ width, height }) => {
          if (width < 20 || height < 20) return null

          const margin = { top: 20, right: 12, bottom: 30, left: 36 }
          const minTimestep = points[0]?.timestep ?? 0
          const maxTimestep = points[points.length - 1]?.timestep ?? TIMESTEP_MAX
          const maxValue = Math.max(1, ...points.map((point) => Math.max(0, point.value)))
          const xScale = scaleLinear<number>({
            domain: [minTimestep, maxTimestep],
            range: [margin.left, width - margin.right],
          })
          const yScale = scaleLinear<number>({
            domain: [0, maxValue * 1.05],
            range: [height - margin.bottom, margin.top],
          })
          const xTicks = [
            minTimestep,
            minTimestep + (maxTimestep - minTimestep) * 0.25,
            minTimestep + (maxTimestep - minTimestep) * 0.5,
            minTimestep + (maxTimestep - minTimestep) * 0.75,
            maxTimestep,
          ]
          const yTicks = [0, maxValue * 0.5, maxValue]
          const axisColor = 'var(--muted-foreground)'
          const path = toLinePath(points, xScale, yScale)

          return (
            <svg width={width} height={height} className="block h-full w-full">
              <rect x={0} y={0} width={width} height={height} fill="var(--background)" />

              <text x={margin.left} y={14} fontSize={11} fill={axisColor}>
                {`Input (${activeLayer})`}
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

              <path d={path} fill="none" stroke="var(--foreground)" strokeWidth={1.6} />

              {xTicks.map((tick) => {
                const x = xScale(tick)
                return (
                  <text
                    key={`x-${tick}`}
                    x={x}
                    y={height - 8}
                    textAnchor="middle"
                    fontSize={9}
                    fill={axisColor}
                  >
                    {Math.round(tick)}
                  </text>
                )
              })}
              {yTicks.map((tick) => {
                const y = yScale(tick)
                return (
                  <text
                    key={`y-${tick}`}
                    x={margin.left - 6}
                    y={y + 3}
                    textAnchor="end"
                    fontSize={9}
                    fill={axisColor}
                  >
                    {formatAxisTick(tick)}
                  </text>
                )
              })}
            </svg>
          )
        }}
      </ParentSize>
    </div>
  )
}
