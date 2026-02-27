import { useMemo } from 'react'
import { ParentSize } from '@visx/responsive'
import { scaleLinear } from '@visx/scale'
import { useSimulationContext } from '@/context/simulation-context'
import { formatAxisTick } from '@/lib/axis-format'

type Point = {
  tau: number
  crossCorrelation: number
}

const TAU_MIN = -60
const TAU_MAX = 60

type PlotCrossCorrelationProps = {
  layerLabel?: string
}

function toLinePath(points: Point[], x: (v: number) => number, y: (v: number) => number) {
  if (points.length === 0) {
    return ''
  }

  return points
    .map((point, i) => `${i === 0 ? 'M' : 'L'} ${x(point.tau)} ${y(point.crossCorrelation)}`)
    .join(' ')
}

export default function PlotCrossCorrelation({ layerLabel }: PlotCrossCorrelationProps) {
  const { runData } = useSimulationContext()
  const layerIdx =
    runData && layerLabel ? (runData.layer_labels ?? []).indexOf(layerLabel) : -1
  const points = useMemo(() => {
    if (!runData) {
      return []
    }
    const layerLags = layerIdx >= 0 ? runData.xcorr_lags_layers_ms?.[layerIdx] : undefined
    const layerCorr = layerIdx >= 0 ? runData.xcorr_corr_layers?.[layerIdx] : undefined
    const lags = layerLags && layerLags.length ? layerLags : runData.xcorr_lags_ms
    const corr = layerCorr && layerCorr.length ? layerCorr : runData.xcorr_corr
    if (!lags?.length || !corr?.length) {
      return []
    }
    return lags.map((tau, idx) => ({
      tau,
      crossCorrelation: corr[idx] ?? 0,
    }))
  }, [runData, layerIdx])

  return (
    <div className="flex h-full w-full min-h-0 min-w-0 overflow-hidden rounded-md border bg-background">
      <ParentSize className="h-full w-full min-h-0 min-w-0">
        {({ width, height }) => {
          if (width < 20 || height < 20) {
            return null
          }

          const margin = { top: 12, right: 14, bottom: 34, left: 44 }
          const minTau = points[0]?.tau ?? TAU_MIN
          const maxTau = points[points.length - 1]?.tau ?? TAU_MAX
          const xScale = scaleLinear<number>({
            domain: [minTau, maxTau],
            range: [margin.left, width - margin.right],
          })
          const yScale = scaleLinear<number>({
            domain: [-1.05, 1.05],
            range: [height - margin.bottom, margin.top],
          })
          const xTicks = [
            minTau,
            minTau + (maxTau - minTau) * 0.25,
            minTau + (maxTau - minTau) * 0.5,
            minTau + (maxTau - minTau) * 0.75,
            maxTau,
          ]
          const yTicks = [-1, -0.5, 0, 0.5, 1]
          const axisColor = 'var(--muted-foreground)'
          const path = toLinePath(points, xScale, yScale)

          return (
            <svg width={width} height={height} className="block h-full w-full">
              <rect x={0} y={0} width={width} height={height} fill="var(--background)" />

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

              <line
                x1={margin.left}
                x2={width - margin.right}
                y1={yScale(0)}
                y2={yScale(0)}
                stroke="var(--border)"
                strokeWidth={1}
                strokeDasharray="3 3"
              />
              <line
                x1={xScale(0)}
                x2={xScale(0)}
                y1={margin.top}
                y2={height - margin.bottom}
                stroke="var(--border)"
                strokeWidth={1}
                strokeDasharray="3 3"
              />

              <path d={path} fill="none" stroke="var(--foreground)" strokeWidth={1.8} />

              <text x={margin.left} y={14} fontSize={11} fill={axisColor}>
                Cross-corr {layerLabel ? `(${layerLabel})` : ''}
              </text>

              {xTicks.map((tick) => {
                const x = xScale(tick)
                return (
                  <g key={`x-${tick}`}>
                    <line
                      x1={x}
                      x2={x}
                      y1={height - margin.bottom}
                      y2={height - margin.bottom + 5}
                      stroke={axisColor}
                      strokeWidth={1}
                    />
                    <text
                      x={x}
                      y={height - margin.bottom + 18}
                      textAnchor="middle"
                      fontSize={10}
                      fill={axisColor}
                    >
                      {formatAxisTick(tick)}
                    </text>
                  </g>
                )
              })}

              {yTicks.map((tick) => {
                const y = yScale(tick)
                return (
                  <g key={`y-${tick}`}>
                    <line
                      x1={margin.left - 5}
                      x2={margin.left}
                      y1={y}
                      y2={y}
                      stroke={axisColor}
                      strokeWidth={1}
                    />
                    <text
                      x={margin.left - 8}
                      y={y + 3}
                      textAnchor="end"
                      fontSize={10}
                      fill={axisColor}
                    >
                      {tick}
                    </text>
                  </g>
                )
              })}

              <text
                x={(margin.left + width - margin.right) / 2}
                y={height - 6}
                textAnchor="middle"
                fontSize={11}
                fill={axisColor}
              >
                tau
              </text>
              <text
                x={12}
                y={(margin.top + height - margin.bottom) / 2}
                textAnchor="middle"
                fontSize={11}
                fill={axisColor}
                transform={`rotate(-90 12 ${(margin.top + height - margin.bottom) / 2})`}
              >
                cross correlation
              </text>
            </svg>
          )
        }}
      </ParentSize>
    </div>
  )
}
