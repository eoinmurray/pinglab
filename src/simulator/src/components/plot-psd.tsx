import { useMemo } from 'react'
import { ParentSize } from '@visx/responsive'
import { scaleLinear } from '@visx/scale'
import { useSimulationContext } from '@/context/simulation-context'
import { formatAxisTick } from '@/lib/axis-format'

type PlotPsdProps = {
  layerLabel?: string
}

export default function PlotPsd({ layerLabel }: PlotPsdProps) {
  const { runData, parameters } = useSimulationContext()
  const fallbackLabels = useMemo(() => {
    const labels = ['L1']
    if (Math.round(parameters.config.N_I.value) > 0) labels.push('I')
    return labels
  }, [parameters.config.N_I.value])
  const layerLabels = runData?.layer_labels?.length ? runData.layer_labels : fallbackLabels
  const activeLayer =
    layerLabel && layerLabels.includes(layerLabel) ? layerLabel : (layerLabels[0] ?? 'L1')
  const activeLayerIdx = Math.max(0, layerLabels.indexOf(activeLayer))

  const bins = useMemo(() => {
    if (!runData || !runData.psd_freqs_hz.length) return []
    const layerPower = runData.psd_power_layers?.[activeLayerIdx]
    const values = layerPower && layerPower.length ? layerPower : runData.psd_power
    return runData.psd_freqs_hz.map((frequency, idx) => ({
      frequency,
      power: values[idx] ?? 0,
    }))
  }, [runData, activeLayerIdx])

  return (
    <div className="relative flex h-full w-full min-h-0 min-w-0 overflow-hidden rounded-md border bg-background">
      <ParentSize className="h-full w-full min-h-0 min-w-0">
        {({ width, height }) => {
          if (width < 20 || height < 20) return null

          const margin = { top: 12, right: 14, bottom: 34, left: 44 }
          const innerWidth = width - margin.left - margin.right
          const innerHeight = height - margin.top - margin.bottom
          const maxPower = Math.max(1, ...bins.map((b) => b.power))
          const firstFrequency = bins[0]?.frequency ?? 0
          const lastFrequency = bins[bins.length - 1]?.frequency ?? 1

          const xScale = scaleLinear<number>({
            domain: [firstFrequency, lastFrequency],
            range: [margin.left, margin.left + innerWidth],
          })

          const yScale = scaleLinear<number>({
            domain: [0, maxPower * 1.05],
            range: [margin.top + innerHeight, margin.top],
          })

          const xTicks = [
            firstFrequency,
            firstFrequency + (lastFrequency - firstFrequency) * 0.25,
            firstFrequency + (lastFrequency - firstFrequency) * 0.5,
            firstFrequency + (lastFrequency - firstFrequency) * 0.75,
            lastFrequency,
          ]
          const yTicks = [0, maxPower * 0.25, maxPower * 0.5, maxPower * 0.75, maxPower]
          const axisColor = 'var(--muted-foreground)'
          const linePath = bins
            .map((bin, idx) => `${idx === 0 ? 'M' : 'L'} ${xScale(bin.frequency)} ${yScale(bin.power)}`)
            .join(' ')

          return (
            <svg width={width} height={height} className="block h-full w-full">
              <rect x={0} y={0} width={width} height={height} fill="var(--background)" />

              <text x={margin.left} y={14} fontSize={11} fill={axisColor}>
                PSD ({activeLayer})
              </text>

              {bins.length > 1 ? (
                <path d={linePath} fill="none" stroke="var(--foreground)" strokeWidth={1.5} />
              ) : null}

              <line
                x1={margin.left}
                x2={width - margin.right}
                y1={margin.top + innerHeight}
                y2={margin.top + innerHeight}
                stroke={axisColor}
                strokeWidth={1}
              />
              <line
                x1={margin.left}
                x2={margin.left}
                y1={margin.top}
                y2={margin.top + innerHeight}
                stroke={axisColor}
                strokeWidth={1}
              />

              {xTicks.map((tick) => {
                const centered = xScale(tick)
                return (
                  <g key={`x-${tick}`}>
                    <line
                      x1={centered}
                      x2={centered}
                      y1={margin.top + innerHeight}
                      y2={margin.top + innerHeight + 5}
                      stroke={axisColor}
                      strokeWidth={1}
                    />
                    <text
                      x={centered}
                      y={margin.top + innerHeight + 18}
                      textAnchor="middle"
                      fontSize={10}
                      fill={axisColor}
                    >
                      {Math.round(tick)}
                    </text>
                  </g>
                )
              })}

              {yTicks.map((tick, i) => {
                const y = yScale(tick)
                return (
                  <g key={`y-${i}`}>
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
                      {formatAxisTick(tick)}
                    </text>
                  </g>
                )
              })}
            </svg>
          )
        }}
      </ParentSize>
    </div>
  )
}
