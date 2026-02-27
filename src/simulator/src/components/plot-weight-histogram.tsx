import { useMemo } from 'react'
import { ParentSize } from '@visx/responsive'
import { scaleBand, scaleLinear } from '@visx/scale'
import { useSimulationContext } from '@/context/simulation-context'

type PlotWeightHistogramProps = {
  title: string
  mode: 'EE' | 'EI' | 'IE' | 'II'
  counts?: number[]
  bins?: number[]
}

export default function PlotWeightHistogram({
  title,
  mode,
  counts,
  bins,
}: PlotWeightHistogramProps) {
  const { runData } = useSimulationContext()
  const histogramBins = useMemo(() => {
    const sourceCounts =
      counts ??
      (mode === 'EE'
        ? runData?.weights_hist_counts_ee
        : mode === 'EI'
          ? runData?.weights_hist_counts_ei
          : mode === 'IE'
            ? runData?.weights_hist_counts_ie
            : runData?.weights_hist_counts_ii)
    const sourceBins = bins ?? runData?.weights_hist_bins
    if (!sourceCounts?.length) {
      return []
    }
    return sourceCounts.map((count, idx) => ({
      bin: Number(sourceBins?.[idx] ?? idx),
      count,
    }))
  }, [bins, counts, mode, runData])

  return (
    <div className="flex h-full w-full min-h-0 min-w-0 overflow-hidden rounded-md border bg-background">
      <ParentSize className="h-full w-full min-h-0 min-w-0">
        {({ width, height }) => {
          if (width < 20 || height < 20) {
            return null
          }

          const margin = { top: 20, right: 10, bottom: 28, left: 24 }
          const innerWidth = width - margin.left - margin.right
          const innerHeight = height - margin.top - margin.bottom
          const maxCount = Math.max(1, ...histogramBins.map((b) => b.count))

          const xScale = scaleBand<string>({
            domain: histogramBins.map((b) => String(b.bin)),
            range: [margin.left, margin.left + innerWidth],
            padding: 0.12,
          })
          const yScale = scaleLinear<number>({
            domain: [0, maxCount * 1.05],
            range: [margin.top + innerHeight, margin.top],
          })
          const axisColor = 'var(--muted-foreground)'

          return (
            <svg width={width} height={height} className="block h-full w-full">
              <rect x={0} y={0} width={width} height={height} fill="var(--background)" />

              <text x={margin.left} y={14} fontSize={11} fill={axisColor}>
                {title}
              </text>

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

              {histogramBins.map((bin) => {
                const x = xScale(String(bin.bin))
                if (x == null) {
                  return null
                }
                const y = yScale(bin.count)
                const barHeight = margin.top + innerHeight - y

                return (
                  <rect
                    key={`${mode}-${bin.bin}`}
                    x={x}
                    y={y}
                    width={xScale.bandwidth()}
                    height={barHeight}
                    fill="var(--foreground)"
                    opacity={0.82}
                  />
                )
              })}
            </svg>
          )
        }}
      </ParentSize>
    </div>
  )
}
