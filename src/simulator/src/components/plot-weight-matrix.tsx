import { useMemo } from 'react'
import { ParentSize } from '@visx/responsive'
import { useSimulationContext } from '@/context/simulation-context'

type Cell = {
  i: number
  j: number
  value: number
}

type MatrixData = {
  cells: Cell[]
  rows: number
  cols: number
  minValue: number
  maxValue: number
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

function colorForValue(v: number, minValue: number, maxValue: number) {
  const span = Math.max(1e-9, maxValue - minValue)
  const t = Math.max(0, Math.min(1, (v - minValue) / span))
  const r = Math.round(lerp(255, 0, t))
  const g = Math.round(lerp(255, 0, t))
  const b = Math.round(lerp(255, 0, t))
  return `rgb(${r}, ${g}, ${b})`
}

export default function PlotWeightMatrix() {
  const { weightsPreview, runData, parameters } = useSimulationContext()
  const matrix = weightsPreview?.weights_heatmap ?? runData?.weights_heatmap ?? []
  const nE = Math.max(0, Math.round(parameters.config.N_E.value))
  const nI = Math.max(0, Math.round(parameters.config.N_I.value))
  const totalNeurons = Math.max(1, nE + nI)
  const templateBlocks = 1
  const eiBoundaryRatio = totalNeurons > 0 ? nE / totalNeurons : 0
  const blockBoundaryRatios = useMemo(() => {
    if (templateBlocks <= 1 || nE <= 0) {
      return []
    }
    return Array.from({ length: templateBlocks - 1 }, (_, idx) =>
      (((idx + 1) * nE) / templateBlocks) / totalNeurons
    ).filter((ratio) => ratio > 0 && ratio < 1)
  }, [templateBlocks, nE, totalNeurons])
  const layerLabelPositions = useMemo(() => {
    if (templateBlocks <= 0 || nE <= 0 || totalNeurons <= 0) {
      return []
    }
    const eRatio = nE / totalNeurons
    return Array.from({ length: templateBlocks }, (_, idx) => {
      const start = (idx * eRatio) / templateBlocks
      const end = ((idx + 1) * eRatio) / templateBlocks
      return {
        index: idx + 1,
        ratio: (start + end) / 2,
      }
    })
  }, [templateBlocks, nE, totalNeurons])
  const inhibitoryLabelRatio = useMemo(() => {
    if (nI <= 0 || totalNeurons <= 0) {
      return null
    }
    return (eiBoundaryRatio + 1) / 2
  }, [nI, totalNeurons, eiBoundaryRatio])

  const matrixData = useMemo<MatrixData>(() => {
    if (!matrix.length || !matrix[0]?.length) {
      return { cells: [], rows: 1, cols: 1, minValue: 0, maxValue: 1 }
    }

    const rows = matrix.length
    const cols = matrix[0].length
    const cells: Cell[] = []
    let minValue = Number.POSITIVE_INFINITY
    let maxValue = Number.NEGATIVE_INFINITY

    for (let j = 0; j < rows; j += 1) {
      const row = matrix[j]
      for (let i = 0; i < cols; i += 1) {
        const value = row[i] ?? 0
        minValue = Math.min(minValue, value)
        maxValue = Math.max(maxValue, value)
        cells.push({ i, j, value })
      }
    }

    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
      return { cells: [], rows: 1, cols: 1, minValue: 0, maxValue: 1 }
    }

    return { cells, rows, cols, minValue, maxValue }
  }, [matrix])

  return (
    <div className="flex h-full w-full min-h-0 min-w-0 overflow-hidden rounded-lg border bg-card p-2">
      <ParentSize className="h-full w-full min-h-0 min-w-0">
        {({ width, height }) => {
          if (width < 50 || height < 50) {
            return null
          }

          const margin = { top: 30, right: 24, bottom: 24, left: 40 }
          const plotWidth = width - margin.left - margin.right
          const plotHeight = height - margin.top - margin.bottom
          const cellW = plotWidth / matrixData.cols
          const cellH = plotHeight / matrixData.rows
          const axisColor = 'var(--muted-foreground)'

          return (
            <svg width={width} height={height} className="block h-full w-full">
              <rect x={0} y={0} width={width} height={height} fill="var(--background)" />

              {matrixData.cells.map((cell) => (
                <rect
                  key={`${cell.i}-${cell.j}`}
                  x={margin.left + cell.i * cellW}
                  y={margin.top + cell.j * cellH}
                  width={cellW + 0.5}
                  height={cellH + 0.5}
                  fill={colorForValue(cell.value, matrixData.minValue, matrixData.maxValue)}
                />
              ))}

              <line
                x1={margin.left}
                x2={margin.left + plotWidth}
                y1={margin.top + plotHeight}
                y2={margin.top + plotHeight}
                stroke={axisColor}
                strokeWidth={1}
              />
              <line
                x1={margin.left}
                x2={margin.left}
                y1={margin.top}
                y2={margin.top + plotHeight}
                stroke={axisColor}
                strokeWidth={1}
              />
              {blockBoundaryRatios.map((ratio, idx) => {
                const x = margin.left + ratio * plotWidth
                const y = margin.top + ratio * plotHeight
                return (
                  <g key={`matrix-boundary-${idx}`}>
                    <line
                      x1={x}
                      x2={x}
                      y1={margin.top}
                      y2={margin.top + plotHeight}
                      stroke="#22c55e"
                      strokeWidth={1.5}
                      strokeDasharray="6 4"
                      opacity={0.9}
                    />
                    <line
                      x1={margin.left}
                      x2={margin.left + plotWidth}
                      y1={y}
                      y2={y}
                      stroke="#22c55e"
                      strokeWidth={1.5}
                      strokeDasharray="6 4"
                      opacity={0.9}
                    />
                  </g>
                )
              })}
              {layerLabelPositions.map((layer) => (
                <text
                  key={`layer-label-x-${layer.index}`}
                  x={margin.left + layer.ratio * plotWidth}
                  y={margin.top - 6}
                  textAnchor="middle"
                  fontSize={14}
                  fill="#22c55e"
                >
                  L{layer.index}
                </text>
              ))}
              {inhibitoryLabelRatio !== null ? (
                <text
                  x={margin.left + inhibitoryLabelRatio * plotWidth}
                  y={margin.top - 6}
                  textAnchor="middle"
                  fontSize={14}
                  fill="#3b82f6"
                >
                  I
                </text>
              ) : null}
              {layerLabelPositions.map((layer) => (
                <text
                  key={`layer-label-y-${layer.index}`}
                  x={margin.left - 6}
                  y={margin.top + layer.ratio * plotHeight + 3}
                  textAnchor="end"
                  fontSize={14}
                  fill="#22c55e"
                >
                  L{layer.index}
                </text>
              ))}
              {inhibitoryLabelRatio !== null ? (
                <text
                  x={margin.left - 6}
                  y={margin.top + inhibitoryLabelRatio * plotHeight + 3}
                  textAnchor="end"
                  fontSize={14}
                  fill="#3b82f6"
                >
                  I
                </text>
              ) : null}
              {nE > 0 && nI > 0 && eiBoundaryRatio > 0 && eiBoundaryRatio < 1 ? (
                <g>
                  <line
                    x1={margin.left + eiBoundaryRatio * plotWidth}
                    x2={margin.left + eiBoundaryRatio * plotWidth}
                    y1={margin.top}
                    y2={margin.top + plotHeight}
                    stroke="#3b82f6"
                    strokeWidth={1.5}
                    strokeDasharray="6 4"
                    opacity={0.95}
                  />
                  <line
                    x1={margin.left}
                    x2={margin.left + plotWidth}
                    y1={margin.top + eiBoundaryRatio * plotHeight}
                    y2={margin.top + eiBoundaryRatio * plotHeight}
                    stroke="#3b82f6"
                    strokeWidth={1.5}
                    strokeDasharray="6 4"
                    opacity={0.95}
                  />
                </g>
              ) : null}

            </svg>
          )
        }}
      </ParentSize>
    </div>
  )
}
