import { useEffect, useMemo, useRef } from 'react'
import { ParentSize } from '@visx/responsive'
import { scaleLinear } from '@visx/scale'
import { useSimulationContext } from '@/context/simulation-context'
import { formatAxisTick } from '@/lib/axis-format'
import { getTargetedInputOverlay } from '@/lib/targeted-input-overlay'

const DURATION_MS = 1000
const LAYER_GAP_UNITS = 8
const EI_GAP_UNITS = 10
const MAX_RENDER_SPIKES = 12000

type SpikePoint = {
  time: number
  neuron: number
  type: number
}

type LayerLabelPosition = {
  label: string
  center: number
}

type BoundaryLine = {
  boundary: number
  stroke: string
}

type RasterCanvasSceneProps = {
  width: number
  height: number
  spikes: SpikePoint[]
  maxTime: number
  maxNeuronWithGap: number
  mapNeuronWithLayerGap: (neuron: number) => number
  boundaryLines: BoundaryLine[]
  nE: number
  nI: number
  excitatoryLayerLabelPositions: LayerLabelPosition[]
  inhibitoryLayerLabelPositions: LayerLabelPosition[]
}

function RasterCanvasScene({
  width,
  height,
  spikes,
  maxTime,
  maxNeuronWithGap,
  mapNeuronWithLayerGap,
  boundaryLines,
  nE,
  nI,
  excitatoryLayerLabelPositions,
  inhibitoryLayerLabelPositions,
}: RasterCanvasSceneProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const margin = { top: 12, right: 14, bottom: 34, left: 44 }

  const xScale = useMemo(
    () =>
      scaleLinear<number>({
        domain: [0, maxTime],
        range: [margin.left, width - margin.right],
      }),
    [maxTime, width]
  )

  const yScale = useMemo(
    () =>
      scaleLinear<number>({
        domain: [0, Math.max(1, maxNeuronWithGap)],
        range: [height - margin.bottom, margin.top],
      }),
    [height, maxNeuronWithGap]
  )

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.floor(width * dpr)
    canvas.height = Math.floor(height * dpr)
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`

    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, height)

    const xMin = margin.left
    const xMax = width - margin.right
    const yMin = margin.top
    const yMax = height - margin.bottom

    const isDark = document.documentElement.classList.contains('dark')
    const excitatoryColor = isDark ? '#ffffff' : '#111111'
    const inhibitoryColor = '#ef4444'

    ctx.fillStyle = excitatoryColor
    for (const spike of spikes) {
      if (spike.type === 1) {
        continue
      }
      const x = xScale(spike.time)
      const y = yScale(mapNeuronWithLayerGap(spike.neuron))
      if (x < xMin || x > xMax || y < yMin || y > yMax) {
        continue
      }
      ctx.fillRect(x - 1, y - 1, 2, 2)
    }

    ctx.fillStyle = inhibitoryColor
    for (const spike of spikes) {
      if (spike.type !== 1) {
        continue
      }
      const x = xScale(spike.time)
      const y = yScale(mapNeuronWithLayerGap(spike.neuron))
      if (x < xMin || x > xMax || y < yMin || y > yMax) {
        continue
      }
      ctx.fillRect(x - 1, y - 1, 2, 2)
    }
  }, [height, mapNeuronWithLayerGap, spikes, width, xScale, yScale])

  const xTicks = [0, maxTime * 0.25, maxTime * 0.5, maxTime * 0.75, maxTime]
  const axisColor = 'var(--muted-foreground)'

  return (
    <div className="relative h-full w-full min-h-0 min-w-0">
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full"
        aria-label="Spike raster canvas"
      />
      <svg width={width} height={height} className="absolute inset-0 block h-full w-full">
        <rect x={0} y={0} width={width} height={height} fill="none" />

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
                {formatAxisTick(tick, { integer: true })}
              </text>
            </g>
          )
        })}

        {excitatoryLayerLabelPositions.map((layer) => (
          <text
            key={`layer-label-y-${layer.label}`}
            x={margin.left - 8}
            y={yScale(mapNeuronWithLayerGap(layer.center)) + 4}
            textAnchor="end"
            fontSize={14}
            fill="#22c55e"
          >
            {layer.label}
          </text>
        ))}

        {inhibitoryLayerLabelPositions.map((layer) => (
          <text
            key={`layer-label-y-${layer.label}`}
            x={margin.left - 8}
            y={yScale(mapNeuronWithLayerGap(layer.center)) + 4}
            textAnchor="end"
            fontSize={14}
            fill="#ef4444"
          >
            {layer.label}
          </text>
        ))}

        {boundaryLines.map((boundaryLine, idx) => {
          const boundary = boundaryLine.boundary
          const previousNeuron = Math.max(0, boundary - 1)
          const yPrev = yScale(mapNeuronWithLayerGap(previousNeuron))
          const yNext = yScale(mapNeuronWithLayerGap(boundary))
          const y = (yPrev + yNext) / 2
          return (
            <line
              key={`block-boundary-${idx}`}
              x1={margin.left}
              x2={width - margin.right}
              y1={y}
              y2={y}
              stroke={boundaryLine.stroke}
              strokeWidth={1.5}
              strokeDasharray="6 4"
              opacity={0.9}
            />
          )
        })}

        {nE > 0 && nI > 0 ? (
          (() => {
            const yPrev = yScale(mapNeuronWithLayerGap(0))
            const yNext = yScale(mapNeuronWithLayerGap(nE))
            const y = (yPrev + yNext) / 2
            return (
              <line
                x1={margin.left}
                x2={width - margin.right}
                y1={y}
                y2={y}
                stroke="#ef4444"
                strokeWidth={1.75}
                strokeDasharray="6 4"
                opacity={0.95}
              />
            )
          })()
        ) : null}

        <text
          x={(margin.left + width - margin.right) / 2}
          y={height - 6}
          textAnchor="middle"
          fontSize={11}
          fill={axisColor}
        >
          timestep
        </text>
      </svg>
    </div>
  )
}

export default function PlotRaster() {
  const { runData, parameters } = useSimulationContext()
  const targetedInputOverlay = getTargetedInputOverlay(parameters)
  const downsampleEnabled = parameters.downsampleEnabled.value

  const spikes = useMemo(() => {
    if (!runData || !runData.spikes.times.length || !runData.spikes.ids.length) {
      return [] as SpikePoint[]
    }
    return runData.spikes.times.map((time, idx) => ({
      time,
      neuron: runData.spikes.ids[idx] ?? 0,
      type: runData.spikes.types[idx] ?? 0,
    }))
  }, [runData])

  const nE = Math.max(0, Math.round(parameters.config.N_E.value))
  const nI = Math.max(0, Math.round(parameters.config.N_I.value))
  const totalNeurons = Math.max(1, nE + nI)
  const eLayerInfo = useMemo(() => {
    const eNodes = parameters.graph.nodes.filter(
      (node) => node.kind === 'population' && node.type === 'E' && node.size > 0
    )
    if (eNodes.length === 0) {
      return nE > 0 ? [{ label: 'L1', size: nE }] : []
    }

    const withOrder = eNodes.map((node, idx) => {
      const m = node.id.match(/(\d+)$/)
      const layerOrder = m ? Number(m[1]) : Number.NaN
      return {
        label: Number.isFinite(layerOrder) ? `L${layerOrder}` : `L${idx + 1}`,
        size: node.size,
        order: Number.isFinite(layerOrder) ? layerOrder : idx + 1,
      }
    })

    withOrder.sort((a, b) => a.order - b.order)
    return withOrder.map(({ label, size }) => ({ label, size }))
  }, [nE, parameters.graph.nodes])

  const iLayerInfo = useMemo(() => {
    const iNodes = parameters.graph.nodes.filter(
      (node) => node.kind === 'population' && node.type === 'I' && node.size > 0
    )
    if (iNodes.length === 0) {
      return nI > 0 ? [{ label: 'I', size: nI }] : []
    }

    const withOrder = iNodes.map((node, idx) => {
      const m = node.id.match(/(\d+)$/)
      const layerOrder = m ? Number(m[1]) : Number.NaN
      return {
        label: Number.isFinite(layerOrder) ? `I${layerOrder}` : `I${idx + 1}`,
        size: node.size,
        order: Number.isFinite(layerOrder) ? layerOrder : idx + 1,
      }
    })

    withOrder.sort((a, b) => a.order - b.order)
    return withOrder.map(({ label, size }) => ({ label, size }))
  }, [nI, parameters.graph.nodes])

  const eBlockBoundaries = useMemo(() => {
    if (eLayerInfo.length <= 1 || nE <= 0) {
      return []
    }
    const totalLayerSize = eLayerInfo.reduce((acc, layer) => acc + layer.size, 0)
    if (totalLayerSize <= 0) {
      return []
    }
    const scale = nE / totalLayerSize
    let cumulative = 0
    const boundaries: number[] = []
    for (let idx = 0; idx < eLayerInfo.length - 1; idx += 1) {
      cumulative += eLayerInfo[idx].size
      const boundary = Math.round(cumulative * scale)
      if (boundary > 0 && boundary < nE) {
        boundaries.push(boundary)
      }
    }
    return [...new Set(boundaries)]
  }, [eLayerInfo, nE])

  const iBlockBoundaries = useMemo(() => {
    if (iLayerInfo.length <= 1 || nI <= 0) {
      return []
    }
    const totalLayerSize = iLayerInfo.reduce((acc, layer) => acc + layer.size, 0)
    if (totalLayerSize <= 0) {
      return []
    }
    const scale = nI / totalLayerSize
    let cumulative = 0
    const boundaries: number[] = []
    for (let idx = 0; idx < iLayerInfo.length - 1; idx += 1) {
      cumulative += iLayerInfo[idx].size
      const boundary = nE + Math.round(cumulative * scale)
      if (boundary > nE && boundary < nE + nI) {
        boundaries.push(boundary)
      }
    }
    return [...new Set(boundaries)]
  }, [iLayerInfo, nE, nI])

  const maxNeuron = useMemo(
    () =>
      Math.max(
        totalNeurons - 1,
        spikes.length > 0 ? Math.max(0, ...spikes.map((spike) => spike.neuron)) : 1
      ),
    [spikes, totalNeurons]
  )

  const mapNeuronWithLayerGap = useMemo(() => {
    const allBoundaries = [...eBlockBoundaries, ...iBlockBoundaries].sort((a, b) => a - b)
    return (neuron: number) => {
      const displayNeuron = neuron < nE ? Math.max(0, nE - 1 - neuron) : neuron
      let offset = 0
      for (const boundary of allBoundaries) {
        if (displayNeuron >= boundary) {
          offset += LAYER_GAP_UNITS
        }
      }
      if (nE > 0 && nI > 0 && displayNeuron >= nE) {
        offset += EI_GAP_UNITS
      }
      return displayNeuron + offset
    }
  }, [eBlockBoundaries, iBlockBoundaries, nE, nI])

  const maxNeuronWithGap = useMemo(() => {
    return Math.max(mapNeuronWithLayerGap(0), mapNeuronWithLayerGap(maxNeuron))
  }, [mapNeuronWithLayerGap, maxNeuron])

  const excitatoryLayerLabelPositions = useMemo(() => {
    if (eLayerInfo.length <= 0 || nE <= 0) {
      return [] as LayerLabelPosition[]
    }
    const totalLayerSize = eLayerInfo.reduce((acc, layer) => acc + layer.size, 0)
    if (totalLayerSize <= 0) {
      return [] as LayerLabelPosition[]
    }
    const scale = nE / totalLayerSize
    let cumulative = 0
    return eLayerInfo
      .map((layer) => {
        const start = Math.round(cumulative * scale)
        cumulative += layer.size
        const end = Math.round(cumulative * scale)
        if (end <= start) {
          return null
        }
        const center = (start + (end - 1)) / 2
        return {
          label: layer.label,
          center,
        }
      })
      .filter((entry): entry is LayerLabelPosition => entry !== null)
  }, [eLayerInfo, nE])

  const inhibitoryLayerLabelPositions = useMemo(() => {
    if (iLayerInfo.length <= 0 || nI <= 0) {
      return [] as LayerLabelPosition[]
    }
    const totalLayerSize = iLayerInfo.reduce((acc, layer) => acc + layer.size, 0)
    if (totalLayerSize <= 0) {
      return [] as LayerLabelPosition[]
    }
    const scale = nI / totalLayerSize
    let cumulative = 0
    return iLayerInfo
      .map((layer) => {
        const start = nE + Math.round(cumulative * scale)
        cumulative += layer.size
        const end = nE + Math.round(cumulative * scale)
        if (end <= start) {
          return null
        }
        const center = (start + (end - 1)) / 2
        return {
          label: layer.label,
          center,
        }
      })
      .filter((entry): entry is LayerLabelPosition => entry !== null)
  }, [iLayerInfo, nE, nI])

  const boundaryLines = useMemo(() => {
    const eLines = eBlockBoundaries.map((boundary) => ({ boundary, stroke: '#22c55e' }))
    const iLines = iBlockBoundaries.map((boundary) => ({ boundary, stroke: '#ef4444' }))
    return [...eLines, ...iLines]
  }, [eBlockBoundaries, iBlockBoundaries])

  const maxTime = useMemo(
    () =>
      spikes.length > 0
        ? Math.max(DURATION_MS, ...spikes.map((spike) => spike.time))
        : DURATION_MS,
    [spikes]
  )

  const rasterSpikes = useMemo(() => {
    if (!downsampleEnabled || spikes.length <= MAX_RENDER_SPIKES) {
      return spikes
    }
    const step = spikes.length / MAX_RENDER_SPIKES
    return Array.from({ length: MAX_RENDER_SPIKES }, (_, idx) => {
      const sourceIdx = Math.min(spikes.length - 1, Math.floor(idx * step))
      return spikes[sourceIdx]
    })
  }, [downsampleEnabled, spikes])

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

      {downsampleEnabled && spikes.length > MAX_RENDER_SPIKES ? (
        <div className="pointer-events-none absolute left-2 top-2 z-10 rounded border bg-background/90 px-2 py-0.5 text-[10px] leading-tight text-muted-foreground">
          {`displaying ${MAX_RENDER_SPIKES.toLocaleString()} / ${spikes.length.toLocaleString()} spikes`}
        </div>
      ) : null}

      <ParentSize className="h-full w-full min-h-0 min-w-0">
        {({ width, height }) => {
          if (width < 20 || height < 20) {
            return null
          }
          return (
            <RasterCanvasScene
              width={width}
              height={height}
              spikes={rasterSpikes}
              maxTime={maxTime}
              maxNeuronWithGap={maxNeuronWithGap}
              mapNeuronWithLayerGap={mapNeuronWithLayerGap}
              boundaryLines={boundaryLines}
              nE={nE}
              nI={nI}
              excitatoryLayerLabelPositions={excitatoryLayerLabelPositions}
              inhibitoryLayerLabelPositions={inhibitoryLayerLabelPositions}
            />
          )
        }}
      </ParentSize>
    </div>
  )
}
