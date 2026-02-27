import { useMemo } from 'react'
import {
  BaseEdge,
  Background,
  BackgroundVariant,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  type EdgeProps,
  type Edge,
  type Node,
  type NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useSimulationContext } from '@/context/simulation-context'

const INPUT_X = 64
const LAYER_X_START = 360
const LAYER_X_GAP = 280
const E_ROW_Y = 164
const I_ROW_Y = 336
const OTHER_ROW_Y = 250
const INPUT_ROW_GAP = 100
const COLOR_INPUT = '#2dd4bf'
const COLOR_EXCITATORY = '#22c55e'
const COLOR_EXCITATORY_SOFT = '#4ade80'
const COLOR_INHIBITORY = '#ef4444'
const COLOR_INHIBITORY_SOFT = '#f87171'
const COLOR_NEUTRAL = '#94a3b8'
const HANDLE_STYLE = {
  width: 6,
  height: 6,
  background: 'hsl(var(--muted-foreground) / 0.75)',
  border: '1px solid hsl(var(--background))',
}

function InputNode({ data }: NodeProps) {
  return (
    <div className="relative w-[172px] rounded-lg border border-border/80 bg-card px-3 py-2 text-xs shadow-sm">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">input</div>
      <div className="text-sm font-semibold">{String(data.label)}</div>
      <Handle
        id="right-source-top"
        type="source"
        position={Position.Right}
        style={{ ...HANDLE_STYLE, top: '32%' }}
      />
      <Handle
        id="right-source-bottom"
        type="source"
        position={Position.Right}
        style={{ ...HANDLE_STYLE, top: '68%' }}
      />
    </div>
  )
}

function PopulationNode({ data }: NodeProps) {
  return (
    <div className="relative w-[172px] rounded-lg border border-border/80 bg-card px-3 py-2 text-xs shadow-sm">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
        {String(data.kind)}
      </div>
      <div className="text-sm font-semibold">
        {String(data.label)} ({String(data.size)})
      </div>
      <Handle
        id="left-target-top"
        type="target"
        position={Position.Left}
        style={{ ...HANDLE_STYLE, top: '30%' }}
      />
      <Handle
        id="left-target-mid"
        type="target"
        position={Position.Left}
        style={{ ...HANDLE_STYLE, top: '50%' }}
      />
      <Handle
        id="left-target-bottom"
        type="target"
        position={Position.Left}
        style={{ ...HANDLE_STYLE, top: '70%' }}
      />
      <Handle
        id="left-source-top"
        type="source"
        position={Position.Left}
        style={{ ...HANDLE_STYLE, top: '30%' }}
      />
      <Handle
        id="left-source-mid"
        type="source"
        position={Position.Left}
        style={{ ...HANDLE_STYLE, top: '50%' }}
      />
      <Handle
        id="left-source-bottom"
        type="source"
        position={Position.Left}
        style={{ ...HANDLE_STYLE, top: '70%' }}
      />
      <Handle
        id="right-target-top"
        type="target"
        position={Position.Right}
        style={{ ...HANDLE_STYLE, top: '30%' }}
      />
      <Handle
        id="right-target-mid"
        type="target"
        position={Position.Right}
        style={{ ...HANDLE_STYLE, top: '50%' }}
      />
      <Handle
        id="right-target-bottom"
        type="target"
        position={Position.Right}
        style={{ ...HANDLE_STYLE, top: '70%' }}
      />
      <Handle
        id="right-source-top"
        type="source"
        position={Position.Right}
        style={{ ...HANDLE_STYLE, top: '30%' }}
      />
      <Handle
        id="right-source-mid"
        type="source"
        position={Position.Right}
        style={{ ...HANDLE_STYLE, top: '50%' }}
      />
      <Handle
        id="right-source-bottom"
        type="source"
        position={Position.Right}
        style={{ ...HANDLE_STYLE, top: '70%' }}
      />
      <Handle
        id="top-target-left"
        type="target"
        position={Position.Top}
        style={{ ...HANDLE_STYLE, left: '34%' }}
      />
      <Handle
        id="top-target-left-ie"
        type="target"
        position={Position.Top}
        style={{ ...HANDLE_STYLE, left: '42%' }}
      />
      <Handle
        id="top-target-right"
        type="target"
        position={Position.Top}
        style={{ ...HANDLE_STYLE, left: '66%' }}
      />
      <Handle
        id="top-target-right-ie"
        type="target"
        position={Position.Top}
        style={{ ...HANDLE_STYLE, left: '74%' }}
      />
      <Handle
        id="top-target-mid"
        type="target"
        position={Position.Top}
        style={{ ...HANDLE_STYLE, left: '50%' }}
      />
      <Handle
        id="top-target-mid-ie"
        type="target"
        position={Position.Top}
        style={{ ...HANDLE_STYLE, left: '58%' }}
      />
      <Handle
        id="top-source-left"
        type="source"
        position={Position.Top}
        style={{ ...HANDLE_STYLE, left: '34%' }}
      />
      <Handle
        id="top-source-left-ie"
        type="source"
        position={Position.Top}
        style={{ ...HANDLE_STYLE, left: '42%' }}
      />
      <Handle
        id="top-source-right"
        type="source"
        position={Position.Top}
        style={{ ...HANDLE_STYLE, left: '66%' }}
      />
      <Handle
        id="top-source-right-ie"
        type="source"
        position={Position.Top}
        style={{ ...HANDLE_STYLE, left: '74%' }}
      />
      <Handle
        id="top-source-mid"
        type="source"
        position={Position.Top}
        style={{ ...HANDLE_STYLE, left: '50%' }}
      />
      <Handle
        id="top-source-mid-ie"
        type="source"
        position={Position.Top}
        style={{ ...HANDLE_STYLE, left: '58%' }}
      />
      <Handle
        id="bottom-target-left"
        type="target"
        position={Position.Bottom}
        style={{ ...HANDLE_STYLE, left: '34%' }}
      />
      <Handle
        id="bottom-target-left-ie"
        type="target"
        position={Position.Bottom}
        style={{ ...HANDLE_STYLE, left: '42%' }}
      />
      <Handle
        id="bottom-target-right"
        type="target"
        position={Position.Bottom}
        style={{ ...HANDLE_STYLE, left: '66%' }}
      />
      <Handle
        id="bottom-target-right-ie"
        type="target"
        position={Position.Bottom}
        style={{ ...HANDLE_STYLE, left: '74%' }}
      />
      <Handle
        id="bottom-target-mid"
        type="target"
        position={Position.Bottom}
        style={{ ...HANDLE_STYLE, left: '50%' }}
      />
      <Handle
        id="bottom-target-mid-ie"
        type="target"
        position={Position.Bottom}
        style={{ ...HANDLE_STYLE, left: '58%' }}
      />
      <Handle
        id="bottom-source-left"
        type="source"
        position={Position.Bottom}
        style={{ ...HANDLE_STYLE, left: '34%' }}
      />
      <Handle
        id="bottom-source-right"
        type="source"
        position={Position.Bottom}
        style={{ ...HANDLE_STYLE, left: '66%' }}
      />
      <Handle
        id="bottom-source-mid"
        type="source"
        position={Position.Bottom}
        style={{ ...HANDLE_STYLE, left: '50%' }}
      />
    </div>
  )
}

const nodeTypes = {
  topologyInput: InputNode,
  topologyPopulation: PopulationNode,
}

function TopologySelfLoopEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  markerEnd,
  style,
}: EdgeProps) {
  const protrusion = 30
  const bend = 18
  const compactPath = `M ${sourceX} ${sourceY} C ${sourceX + protrusion} ${sourceY - bend}, ${
    targetX + protrusion
  } ${targetY + bend}, ${targetX} ${targetY}`
  return <BaseEdge id={id} path={compactPath} markerEnd={markerEnd} style={style} />
}

function TopologyReturnEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  data,
  markerEnd,
  style,
}: EdgeProps) {
  const span = 64
  const lift = 72
  const arc = (data as { arc?: 'up' | 'down' } | undefined)?.arc ?? 'up'
  const yDelta = arc === 'down' ? lift : -lift
  const path = `M ${sourceX} ${sourceY} C ${sourceX - span} ${sourceY + yDelta}, ${
    targetX + span
  } ${targetY + yDelta}, ${targetX} ${targetY}`
  return <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />
}

const edgeTypes = {
  topologySelf: TopologySelfLoopEdge,
  topologyReturn: TopologyReturnEdge,
}

type NodePlacement = {
  x: number
  y: number
  layer: number
}

function layerFromNodeId(nodeId: string): number | null {
  const m = nodeId.match(/(\d+)$/)
  if (!m) return null
  const layer = Number(m[1])
  return Number.isFinite(layer) && layer > 0 ? layer : null
}

function markerFor(stroke: string) {
  return { type: MarkerType.ArrowClosed, width: 14, height: 14, color: stroke }
}

function widthFromWeight(mean: number, std: number): number {
  const magnitude = Math.max(0, mean + std)
  const scaled = 1.1 + 0.85 * Math.log10(1 + magnitude * 1000)
  return Math.min(5.5, Math.max(1.1, scaled))
}

function slotFromLayer(layer: number): 'left' | 'mid' | 'right' {
  const idx = ((Math.max(1, layer) - 1) % 3 + 3) % 3
  if (idx === 0) return 'left'
  if (idx === 1) return 'mid'
  return 'right'
}

export default function TabTopology() {
  const { parameters } = useSimulationContext()
  const nodeById = useMemo(
    () => new Map(parameters.graph.nodes.map((node) => [node.id, node])),
    [parameters.graph.nodes]
  )

  const placementById = useMemo(() => {
    const placements = new Map<string, NodePlacement>()
    const inputNodes = [...parameters.graph.nodes]
      .filter((node) => node.kind === 'input')
      .sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true, sensitivity: 'base' }))
    for (const [idx, node] of inputNodes.entries()) {
      placements.set(node.id, {
        x: INPUT_X,
        y: OTHER_ROW_Y - ((inputNodes.length - 1) * INPUT_ROW_GAP) / 2 + idx * INPUT_ROW_GAP,
        layer: 0,
      })
    }

    const populationNodes = parameters.graph.nodes
      .filter((node) => node.kind === 'population')
      .sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true, sensitivity: 'base' }))
    const inferredLayers = populationNodes
      .map((node) => layerFromNodeId(node.id))
      .filter((value): value is number => value !== null)
    const fallbackLayer = 1
    let nextFallbackLayer = 1
    const eLayers = populationNodes
      .filter((node) => node.type === 'E')
      .map((node) => layerFromNodeId(node.id))
      .filter((value): value is number => value !== null)
    const minELayer = eLayers.length > 0 ? Math.min(...eLayers) : 1
    const maxELayer = eLayers.length > 0 ? Math.max(...eLayers) : 1
    const centerLayer = (minELayer + maxELayer) / 2
    const iNodes = populationNodes.filter((node) => node.type === 'I')
    const singleICenteredX = LAYER_X_START + (centerLayer - 1) * LAYER_X_GAP

    for (const node of populationNodes) {
      const inferred = layerFromNodeId(node.id)
      const layer = inferred ?? (inferredLayers.length === 0 ? nextFallbackLayer++ : fallbackLayer)
      const x =
        node.type === 'I' && iNodes.length === 1
          ? singleICenteredX
          : LAYER_X_START + (layer - 1) * LAYER_X_GAP
      const y = node.type === 'E' ? E_ROW_Y : node.type === 'I' ? I_ROW_Y : OTHER_ROW_Y
      placements.set(node.id, { x, y, layer })
    }

    return placements
  }, [parameters.graph.nodes])

  const layerColumns = useMemo(() => {
    const layers = new Set<number>()
    for (const node of parameters.graph.nodes) {
      if (node.kind !== 'population') continue
      const placement = placementById.get(node.id)
      if (placement) layers.add(placement.layer)
    }
    return [...layers].sort((a, b) => a - b)
  }, [placementById, parameters.graph.nodes])

  const nodes = useMemo<Node[]>(() => {
    const sortById = <T extends { id: string }>(items: T[]) =>
      [...items].sort((a, b) =>
        a.id.localeCompare(b.id, undefined, { numeric: true, sensitivity: 'base' })
      )
    const allNodes = sortById(parameters.graph.nodes).map((node) => {
      const placement = placementById.get(node.id) ?? { x: 0, y: 0 }
      return { node, x: placement.x, y: placement.y }
    })

    return allNodes.map(({ node, x, y }) => {
      return {
        id: node.id,
        type: node.kind === 'input' ? 'topologyInput' : 'topologyPopulation',
        draggable: false,
        selectable: false,
        position: { x, y },
        data:
          node.kind === 'input'
            ? { label: node.id }
            : { label: node.id, kind: node.type, size: node.size },
      } as Node
    })
  }, [parameters.graph.nodes, placementById])
  const edges = useMemo<Edge[]>(() => {
    return parameters.graph.edges.map((edge): Edge => {
      const isSelf = edge.from === edge.to
      if (isSelf) {
        const sourceHandle = 'right-source-top'
        const targetHandle = 'right-target-bottom'
        const selfStroke =
          edge.kind === 'IE' || edge.kind === 'II' ? COLOR_INHIBITORY_SOFT : COLOR_EXCITATORY_SOFT
        const weightedWidth = widthFromWeight(edge.w.mean, edge.w.std)
        return {
          id: edge.id,
          source: edge.from,
          target: edge.to,
          sourceHandle,
          targetHandle,
          type: 'topologySelf',
          animated: edge.kind === 'input',
          selectable: false,
          style: {
            strokeWidth: Math.max(2.1, weightedWidth),
            stroke: selfStroke,
          },
          markerEnd: markerFor(selfStroke),
        }
      }
      let sourceHandle = 'right-source-mid'
      let targetHandle = 'left-target-mid'
      const sourceLayer = placementById.get(edge.from)?.layer ?? 0
      const targetLayer = placementById.get(edge.to)?.layer ?? 0
      let stroke = COLOR_NEUTRAL

      if (edge.kind === 'input') {
        sourceHandle = 'right-source-top'
        targetHandle = 'left-target-mid'
        stroke = COLOR_INPUT
      } else if (edge.kind === 'EI') {
        const slot = slotFromLayer(sourceLayer)
        sourceHandle = `bottom-source-${slot}`
        targetHandle = `top-target-${slot}`
        stroke = COLOR_EXCITATORY
      } else if (edge.kind === 'IE') {
        const slot = slotFromLayer(targetLayer)
        sourceHandle = `top-source-${slot}-ie`
        targetHandle = `bottom-target-${slot}-ie`
        stroke = COLOR_INHIBITORY
      } else if (edge.kind === 'EE') {
        sourceHandle = sourceLayer <= targetLayer ? 'right-source-mid' : 'left-source-mid'
        targetHandle = sourceLayer <= targetLayer ? 'left-target-mid' : 'right-target-mid'
        stroke = COLOR_EXCITATORY
      } else if (edge.kind === 'II') {
        sourceHandle = sourceLayer <= targetLayer ? 'right-source-mid' : 'left-source-mid'
        targetHandle = sourceLayer <= targetLayer ? 'left-target-mid' : 'right-target-mid'
        stroke = COLOR_INHIBITORY
      }
      const strokeWidth = Math.max(edge.kind === 'input' ? 1.8 : 1.2, widthFromWeight(edge.w.mean, edge.w.std))
      const opacity = edge.kind === 'input' ? 0.95 : 0.9

      return {
        id: edge.id,
        source: edge.from,
        target: edge.to,
        sourceHandle,
        targetHandle,
        type: edge.kind === 'input' ? 'smoothstep' : 'straight',
        animated: false,
        markerEnd: markerFor(stroke),
        selectable: false,
        style: {
          strokeWidth,
          stroke,
          opacity,
        },
      }
    })
  }, [nodeById, placementById, parameters.graph.edges])

  return (
    <div className="relative h-full w-full overflow-hidden rounded-lg border border-border/70 bg-card/65">
      <div className="pointer-events-none absolute left-3 top-3 z-20 rounded-md border border-border/70 bg-background/92 px-2.5 py-2 text-[11px] text-muted-foreground shadow-sm">
        <div className="flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-5" style={{ backgroundColor: COLOR_INPUT }} />
          <span>Input</span>
        </div>
        <div className="mt-1 flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-5" style={{ backgroundColor: COLOR_EXCITATORY }} />
          <span>Excitatory</span>
        </div>
        <div className="mt-1 flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-5" style={{ backgroundColor: COLOR_INHIBITORY }} />
          <span>Inhibitory</span>
        </div>
      </div>
      {layerColumns.length > 0 ? (
        <div className="pointer-events-none absolute inset-x-0 top-3 z-10">
          {layerColumns.map((layer) => (
            <div
              key={`layer-column-${layer}`}
              className="absolute -translate-x-1/2 rounded-sm border border-border/50 bg-background/75 px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground"
              style={{
                left: LAYER_X_START + (layer - 1) * LAYER_X_GAP + 86,
              }}
            >
              Layer {layer}
            </div>
          ))}
        </div>
      ) : null}
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        fitViewOptions={{ padding: 0.22 }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
      >
        <Background variant={BackgroundVariant.Dots} gap={18} size={1} />
      </ReactFlow>
    </div>
  )
}
