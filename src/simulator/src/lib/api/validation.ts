import type { ParametersState } from '@/hooks/use-parameters'

export function getRunValidationError(parameters: ParametersState): string | null {
  const { nodes, edges } = parameters.graph

  if (nodes.length === 0) {
    return 'Add at least one node before running.'
  }

  const nodeIds = new Set<string>()
  let inputCount = 0
  let ePopulationCount = 0

  for (const node of nodes) {
    const id = node.id.trim()
    if (!id) {
      return 'Every node must have a non-empty id.'
    }
    if (nodeIds.has(id)) {
      return `Duplicate node id '${id}'.`
    }
    nodeIds.add(id)

    if (node.kind === 'input') {
      inputCount += 1
      if (node.size !== 0) {
        return `Input node '${id}' must have size 0.`
      }
      continue
    }

    if (node.size <= 0) {
      return `Population node '${id}' must have size > 0.`
    }
    if (node.type === 'E') {
      ePopulationCount += 1
    }
  }

  if (inputCount <= 0) {
    return 'Add at least one input node before running.'
  }
  if (ePopulationCount <= 0) {
    return 'Add at least one E population before running.'
  }

  const edgeIds = new Set<string>()
  const inputEdgeSources = new Set<string>()
  for (const edge of edges) {
    const edgeId = edge.id.trim()
    if (!edgeId) {
      return 'Every edge must have a non-empty id.'
    }
    if (edgeIds.has(edgeId)) {
      return `Duplicate edge id '${edgeId}'.`
    }
    edgeIds.add(edgeId)

    if (!nodeIds.has(edge.from)) {
      return `Edge '${edgeId}' has unknown source node '${edge.from}'.`
    }
    if (!nodeIds.has(edge.to)) {
      return `Edge '${edgeId}' has unknown target node '${edge.to}'.`
    }
    if (edge.kind === 'input') {
      inputEdgeSources.add(edge.from)
    }
  }

  for (const node of nodes) {
    if (node.kind !== 'input') continue
    if (!inputEdgeSources.has(node.id)) {
      return `Input node '${node.id}' must connect to at least one input edge.`
    }
  }

  return null
}
