import type { ParametersState } from '@/hooks/use-parameters'

type TargetedInputOverlay = {
  text: string
  emphasized: boolean
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(1, value))
}

export function getTargetedInputOverlay(
  parameters: ParametersState
): TargetedInputOverlay | null {
  const primaryInputId = parameters.graph.nodes.find((node) => node.kind === 'input')?.id
  if (!primaryInputId) return null
  const inputProgram = parameters.graph.inputPrograms[primaryInputId]
  if (!inputProgram) return null
  const inputType = inputProgram.mode
  if (inputType !== 'pulse' && inputType !== 'pulses') {
    return null
  }

  const nE = Math.max(0, Math.round(parameters.config.N_E.value))
  const nI = Math.max(0, Math.round(parameters.config.N_I.value))
  const enabled = inputProgram.targeted_subset_enabled
  if (!enabled) {
    return {
      text: `Target full pop: E ${nE}, I ${nI}`,
      emphasized: false,
    }
  }

  const fraction = clamp01(inputProgram.target_fraction)
  const population = inputProgram.target_population
  const includeE = population === 'all' || population === 'e'
  const includeI = population === 'all' || population === 'i'
  const selectedE = includeE ? Math.floor(nE * fraction) : 0
  const selectedI = includeI ? Math.floor(nI * fraction) : 0
  const percent = Math.round(fraction * 100)
  return {
    text: `Target subset (${percent}%): E ${selectedE}, I ${selectedI}`,
    emphasized: true,
  }
}
