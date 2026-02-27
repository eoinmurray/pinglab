import { useState } from 'react'

export type InputType = 'ramp' | 'pulse' | 'pulses' | 'tonic' | 'sine' | 'external_spike_train'
export type NeuronModel = 'lif' | 'mqif'
export type InputTargetPopulation = 'all' | 'e' | 'i'
export type InputTargetStrategy = 'random' | 'first'
export type ConfigPreset =
  | 'ai'
  | 'empty'
  | 'ping'
  | 'input-spike-e'
  | 'three-layer'
  | 'three-layer-local-i'

type BaseParameterConfig<TValue> = {
  label: string
  info: string
  defaultValue: TValue
  value: TValue
}

export type SliderParameterConfig = BaseParameterConfig<number> & {
  min: number
  max: number
  step: number
}

export type SelectParameterOption<TValue extends string> = {
  value: TValue
  label: string
}

export type SelectParameterConfig<TValue extends string> = BaseParameterConfig<TValue> & {
  options: SelectParameterOption<TValue>[]
}

export type SwitchParameterConfig = BaseParameterConfig<boolean>

type ConfigSection = 'Core' | 'Timings' | 'Neuron' | 'Heterogeneity'

type ConfigParameterDefinition = {
  key: string
  section: ConfigSection
  label: string
  info: string
  defaultValue: number
  min: number
  max: number
  step: number
  integer?: boolean
}

export const CONFIG_PARAMETER_DEFINITIONS = [
  { key: 'dt', section: 'Core', label: 'dt', info: 'Simulation timestep in milliseconds.', defaultValue: 0.1, min: 0.05, max: 1.0, step: 0.05 },
  { key: 'T', section: 'Core', label: 'T', info: 'Total simulation duration in milliseconds.', defaultValue: 1000, min: 200, max: 5000, step: 50 },
  { key: 'N_E', section: 'Core', label: 'N_E', info: 'Number of excitatory neurons.', defaultValue: 400, min: 0, max: 800, step: 50, integer: true },
  { key: 'N_I', section: 'Core', label: 'N_I', info: 'Number of inhibitory neurons.', defaultValue: 100, min: 0, max: 200, step: 1, integer: true },
  { key: 'seed', section: 'Core', label: 'seed', info: 'Simulation seed for network-level randomness.', defaultValue: 0, min: 0, max: 1000, step: 1, integer: true },

  { key: 'tau_ampa', section: 'Timings', label: 'tau_ampa', info: 'Excitatory synaptic conductance decay time constant.', defaultValue: 2.0, min: 0.01, max: 10, step: 0.01 },
  { key: 'tau_gaba', section: 'Timings', label: 'tau_gaba', info: 'Inhibitory synaptic conductance decay time constant.', defaultValue: 6.5, min: 0.01, max: 15, step: 0.01 },
  { key: 'delay_ee', section: 'Timings', label: 'delay_ee', info: 'Synaptic delay for E to E projections (ms).', defaultValue: 0.5, min: 0.01, max: 5.0, step: 0.01 },
  { key: 'delay_ei', section: 'Timings', label: 'delay_ei', info: 'Synaptic delay for E to I projections (ms).', defaultValue: 0.5, min: 0.01, max: 5.0, step: 0.01 },
  { key: 'delay_ie', section: 'Timings', label: 'delay_ie', info: 'Synaptic delay for I to E projections (ms).', defaultValue: 1.2, min: 0.01, max: 5.0, step: 0.01 },
  { key: 'delay_ii', section: 'Timings', label: 'delay_ii', info: 'Synaptic delay for I to I projections (ms).', defaultValue: 0.5, min: 0.01, max: 5.0, step: 0.01 },
  { key: 't_ref_E', section: 'Timings', label: 't_ref_E', info: 'Refractory period for excitatory neurons (ms).', defaultValue: 3.0, min: 0, max: 5, step: 0.1 },
  { key: 't_ref_I', section: 'Timings', label: 't_ref_I', info: 'Refractory period for inhibitory neurons (ms).', defaultValue: 1.5, min: 0, max: 5, step: 0.1 },

  { key: 'V_init', section: 'Neuron', label: 'V_init', info: 'Initial membrane voltage.', defaultValue: -65.0, min: -80, max: -50, step: 1 },
  { key: 'E_L', section: 'Neuron', label: 'E_L', info: 'Leak reversal potential.', defaultValue: -65.0, min: -80, max: -50, step: 1 },
  { key: 'E_e', section: 'Neuron', label: 'E_e', info: 'Excitatory reversal potential.', defaultValue: 0.0, min: -10, max: 10, step: 1 },
  { key: 'E_i', section: 'Neuron', label: 'E_i', info: 'Inhibitory reversal potential.', defaultValue: -80.0, min: -100, max: -60, step: 1 },
  { key: 'C_m_E', section: 'Neuron', label: 'C_m_E', info: 'Membrane capacitance for excitatory cells.', defaultValue: 1.0, min: 0.2, max: 2.0, step: 0.1 },
  { key: 'g_L_E', section: 'Neuron', label: 'g_L_E', info: 'Leak conductance for excitatory cells.', defaultValue: 0.05, min: 0.01, max: 0.2, step: 0.005 },
  { key: 'C_m_I', section: 'Neuron', label: 'C_m_I', info: 'Membrane capacitance for inhibitory cells.', defaultValue: 1.0, min: 0.2, max: 2.0, step: 0.1 },
  { key: 'g_L_I', section: 'Neuron', label: 'g_L_I', info: 'Leak conductance for inhibitory cells.', defaultValue: 0.1, min: 0.01, max: 0.2, step: 0.005 },
  { key: 'V_th', section: 'Neuron', label: 'V_th', info: 'Spike threshold voltage.', defaultValue: -50.0, min: -70, max: -40, step: 1 },
  { key: 'V_reset', section: 'Neuron', label: 'V_reset', info: 'Post-spike reset voltage.', defaultValue: -65.0, min: -80, max: -50, step: 1 },

  { key: 'g_L_heterogeneity_sd', section: 'Heterogeneity', label: 'g_L_het_sd', info: 'Standard deviation for leak conductance heterogeneity.', defaultValue: 0.15, min: 0, max: 1.0, step: 0.05 },
  { key: 'C_m_heterogeneity_sd', section: 'Heterogeneity', label: 'C_m_het_sd', info: 'Standard deviation for membrane capacitance heterogeneity.', defaultValue: 0.1, min: 0, max: 1.0, step: 0.05 },
  { key: 'V_th_heterogeneity_sd', section: 'Heterogeneity', label: 'V_th_het_sd', info: 'Standard deviation for threshold heterogeneity.', defaultValue: 1.2, min: 0, max: 3.0, step: 0.1 },
  { key: 't_ref_heterogeneity_sd', section: 'Heterogeneity', label: 't_ref_het_sd', info: 'Standard deviation for refractory period heterogeneity.', defaultValue: 0.3, min: 0, max: 2.0, step: 0.1 },
] as const satisfies readonly ConfigParameterDefinition[]

export type ConfigNumericKey = (typeof CONFIG_PARAMETER_DEFINITIONS)[number]['key']

export const CONFIG_SECTIONS: ConfigSection[] = ['Core', 'Timings', 'Neuron', 'Heterogeneity']

export const INPUT_TYPE_OPTIONS: SelectParameterOption<InputType>[] = [
  { value: 'ramp', label: 'Ramp' },
  { value: 'tonic', label: 'Tonic' },
  { value: 'sine', label: 'Sine' },
  { value: 'external_spike_train', label: 'External Spike Train' },
  { value: 'pulse', label: 'Pulse' },
  { value: 'pulses', label: 'Pulses' },
]

export const NEURON_MODEL_OPTIONS: SelectParameterOption<NeuronModel>[] = [
  { value: 'lif', label: 'LIF' },
  { value: 'mqif', label: 'MQIF' },
]

export const INPUT_TARGET_POPULATION_OPTIONS: SelectParameterOption<InputTargetPopulation>[] = [
  { value: 'all', label: 'E + I' },
  { value: 'e', label: 'E only' },
  { value: 'i', label: 'I only' },
]

export const INPUT_TARGET_STRATEGY_OPTIONS: SelectParameterOption<InputTargetStrategy>[] = [
  { value: 'random', label: 'Random' },
  { value: 'first', label: 'First neurons' },
]

export let CONFIG_PRESET_OPTIONS: SelectParameterOption<ConfigPreset>[] = []

export type InputNumericKey =
  | 'mean'
  | 'std'
  | 'I_E_start'
  | 'I_E_end'
  | 'I_I_start'
  | 'I_I_end'
  | 'I_E_base'
  | 'I_I_base'
  | 'noise_std_E'
  | 'noise_std_I'
  | 'seed'
  | 'sine_freq_hz'
  | 'sine_amp'
  | 'sine_y_offset'
  | 'sine_phase'
  | 'sine_phase_offset_i'
  | 'lambda0_hz'
  | 'mod_depth'
  | 'envelope_freq_hz'
  | 'phase_rad'
  | 'w_in'
  | 'tau_in_ms'
  | 'pulse_t_ms'
  | 'pulse_width_ms'
  | 'pulse_interval_ms'
  | 'pulse_amp_E'
  | 'pulse_amp_I'

export type WeightBlockKey = 'ee' | 'ei' | 'ie' | 'ii'
export const WEIGHT_BLOCK_KEYS: WeightBlockKey[] = ['ee', 'ei', 'ie', 'ii']
export type GraphNodeKind = 'input' | 'population'
export type GraphPopulationType = 'E' | 'I'
export type GraphEdgeKind = 'input' | 'EE' | 'EI' | 'IE' | 'II'

export type GraphNodeState = {
  id: string
  kind: GraphNodeKind
  type: InputType | GraphPopulationType
  size: number
}

export type GraphEdgeState = {
  id: string
  from: string
  to: string
  kind: GraphEdgeKind
  w: {
    mean: number
    std: number
  }
  delay_ms: number
}

export type InputProgramState = {
  mode: InputType
  input_population: InputTargetPopulation
  values: Record<InputNumericKey, number>
  targeted_subset_enabled: boolean
  target_population: InputTargetPopulation
  target_strategy: InputTargetStrategy
  target_fraction: number
  target_seed: number
}

type WeightNumericKey = 'mean' | 'std'

export type MqifNumericKey = 'a' | 'Vr' | 'w_a' | 'w_Vr' | 'w_tau'

type WeightBlockState = {
  mean: SliderParameterConfig
  std: SliderParameterConfig
}

export type ParametersState = {
  performanceMode: SwitchParameterConfig
  downsampleEnabled: SwitchParameterConfig
  useArrowTransport: SwitchParameterConfig
  burnInMs: SliderParameterConfig
  config: Record<ConfigNumericKey, SliderParameterConfig>
  neuronModel: SelectParameterConfig<NeuronModel>
  mqif: Record<MqifNumericKey, SliderParameterConfig>
  inputs: {
    inputType: SelectParameterConfig<InputType>
    inputPopulation: SelectParameterConfig<InputTargetPopulation>
    values: Record<InputNumericKey, SliderParameterConfig>
    targetedSubsetEnabled: SwitchParameterConfig
    targetPopulation: SelectParameterConfig<InputTargetPopulation>
    targetStrategy: SelectParameterConfig<InputTargetStrategy>
    targetFraction: SliderParameterConfig
    targetSeed: SliderParameterConfig
  }
  weights: {
    blocks: Record<WeightBlockKey, WeightBlockState>
    clampMin: SliderParameterConfig
    seed: SliderParameterConfig
  }
  graph: {
    nodes: GraphNodeState[]
    edges: GraphEdgeState[]
    inputPrograms: Record<string, InputProgramState>
  }
}

function slider(label: string, info: string, def: number, min: number, max: number, step: number): SliderParameterConfig {
  return { label, info, defaultValue: def, value: def, min, max, step }
}

function buildDefaultConfigParameters(): Record<ConfigNumericKey, SliderParameterConfig> {
  return Object.fromEntries(
    CONFIG_PARAMETER_DEFINITIONS.map((definition) => [
      definition.key,
      slider(definition.label, definition.info, definition.defaultValue, definition.min, definition.max, definition.step),
    ])
  ) as Record<ConfigNumericKey, SliderParameterConfig>
}

function buildDefaultInputs(): ParametersState['inputs'] {
  return {
    inputType: {
      label: 'Input Type',
      info: 'Input profile over time.',
      defaultValue: 'tonic',
      value: 'tonic',
      options: INPUT_TYPE_OPTIONS,
    },
    inputPopulation: {
      label: 'Input Population',
      info: 'Which population receives external input.',
      defaultValue: 'e',
      value: 'e',
      options: INPUT_TARGET_POPULATION_OPTIONS,
    },
    values: {
      mean: slider('mean', 'Input mean drive level applied by this input node.', 0.0, 0.0, 4.0, 0.01),
      std: slider('std', 'Input noise standard deviation for this input node.', 0.0, 0.0, 5.0, 0.05),
      I_E_start: slider('I_E_start', 'Ramp start current for excitatory population.', 0.0, 0.0, 4.0, 0.01),
      I_E_end: slider('I_E_end', 'Ramp end current for excitatory population.', 0.0, 0.0, 4.0, 0.01),
      I_I_start: slider('I_I_start', 'Ramp start current for inhibitory population.', 0.0, 0.0, 4.0, 0.01),
      I_I_end: slider('I_I_end', 'Ramp end current for inhibitory population.', 0.0, 0.0, 4.0, 0.01),
      I_E_base: slider('I_E_base', 'Baseline current for excitatory population.', 0.0, 0.0, 4.0, 0.01),
      I_I_base: slider('I_I_base', 'Baseline current for inhibitory population.', 0.0, 0.0, 4.0, 0.01),
      noise_std_E: slider('noise_std_E', 'Input noise standard deviation for E.', 2.0, 0.0, 5.0, 0.05),
      noise_std_I: slider('noise_std_I', 'Input noise standard deviation for I.', 0.0, 0.0, 5.0, 0.05),
      seed: slider('input_seed', 'Input generation seed.', 0, 0, 20, 1),
      sine_freq_hz: slider('sine_freq_hz', 'Sine frequency in Hz.', 5, 0, 200, 1),
      sine_amp: slider('sine_amp', 'Sine oscillation amplitude.', 2, 0, 5, 0.05),
      sine_y_offset: slider('sine_y_offset', 'Additive vertical offset for sine input.', 0, -5, 5, 0.05),
      sine_phase: slider('sine_phase', 'Sine phase for E (radians).', 0, -6.2832, 6.2832, 0.05),
      sine_phase_offset_i: slider('sine_phase_offset_i', 'Phase offset for I relative to E (radians).', 0, -6.2832, 6.2832, 0.05),
      lambda0_hz: slider('lambda0_hz', 'Base external spike rate in Hz.', 30, 0, 300, 1),
      mod_depth: slider('mod_depth', 'Envelope modulation depth for external spikes.', 0.5, 0, 1, 0.01),
      envelope_freq_hz: slider('envelope_freq_hz', 'Envelope frequency for external spikes in Hz.', 5, 0, 40, 0.5),
      phase_rad: slider('phase_rad', 'Envelope phase offset in radians.', 0, -6.2832, 6.2832, 0.05),
      w_in: slider('w_in', 'External synaptic weight to E cells.', 0.25, 0, 4, 0.01),
      tau_in_ms: slider('tau_in_ms', 'External synaptic decay time constant (ms).', 3, 0.5, 30, 0.5),
      pulse_t_ms: slider('pulse_t_ms', 'Pulse start time.', 0, 0, 1000, 10),
      pulse_width_ms: slider('pulse_width_ms', 'Pulse width.', 1, 1, 200, 5),
      pulse_interval_ms: slider('pulse_interval_ms', 'Pulse interval for repeated pulses.', 10, 10, 1000, 10),
      pulse_amp_E: slider('pulse_amp_E', 'Pulse amplitude for E.', 0, 0, 5, 0.1),
      pulse_amp_I: slider('pulse_amp_I', 'Pulse amplitude for I.', 0, 0, 5, 0.1),
    },
    targetedSubsetEnabled: {
      label: 'Targeted Subset',
      info: 'Apply pulse input to a selected subset instead of the full population.',
      defaultValue: false,
      value: false,
    },
    targetPopulation: {
      label: 'Target Population',
      info: 'Choose which population receives the targeted pulse input.',
      defaultValue: 'all',
      value: 'all',
      options: INPUT_TARGET_POPULATION_OPTIONS,
    },
    targetStrategy: {
      label: 'Subset Strategy',
      info: 'How neurons are selected for the targeted subset.',
      defaultValue: 'random',
      value: 'random',
      options: INPUT_TARGET_STRATEGY_OPTIONS,
    },
    targetFraction: slider('target_fraction', 'Fraction of selected population to stimulate.', 0.25, 0.0, 1.0, 0.01),
    targetSeed: slider('target_seed', 'Subset selection seed.', 0, 0, 1000, 1),
  }
}

function buildDefaultInputProgram(mode: InputType = 'tonic'): InputProgramState {
  return {
    mode,
    input_population: mode === 'external_spike_train' ? 'e' : 'all',
    values: {
      mean: 0.0,
      std: 0.0,
      I_E_start: 0.0,
      I_E_end: 0.0,
      I_I_start: 0.0,
      I_I_end: 0.0,
      I_E_base: 0.0,
      I_I_base: 0.0,
      noise_std_E: 2.0,
      noise_std_I: 0.0,
      seed: 0,
      sine_freq_hz: 5,
      sine_amp: 2,
      sine_y_offset: 0,
      sine_phase: 0,
      sine_phase_offset_i: 0,
      lambda0_hz: 30,
      mod_depth: 0.5,
      envelope_freq_hz: 5,
      phase_rad: 0,
      w_in: 0.25,
      tau_in_ms: 3,
      pulse_t_ms: 0,
      pulse_width_ms: 1,
      pulse_interval_ms: 10,
      pulse_amp_E: 0,
      pulse_amp_I: 0,
    },
    targeted_subset_enabled: false,
    target_population: 'all',
    target_strategy: 'random',
    target_fraction: 0.25,
    target_seed: 0,
  }
}

function buildDefaultWeightBlock(
  labelPrefix: string,
  defaultMean: number,
  defaultStd: number
): WeightBlockState {
  return {
    mean: slider(`${labelPrefix} mean`, 'Distribution mean.', defaultMean, 0.0, 1.0, 0.0002),
    std: slider(`${labelPrefix} std`, 'Distribution standard deviation.', defaultStd, 0.0, 0.5, 0.0002),
  }
}

function buildDefaultWeights(): ParametersState['weights'] {
  const eeBlock = buildDefaultWeightBlock('E->E', 0.0, 0.0)
  return {
    blocks: {
      ee: eeBlock,
      ei: buildDefaultWeightBlock('E->I', 0.0006, 0.2),
      ie: buildDefaultWeightBlock('I->E', 0.0006, 0.04),
      ii: buildDefaultWeightBlock('I->I', 0.0, 0.0),
    },
    clampMin: slider('clamp_min', 'Minimum clamp applied to all weights.', 0.0, 0.0, 0.2, 0.005),
    seed: slider('weights_seed', 'Weight matrix construction seed.', 0, 0, 20, 1),
  }
}

function buildDefaultGraph(): ParametersState['graph'] {
  return {
    nodes: [
      { id: 'input_1', kind: 'input', type: 'tonic', size: 0 },
      { id: 'E', kind: 'population', type: 'E', size: 400 },
      { id: 'I', kind: 'population', type: 'I', size: 100 },
    ],
    edges: [
      { id: 'in_to_e', from: 'input_1', to: 'E', kind: 'input', w: { mean: 1, std: 0 }, delay_ms: 0.5 },
      { id: 'e_to_e', from: 'E', to: 'E', kind: 'EE', w: { mean: 0, std: 0 }, delay_ms: 0.5 },
      { id: 'e_to_i', from: 'E', to: 'I', kind: 'EI', w: { mean: 0.0006, std: 0.2 }, delay_ms: 0.5 },
      { id: 'i_to_e', from: 'I', to: 'E', kind: 'IE', w: { mean: 0.0006, std: 0.04 }, delay_ms: 1.2 },
      { id: 'i_to_i', from: 'I', to: 'I', kind: 'II', w: { mean: 0, std: 0 }, delay_ms: 0.5 },
    ],
    inputPrograms: {
      input_1: buildDefaultInputProgram('tonic'),
    },
  }
}

const DEFAULT_PARAMETERS: ParametersState = {
  performanceMode: {
    label: 'Performance Mode',
    info: 'Run with reduced instrumentation for faster simulation.',
    defaultValue: false,
    value: false,
  },
  downsampleEnabled: {
    label: 'Downsample Spikes',
    info: 'Cap spikes returned by API for responsive plotting.',
    defaultValue: true,
    value: true,
  },
  useArrowTransport: {
    label: 'Use Arrow Transport',
    info: 'Use Apache Arrow for API responses. Disable to force JSON.',
    defaultValue: true,
    value: true,
  },
  burnInMs: slider('burn_in_ms', 'Discard this amount of initial simulation time.', 200, 0, 1000, 25),
  config: buildDefaultConfigParameters(),
  neuronModel: {
    label: 'Neuron Model',
    info: 'Neuron dynamics model.',
    defaultValue: 'lif',
    value: 'lif',
    options: NEURON_MODEL_OPTIONS,
  },
  mqif: {
    a: slider('mqif_a', 'MQIF parameter a.', 0.02, 0.001, 0.2, 0.001),
    Vr: slider('mqif_Vr', 'MQIF resting/reset reference voltage.', -55, -80, -40, 0.5),
    w_a: slider('mqif_w_a', 'MQIF adaptation coupling.', 0.02, 0.001, 0.2, 0.001),
    w_Vr: slider('mqif_w_Vr', 'MQIF adaptation voltage reference.', -55, -80, -40, 0.5),
    w_tau: slider('mqif_w_tau', 'MQIF adaptation time constant (ms).', 100, 1, 400, 1),
  },
  inputs: buildDefaultInputs(),
  weights: buildDefaultWeights(),
  graph: buildDefaultGraph(),
}

type ConfigParameterMeta = (typeof CONFIG_PARAMETER_DEFINITIONS)[number]
const CONFIG_PARAMETER_META = Object.fromEntries(
  CONFIG_PARAMETER_DEFINITIONS.map((definition) => [definition.key, definition])
) as Record<ConfigNumericKey, ConfigParameterMeta>

type ParametersInitial = Partial<{
  performanceMode: boolean
  downsampleEnabled: boolean
  useArrowTransport: boolean
  burnInMs: number
  config: Partial<Record<ConfigNumericKey, number>>
  inputType: InputType
}>

function clampSliderConfig(param: SliderParameterConfig, integer = false): SliderParameterConfig {
  const raw = integer ? Math.round(param.value) : param.value
  const clamped = Math.min(param.max, Math.max(param.min, raw))
  return { ...param, value: clamped }
}

function sanitizePresetParameters(state: ParametersState): ParametersState {
  const next: ParametersState = {
    ...state,
    burnInMs: clampSliderConfig(state.burnInMs),
    config: { ...state.config },
    mqif: { ...state.mqif },
    inputs: {
      ...state.inputs,
      values: { ...state.inputs.values },
      targetFraction: clampSliderConfig(state.inputs.targetFraction),
      targetSeed: clampSliderConfig(state.inputs.targetSeed, true),
    },
    weights: {
      ...state.weights,
      blocks: {
        ee: { ...state.weights.blocks.ee },
        ei: { ...state.weights.blocks.ei },
        ie: { ...state.weights.blocks.ie },
        ii: { ...state.weights.blocks.ii },
      },
      clampMin: clampSliderConfig(state.weights.clampMin),
      seed: clampSliderConfig(state.weights.seed, true),
    },
    graph: {
      nodes: state.graph.nodes.map((node) => ({ ...node })),
      edges: state.graph.edges.map((edge) => ({ ...edge, w: { ...edge.w } })),
      inputPrograms: Object.fromEntries(
        Object.entries(state.graph.inputPrograms).map(([inputId, program]) => [
          inputId,
          {
            ...program,
            values: { ...program.values },
          },
        ])
      ),
    },
  }

  ;(Object.keys(next.config) as ConfigNumericKey[]).forEach((key) => {
    const meta = CONFIG_PARAMETER_META[key]
    const integer = 'integer' in meta && Boolean(meta.integer)
    next.config[key] = clampSliderConfig(next.config[key], integer)
  })
  ;(Object.keys(next.mqif) as MqifNumericKey[]).forEach((key) => {
    next.mqif[key] = clampSliderConfig(next.mqif[key])
  })
  ;(Object.keys(next.inputs.values) as InputNumericKey[]).forEach((key) => {
    next.inputs.values[key] = clampSliderConfig(next.inputs.values[key], key === 'seed')
  })
  ;(['ee', 'ei', 'ie', 'ii'] as WeightBlockKey[]).forEach((blockKey) => {
    const block = next.weights.blocks[blockKey]
    next.weights.blocks[blockKey] = {
      ...block,
      mean: clampSliderConfig(block.mean),
      std: clampSliderConfig(block.std),
    }
  })
  next.graph.nodes = next.graph.nodes.map((node) => ({
    ...node,
    size: Math.max(0, Math.round(node.size)),
  }))
  next.graph.edges = next.graph.edges.map((edge) => ({
    ...edge,
    delay_ms: Math.max(0.01, edge.delay_ms),
    w: {
      mean: Math.max(0, edge.w.mean),
      std: Math.max(0, edge.w.std),
    },
  }))
  next.graph.nodes.forEach((node) => {
    if (node.kind !== 'input') return
    const existing = next.graph.inputPrograms[node.id]
    const program = existing ? { ...existing, values: { ...existing.values } } : buildDefaultInputProgram(node.type as InputType)
    if (!program.mode) {
      program.mode = node.type as InputType
    }
    ;(Object.keys(next.inputs.values) as InputNumericKey[]).forEach((key) => {
      const meta = next.inputs.values[key]
      const raw = key === 'seed' ? Math.round(program.values[key]) : program.values[key]
      const fallback = key === 'seed' ? Math.round(meta.defaultValue) : meta.defaultValue
      const finite = Number.isFinite(raw) ? raw : fallback
      program.values[key] = Math.min(meta.max, Math.max(meta.min, finite))
    })
    if (program.mode === 'external_spike_train') {
      program.input_population = 'e'
    }
    program.target_fraction = Math.min(1, Math.max(0, program.target_fraction))
    program.target_seed = Math.max(0, Math.round(program.target_seed))
    next.graph.inputPrograms[node.id] = program
  })
  const validInputIds = new Set(
    next.graph.nodes.filter((node) => node.kind === 'input').map((node) => node.id)
  )
  Object.keys(next.graph.inputPrograms).forEach((inputId) => {
    if (!validInputIds.has(inputId)) {
      delete next.graph.inputPrograms[inputId]
    }
  })

  return next
}

type GraphPresetSpec = {
  schema_version: 'pinglab-graph.v1'
  sim: {
    dt_ms: number
    T_ms: number
    seed: number
    neuron_model: NeuronModel
  }
  execution: {
    performance_mode: boolean
    max_spikes: number | null
    burn_in_ms: number
  }
  biophysics: Partial<Record<ConfigNumericKey, number>> & {
    mqif_a?: number[]
    mqif_Vr?: number[]
    mqif_w_a?: number[]
    mqif_w_Vr?: number[]
    mqif_w_tau?: number[]
  }
  constraints?: Record<string, unknown>
  nodes: Array<{ id: string; kind: 'input' | 'population'; type: string; size: number }>
  edges: Array<{
    id: string
    kind: 'input' | 'EE' | 'EI' | 'IE' | 'II'
    from: string
    to: string
    w?: { mean: number; std: number }
    delay_ms?: number
  }>
  inputs: Record<string, Partial<Record<InputNumericKey, number>> & {
    mode: InputType
    input_population: InputTargetPopulation
    target_population?: InputTargetPopulation
    target_strategy?: InputTargetStrategy
    targeted_subset_enabled?: boolean
    target_fraction?: number
    target_seed?: number
  }>
}

const PRESET_META: Record<ConfigPreset, { label: string; order: number }> = {
  empty: { label: 'Empty', order: 10 },
  ping: { label: 'PING', order: 20 },
  'three-layer': { label: '3-Layer Routing', order: 30 },
  'three-layer-local-i': { label: '3-Layer Local I', order: 40 },
  'input-spike-e': { label: 'Input-Spike-E', order: 50 },
  ai: { label: 'AI', order: 60 },
}

const PRESET_FILE_MODULES = import.meta.glob('../../../presets/*.json', {
  eager: true,
  import: 'default',
}) as Record<string, GraphPresetSpec>

const GRAPH_PRESETS = Object.fromEntries(
  Object.entries(PRESET_FILE_MODULES)
    .map(([path, preset]) => {
      const match = path.match(/\/([^/]+)\.json$/)
      const id = (match?.[1] ?? '') as ConfigPreset
      if (!id || !(id in PRESET_META)) return null
      return [id, preset] as const
    })
    .filter((entry): entry is readonly [ConfigPreset, GraphPresetSpec] => entry !== null)
) as Record<ConfigPreset, GraphPresetSpec>

if (Object.keys(GRAPH_PRESETS).length === 0) {
  throw new Error('No simulator presets found in ./presets')
}

CONFIG_PRESET_OPTIONS = (Object.keys(PRESET_META) as ConfigPreset[])
  .filter((id) => id in GRAPH_PRESETS)
  .sort((a, b) => PRESET_META[a].order - PRESET_META[b].order)
  .map((id) => ({
    value: id,
    label: PRESET_META[id].label,
  }))

function cloneParametersState(source: ParametersState): ParametersState {
  return {
    ...source,
    performanceMode: { ...source.performanceMode },
    downsampleEnabled: { ...source.downsampleEnabled },
    useArrowTransport: { ...source.useArrowTransport },
    burnInMs: { ...source.burnInMs },
    config: Object.fromEntries(
      Object.entries(source.config).map(([key, value]) => [key, { ...value }])
    ) as ParametersState['config'],
    neuronModel: { ...source.neuronModel },
    mqif: Object.fromEntries(
      Object.entries(source.mqif).map(([key, value]) => [key, { ...value }])
    ) as Record<MqifNumericKey, SliderParameterConfig>,
    inputs: {
      ...source.inputs,
      inputType: { ...source.inputs.inputType },
      inputPopulation: { ...source.inputs.inputPopulation },
      values: Object.fromEntries(
        Object.entries(source.inputs.values).map(([key, value]) => [key, { ...value }])
      ) as ParametersState['inputs']['values'],
      targetedSubsetEnabled: { ...source.inputs.targetedSubsetEnabled },
      targetPopulation: { ...source.inputs.targetPopulation },
      targetStrategy: { ...source.inputs.targetStrategy },
      targetFraction: { ...source.inputs.targetFraction },
      targetSeed: { ...source.inputs.targetSeed },
    },
    weights: {
      ...source.weights,
      blocks: {
        ee: { ...source.weights.blocks.ee },
        ei: { ...source.weights.blocks.ei },
        ie: { ...source.weights.blocks.ie },
        ii: { ...source.weights.blocks.ii },
      },
      seed: { ...source.weights.seed },
      clampMin: { ...source.weights.clampMin },
    },
    graph: {
      nodes: source.graph.nodes.map((node) => ({ ...node })),
      edges: source.graph.edges.map((edge) => ({ ...edge, w: { ...edge.w } })),
      inputPrograms: Object.fromEntries(
        Object.entries(source.graph.inputPrograms).map(([inputId, program]) => [
          inputId,
          {
            ...program,
            values: { ...program.values },
          },
        ])
      ),
    },
  }
}

function applyGraphPreset(state: ParametersState, graph: GraphPresetSpec): ParametersState {
  const next = state

  next.config.dt.value = graph.sim.dt_ms
  next.config.T.value = graph.sim.T_ms
  next.config.seed.value = graph.sim.seed
  next.neuronModel.value = graph.sim.neuron_model
  next.weights.seed.value = graph.sim.seed

  next.performanceMode.value = graph.execution.performance_mode
  next.downsampleEnabled.value = graph.execution.max_spikes !== null
  next.burnInMs.value = graph.execution.burn_in_ms

  ;(Object.keys(graph.biophysics) as Array<keyof typeof graph.biophysics>).forEach((key) => {
    if (key in next.config) {
      next.config[key as ConfigNumericKey].value = Number(graph.biophysics[key] ?? 0)
    }
  })

  if (graph.biophysics.mqif_a?.length) next.mqif.a.value = graph.biophysics.mqif_a[0]
  if (graph.biophysics.mqif_Vr?.length) next.mqif.Vr.value = graph.biophysics.mqif_Vr[0]
  if (graph.biophysics.mqif_w_a?.length) next.mqif.w_a.value = graph.biophysics.mqif_w_a[0]
  if (graph.biophysics.mqif_w_Vr?.length) next.mqif.w_Vr.value = graph.biophysics.mqif_w_Vr[0]
  if (graph.biophysics.mqif_w_tau?.length) next.mqif.w_tau.value = graph.biophysics.mqif_w_tau[0]

  next.graph.nodes = graph.nodes.map((node) => ({
    id: node.id,
    kind: node.kind,
    type: node.type as InputType | GraphPopulationType,
    size: node.size,
  }))
  next.graph.edges = graph.edges.map((edge) => ({
    id: edge.id,
    from: edge.from,
    to: edge.to,
    kind: edge.kind,
    w: { mean: edge.w?.mean ?? 0, std: edge.w?.std ?? 0 },
    delay_ms: edge.delay_ms ?? 0.5,
  }))
  next.graph.inputPrograms = {}

  const eNode = next.graph.nodes
    .filter((node) => node.kind === 'population' && node.type === 'E')
    .reduce((acc, node) => acc + node.size, 0)
  const iNode = next.graph.nodes
    .filter((node) => node.kind === 'population' && node.type === 'I')
    .reduce((acc, node) => acc + node.size, 0)
  const inputNode = graph.nodes.find((node) => node.kind === 'input')
  next.config.N_E.value = eNode
  next.config.N_I.value = iNode
  if (inputNode) next.inputs.inputType.value = inputNode.type as InputType

  const edgeByKind = (kind: 'EE' | 'EI' | 'IE' | 'II') =>
    graph.edges.find((edge) => edge.kind === kind)
  const ee = edgeByKind('EE')
  const ei = edgeByKind('EI')
  const ie = edgeByKind('IE')
  const ii = edgeByKind('II')
  if (ee?.w) {
    next.weights.blocks.ee.mean.value = ee.w.mean
    next.weights.blocks.ee.std.value = ee.w.std
    next.config.delay_ee.value = ee.delay_ms ?? next.config.delay_ee.value
  }
  if (ei?.w) {
    next.weights.blocks.ei.mean.value = ei.w.mean
    next.weights.blocks.ei.std.value = ei.w.std
    next.config.delay_ei.value = ei.delay_ms ?? next.config.delay_ei.value
  }
  if (ie?.w) {
    next.weights.blocks.ie.mean.value = ie.w.mean
    next.weights.blocks.ie.std.value = ie.w.std
    next.config.delay_ie.value = ie.delay_ms ?? next.config.delay_ie.value
  }
  if (ii?.w) {
    next.weights.blocks.ii.mean.value = ii.w.mean
    next.weights.blocks.ii.std.value = ii.w.std
    next.config.delay_ii.value = ii.delay_ms ?? next.config.delay_ii.value
  }

  graph.nodes
    .filter((node) => node.kind === 'input')
    .forEach((node) => {
      const mode = node.type as InputType
      const inputSpec = graph.inputs[node.id] ?? {}
      const program = buildDefaultInputProgram(mode)
      program.mode = (inputSpec.mode as InputType | undefined) ?? mode
      program.input_population = (inputSpec.input_population as InputTargetPopulation | undefined) ?? program.input_population
      ;(Object.keys(program.values) as InputNumericKey[]).forEach((key) => {
        const value = inputSpec[key]
        if (typeof value === 'number') {
          program.values[key] = Number(value)
        }
      })
      if (typeof inputSpec.I_E_base === 'number' && typeof inputSpec.mean !== 'number') {
        program.values.mean = Number(inputSpec.I_E_base)
      }
      if (
        typeof inputSpec.noise_std_E === 'number' &&
        typeof inputSpec.std !== 'number'
      ) {
        program.values.std = Number(inputSpec.noise_std_E)
      }
      program.targeted_subset_enabled = Boolean(inputSpec.targeted_subset_enabled ?? false)
      if (inputSpec.target_population) {
        program.target_population = inputSpec.target_population
      }
      if (inputSpec.target_strategy) {
        program.target_strategy = inputSpec.target_strategy
      }
      if (typeof inputSpec.target_fraction === 'number') {
        program.target_fraction = inputSpec.target_fraction
      }
      if (typeof inputSpec.target_seed === 'number') {
        program.target_seed = inputSpec.target_seed
      }
      next.graph.inputPrograms[node.id] = program
    })

  if (inputNode) {
    const primaryProgram = next.graph.inputPrograms[inputNode.id]
    if (primaryProgram) {
      next.inputs.inputType.value = primaryProgram.mode
      next.inputs.inputPopulation.value = primaryProgram.input_population
      ;(Object.keys(next.inputs.values) as InputNumericKey[]).forEach((key) => {
        next.inputs.values[key].value = primaryProgram.values[key]
      })
      next.inputs.targetedSubsetEnabled.value = primaryProgram.targeted_subset_enabled
      next.inputs.targetPopulation.value = primaryProgram.target_population
      next.inputs.targetStrategy.value = primaryProgram.target_strategy
      next.inputs.targetFraction.value = primaryProgram.target_fraction
      next.inputs.targetSeed.value = primaryProgram.target_seed
    }
  }

  return next
}

export function useParameters(initial?: ParametersInitial) {
  const [parameters, setParameters] = useState<ParametersState>({
    ...DEFAULT_PARAMETERS,
    performanceMode: {
      ...DEFAULT_PARAMETERS.performanceMode,
      value: initial?.performanceMode ?? DEFAULT_PARAMETERS.performanceMode.defaultValue,
    },
    downsampleEnabled: {
      ...DEFAULT_PARAMETERS.downsampleEnabled,
      value: initial?.downsampleEnabled ?? DEFAULT_PARAMETERS.downsampleEnabled.defaultValue,
    },
    useArrowTransport: {
      ...DEFAULT_PARAMETERS.useArrowTransport,
      value: initial?.useArrowTransport ?? DEFAULT_PARAMETERS.useArrowTransport.defaultValue,
    },
    burnInMs: {
      ...DEFAULT_PARAMETERS.burnInMs,
      value: initial?.burnInMs ?? DEFAULT_PARAMETERS.burnInMs.defaultValue,
    },
    config: Object.fromEntries(
      Object.entries(DEFAULT_PARAMETERS.config).map(([key, value]) => [
        key,
        { ...value, value: initial?.config?.[key as ConfigNumericKey] ?? value.defaultValue },
      ])
    ) as Record<ConfigNumericKey, SliderParameterConfig>,
    neuronModel: { ...DEFAULT_PARAMETERS.neuronModel },
    mqif: Object.fromEntries(
      Object.entries(DEFAULT_PARAMETERS.mqif).map(([key, value]) => [key, { ...value }])
    ) as Record<MqifNumericKey, SliderParameterConfig>,
    inputs: {
      ...DEFAULT_PARAMETERS.inputs,
      inputType: {
        ...DEFAULT_PARAMETERS.inputs.inputType,
        value: initial?.inputType ?? DEFAULT_PARAMETERS.inputs.inputType.defaultValue,
      },
      values: Object.fromEntries(
        Object.entries(DEFAULT_PARAMETERS.inputs.values).map(([key, value]) => [key, { ...value }])
      ) as Record<InputNumericKey, SliderParameterConfig>,
    },
    weights: {
      blocks: {
        ee: { ...DEFAULT_PARAMETERS.weights.blocks.ee },
        ei: { ...DEFAULT_PARAMETERS.weights.blocks.ei },
        ie: { ...DEFAULT_PARAMETERS.weights.blocks.ie },
        ii: { ...DEFAULT_PARAMETERS.weights.blocks.ii },
      },
      clampMin: { ...DEFAULT_PARAMETERS.weights.clampMin },
      seed: { ...DEFAULT_PARAMETERS.weights.seed },
    },
    graph: {
      nodes: DEFAULT_PARAMETERS.graph.nodes.map((node) => ({ ...node })),
      edges: DEFAULT_PARAMETERS.graph.edges.map((edge) => ({ ...edge, w: { ...edge.w } })),
      inputPrograms: Object.fromEntries(
        Object.entries(DEFAULT_PARAMETERS.graph.inputPrograms).map(([inputId, program]) => [
          inputId,
          {
            ...program,
            values: { ...program.values },
          },
        ])
      ),
    },
  })

  const applyConfigPreset = (preset: ConfigPreset) =>
    setParameters((prev) => {
      const next = cloneParametersState(DEFAULT_PARAMETERS)
      next.useArrowTransport.value = prev.useArrowTransport.value
      applyGraphPreset(next, GRAPH_PRESETS[preset])
      return sanitizePresetParameters(next)
    })

  return {
    parameters,
    setPerformanceMode: (performanceMode: boolean) =>
      setParameters((prev) => ({ ...prev, performanceMode: { ...prev.performanceMode, value: performanceMode } })),
    setDownsampleEnabled: (downsampleEnabled: boolean) =>
      setParameters((prev) => ({ ...prev, downsampleEnabled: { ...prev.downsampleEnabled, value: downsampleEnabled } })),
    setUseArrowTransport: (useArrowTransport: boolean) =>
      setParameters((prev) => ({ ...prev, useArrowTransport: { ...prev.useArrowTransport, value: useArrowTransport } })),
    setBurnInMs: (burnInMs: number) =>
      setParameters((prev) => ({ ...prev, burnInMs: { ...prev.burnInMs, value: burnInMs } })),
    setConfigValue: (key: ConfigNumericKey, nextValue: number) =>
      setParameters((prev) => {
        const meta = CONFIG_PARAMETER_META[key]
        const roundedValue = (meta as { integer?: boolean }).integer ? Math.round(nextValue) : nextValue
        return {
          ...prev,
          config: {
            ...prev.config,
            [key]: { ...prev.config[key], value: roundedValue },
          },
        }
      }),
    setNeuronModel: (neuronModel: NeuronModel) =>
      setParameters((prev) => ({ ...prev, neuronModel: { ...prev.neuronModel, value: neuronModel } })),
    setMqifValue: (key: MqifNumericKey, nextValue: number) =>
      setParameters((prev) => ({ ...prev, mqif: { ...prev.mqif, [key]: { ...prev.mqif[key], value: nextValue } } })),
    setInputType: (inputType: InputType) =>
      setParameters((prev) => ({
        ...prev,
        inputs: {
          ...prev.inputs,
          inputType: { ...prev.inputs.inputType, value: inputType },
          inputPopulation: { ...prev.inputs.inputPopulation, value: inputType === 'external_spike_train' ? 'e' : prev.inputs.inputPopulation.value },
        },
        graph: {
          ...prev.graph,
          nodes: prev.graph.nodes.map((node) =>
            node.kind === 'input' ? { ...node, type: inputType } : node
          ),
          inputPrograms: Object.fromEntries(
            Object.entries(prev.graph.inputPrograms).map(([inputId, program]) => [
              inputId,
              {
                ...program,
                mode: inputType,
                input_population:
                  inputType === 'external_spike_train' ? 'e' : program.input_population,
                values: { ...program.values },
              },
            ])
          ),
        },
      })),
    setInputPopulation: (inputPopulation: InputTargetPopulation) =>
      setParameters((prev) => {
        const primaryInputId =
          prev.graph.nodes.find((node) => node.kind === 'input')?.id
        if (!primaryInputId) return prev
        const current = prev.graph.inputPrograms[primaryInputId] ?? buildDefaultInputProgram()
        return {
          ...prev,
          inputs: { ...prev.inputs, inputPopulation: { ...prev.inputs.inputPopulation, value: inputPopulation } },
          graph: {
            ...prev.graph,
            inputPrograms: {
              ...prev.graph.inputPrograms,
              [primaryInputId]: { ...current, input_population: inputPopulation, values: { ...current.values } },
            },
          },
        }
      }),
    setInputValue: (key: InputNumericKey, nextValue: number) =>
      setParameters((prev) => {
        const primaryInputId =
          prev.graph.nodes.find((node) => node.kind === 'input')?.id
        if (!primaryInputId) return prev
        const current = prev.graph.inputPrograms[primaryInputId] ?? buildDefaultInputProgram()
        const value = key === 'seed' ? Math.round(nextValue) : nextValue
        return {
          ...prev,
          inputs: {
            ...prev.inputs,
            values: {
              ...prev.inputs.values,
              [key]: { ...prev.inputs.values[key], value },
            },
          },
          graph: {
            ...prev.graph,
            inputPrograms: {
              ...prev.graph.inputPrograms,
              [primaryInputId]: {
                ...current,
                values: { ...current.values, [key]: value },
              },
            },
          },
        }
      }),
    setInputTargetedSubsetEnabled: (enabled: boolean) =>
      setParameters((prev) => {
        const primaryInputId =
          prev.graph.nodes.find((node) => node.kind === 'input')?.id
        if (!primaryInputId) return prev
        const current = prev.graph.inputPrograms[primaryInputId] ?? buildDefaultInputProgram()
        return {
          ...prev,
          inputs: { ...prev.inputs, targetedSubsetEnabled: { ...prev.inputs.targetedSubsetEnabled, value: enabled } },
          graph: {
            ...prev.graph,
            inputPrograms: {
              ...prev.graph.inputPrograms,
              [primaryInputId]: {
                ...current,
                values: { ...current.values },
                targeted_subset_enabled: enabled,
              },
            },
          },
        }
      }),
    setInputTargetPopulation: (targetPopulation: InputTargetPopulation) =>
      setParameters((prev) => {
        const primaryInputId =
          prev.graph.nodes.find((node) => node.kind === 'input')?.id
        if (!primaryInputId) return prev
        const current = prev.graph.inputPrograms[primaryInputId] ?? buildDefaultInputProgram()
        return {
          ...prev,
          inputs: { ...prev.inputs, targetPopulation: { ...prev.inputs.targetPopulation, value: targetPopulation } },
          graph: {
            ...prev.graph,
            inputPrograms: {
              ...prev.graph.inputPrograms,
              [primaryInputId]: {
                ...current,
                values: { ...current.values },
                target_population: targetPopulation,
              },
            },
          },
        }
      }),
    setInputTargetStrategy: (targetStrategy: InputTargetStrategy) =>
      setParameters((prev) => {
        const primaryInputId =
          prev.graph.nodes.find((node) => node.kind === 'input')?.id
        if (!primaryInputId) return prev
        const current = prev.graph.inputPrograms[primaryInputId] ?? buildDefaultInputProgram()
        return {
          ...prev,
          inputs: { ...prev.inputs, targetStrategy: { ...prev.inputs.targetStrategy, value: targetStrategy } },
          graph: {
            ...prev.graph,
            inputPrograms: {
              ...prev.graph.inputPrograms,
              [primaryInputId]: {
                ...current,
                values: { ...current.values },
                target_strategy: targetStrategy,
              },
            },
          },
        }
      }),
    setInputTargetFraction: (targetFraction: number) =>
      setParameters((prev) => {
        const primaryInputId =
          prev.graph.nodes.find((node) => node.kind === 'input')?.id
        if (!primaryInputId) return prev
        const current = prev.graph.inputPrograms[primaryInputId] ?? buildDefaultInputProgram()
        return {
          ...prev,
          inputs: { ...prev.inputs, targetFraction: { ...prev.inputs.targetFraction, value: targetFraction } },
          graph: {
            ...prev.graph,
            inputPrograms: {
              ...prev.graph.inputPrograms,
              [primaryInputId]: {
                ...current,
                values: { ...current.values },
                target_fraction: targetFraction,
              },
            },
          },
        }
      }),
    setInputTargetSeed: (targetSeed: number) =>
      setParameters((prev) => {
        const primaryInputId =
          prev.graph.nodes.find((node) => node.kind === 'input')?.id
        if (!primaryInputId) return prev
        const current = prev.graph.inputPrograms[primaryInputId] ?? buildDefaultInputProgram()
        return {
          ...prev,
          inputs: { ...prev.inputs, targetSeed: { ...prev.inputs.targetSeed, value: Math.round(targetSeed) } },
          graph: {
            ...prev.graph,
            inputPrograms: {
              ...prev.graph.inputPrograms,
              [primaryInputId]: {
                ...current,
                values: { ...current.values },
                target_seed: Math.round(targetSeed),
              },
            },
          },
        }
      }),
    setInputProgramType: (inputId: string, mode: InputType) =>
      setParameters((prev) => {
        const current = prev.graph.inputPrograms[inputId] ?? buildDefaultInputProgram(mode)
        return sanitizePresetParameters({
          ...prev,
          graph: {
            ...prev.graph,
            nodes: prev.graph.nodes.map((node) =>
              node.kind === 'input' && node.id === inputId ? { ...node, type: mode } : node
            ),
            inputPrograms: {
              ...prev.graph.inputPrograms,
              [inputId]: {
                ...current,
                values: { ...current.values },
                mode,
                input_population:
                  mode === 'external_spike_train' ? 'e' : current.input_population,
              },
            },
          },
        })
      }),
    setInputProgramPopulation: (inputId: string, inputPopulation: InputTargetPopulation) =>
      setParameters((prev) => {
        const current = prev.graph.inputPrograms[inputId] ?? buildDefaultInputProgram()
        return sanitizePresetParameters({
          ...prev,
          graph: {
            ...prev.graph,
            inputPrograms: {
              ...prev.graph.inputPrograms,
              [inputId]: {
                ...current,
                values: { ...current.values },
                input_population: inputPopulation,
              },
            },
          },
        })
      }),
    setInputProgramValue: (inputId: string, key: InputNumericKey, nextValue: number) =>
      setParameters((prev) => {
        const current = prev.graph.inputPrograms[inputId] ?? buildDefaultInputProgram()
        return sanitizePresetParameters({
          ...prev,
          graph: {
            ...prev.graph,
            inputPrograms: {
              ...prev.graph.inputPrograms,
              [inputId]: {
                ...current,
                values: {
                  ...current.values,
                  [key]: key === 'seed' ? Math.round(nextValue) : nextValue,
                },
              },
            },
          },
        })
      }),
    setInputProgramTargetedSubsetEnabled: (inputId: string, enabled: boolean) =>
      setParameters((prev) => {
        const current = prev.graph.inputPrograms[inputId] ?? buildDefaultInputProgram()
        return sanitizePresetParameters({
          ...prev,
          graph: {
            ...prev.graph,
            inputPrograms: {
              ...prev.graph.inputPrograms,
              [inputId]: {
                ...current,
                values: { ...current.values },
                targeted_subset_enabled: enabled,
              },
            },
          },
        })
      }),
    setInputProgramTargetPopulation: (inputId: string, targetPopulation: InputTargetPopulation) =>
      setParameters((prev) => {
        const current = prev.graph.inputPrograms[inputId] ?? buildDefaultInputProgram()
        return sanitizePresetParameters({
          ...prev,
          graph: {
            ...prev.graph,
            inputPrograms: {
              ...prev.graph.inputPrograms,
              [inputId]: {
                ...current,
                values: { ...current.values },
                target_population: targetPopulation,
              },
            },
          },
        })
      }),
    setInputProgramTargetStrategy: (inputId: string, targetStrategy: InputTargetStrategy) =>
      setParameters((prev) => {
        const current = prev.graph.inputPrograms[inputId] ?? buildDefaultInputProgram()
        return sanitizePresetParameters({
          ...prev,
          graph: {
            ...prev.graph,
            inputPrograms: {
              ...prev.graph.inputPrograms,
              [inputId]: {
                ...current,
                values: { ...current.values },
                target_strategy: targetStrategy,
              },
            },
          },
        })
      }),
    setInputProgramTargetFraction: (inputId: string, targetFraction: number) =>
      setParameters((prev) => {
        const current = prev.graph.inputPrograms[inputId] ?? buildDefaultInputProgram()
        return sanitizePresetParameters({
          ...prev,
          graph: {
            ...prev.graph,
            inputPrograms: {
              ...prev.graph.inputPrograms,
              [inputId]: {
                ...current,
                values: { ...current.values },
                target_fraction: targetFraction,
              },
            },
          },
        })
      }),
    setInputProgramTargetSeed: (inputId: string, targetSeed: number) =>
      setParameters((prev) => {
        const current = prev.graph.inputPrograms[inputId] ?? buildDefaultInputProgram()
        return sanitizePresetParameters({
          ...prev,
          graph: {
            ...prev.graph,
            inputPrograms: {
              ...prev.graph.inputPrograms,
              [inputId]: {
                ...current,
                values: { ...current.values },
                target_seed: Math.round(targetSeed),
              },
            },
          },
        })
      }),
    setWeightValue: (block: WeightBlockKey, key: WeightNumericKey, nextValue: number) =>
      setParameters((prev) => ({
        ...prev,
        weights: {
          ...prev.weights,
          blocks: {
            ...prev.weights.blocks,
            [block]: {
              ...prev.weights.blocks[block],
              [key]: { ...prev.weights.blocks[block][key], value: nextValue },
            },
          },
        },
      })),
    setWeightsSeed: (nextValue: number) =>
      setParameters((prev) => ({ ...prev, weights: { ...prev.weights, seed: { ...prev.weights.seed, value: Math.round(nextValue) } } })),
    addGraphNode: (kind: GraphNodeKind) =>
      setParameters((prev) => {
        const next = cloneParametersState(prev)
        if (kind === 'input') {
          const count = next.graph.nodes.filter((node) => node.kind === 'input').length + 1
          const inputId = `input_${count}`
          next.graph.nodes.push({
            id: inputId,
            kind: 'input',
            type: 'tonic',
            size: 0,
          })
          next.graph.inputPrograms[inputId] = buildDefaultInputProgram('tonic')
        } else {
          const popType: GraphPopulationType = 'E'
          const count =
            next.graph.nodes.filter(
              (node) => node.kind === 'population' && node.type === popType
            ).length + 1
          next.graph.nodes.push({
            id: `${popType}${count}`,
            kind: 'population',
            type: popType,
            size: 50,
          })
        }
        next.config.N_E.value = next.graph.nodes
          .filter((node) => node.kind === 'population' && node.type === 'E')
          .reduce((acc, node) => acc + node.size, 0)
        next.config.N_I.value = next.graph.nodes
          .filter((node) => node.kind === 'population' && node.type === 'I')
          .reduce((acc, node) => acc + node.size, 0)
        return sanitizePresetParameters(next)
      }),
    updateGraphNode: (
      nodeId: string,
      patch: Partial<Pick<GraphNodeState, 'id' | 'type' | 'size'>>
    ) =>
      setParameters((prev) => {
        const next = cloneParametersState(prev)
        const candidateNodes = next.graph.nodes.map((node) =>
          node.id === nodeId ? { ...node, ...patch } : node
        )
        const eCount = candidateNodes.filter(
          (node) => node.kind === 'population' && node.type === 'E'
        ).length
        const inputCount = candidateNodes.filter((node) => node.kind === 'input').length
        if (eCount <= 0 || inputCount <= 0) {
          return prev
        }
        next.graph.nodes = candidateNodes
        next.graph.edges = next.graph.edges.map((edge) => {
          if (edge.from === nodeId && patch.id) {
            return { ...edge, from: patch.id }
          }
          if (edge.to === nodeId && patch.id) {
            return { ...edge, to: patch.id }
          }
          return edge
        })
        if (patch.id && patch.id !== nodeId && next.graph.inputPrograms[nodeId]) {
          next.graph.inputPrograms[patch.id] = {
            ...next.graph.inputPrograms[nodeId],
            values: { ...next.graph.inputPrograms[nodeId].values },
          }
          delete next.graph.inputPrograms[nodeId]
        }
        const updatedNode = next.graph.nodes.find((node) => node.id === (patch.id ?? nodeId))
        if (updatedNode?.kind === 'input') {
          const inputId = updatedNode.id
          const current = next.graph.inputPrograms[inputId] ?? buildDefaultInputProgram(updatedNode.type as InputType)
          next.graph.inputPrograms[inputId] = {
            ...current,
            values: { ...current.values },
            mode: updatedNode.type as InputType,
          }
        }
        next.config.N_E.value = next.graph.nodes
          .filter((node) => node.kind === 'population' && node.type === 'E')
          .reduce((acc, node) => acc + node.size, 0)
        next.config.N_I.value = next.graph.nodes
          .filter((node) => node.kind === 'population' && node.type === 'I')
          .reduce((acc, node) => acc + node.size, 0)
        return sanitizePresetParameters(next)
      }),
    removeGraphNode: (nodeId: string) =>
      setParameters((prev) => {
        const next = cloneParametersState(prev)
        const node = next.graph.nodes.find((entry) => entry.id === nodeId)
        if (!node) return prev
        if (node.kind === 'input') {
          const inputCount = next.graph.nodes.filter((entry) => entry.kind === 'input').length
          if (inputCount <= 1) return prev
        }
        if (node.kind === 'population' && node.type === 'E') {
          const eCount = next.graph.nodes.filter(
            (entry) => entry.kind === 'population' && entry.type === 'E'
          ).length
          if (eCount <= 1) return prev
        }
        next.graph.nodes = next.graph.nodes.filter((node) => node.id !== nodeId)
        next.graph.edges = next.graph.edges.filter(
          (edge) => edge.from !== nodeId && edge.to !== nodeId
        )
        delete next.graph.inputPrograms[nodeId]
        next.config.N_E.value = next.graph.nodes
          .filter((node) => node.kind === 'population' && node.type === 'E')
          .reduce((acc, node) => acc + node.size, 0)
        next.config.N_I.value = next.graph.nodes
          .filter((node) => node.kind === 'population' && node.type === 'I')
          .reduce((acc, node) => acc + node.size, 0)
        return sanitizePresetParameters(next)
      }),
    addGraphEdge: () =>
      setParameters((prev) => {
        const next = cloneParametersState(prev)
        const first = next.graph.nodes[0]?.id ?? 'input_1'
        const second = next.graph.nodes[1]?.id ?? first
        next.graph.edges.push({
          id: `edge_${next.graph.edges.length + 1}`,
          from: first,
          to: second,
          kind: 'input',
          w: { mean: 0, std: 0 },
          delay_ms: 0.5,
        })
        return sanitizePresetParameters(next)
      }),
    updateGraphEdge: (
      edgeId: string,
      patch: Partial<Pick<GraphEdgeState, 'id' | 'from' | 'to' | 'kind' | 'delay_ms'>> & {
        w?: Partial<GraphEdgeState['w']>
      }
    ) =>
      setParameters((prev) => {
        const next = cloneParametersState(prev)
        next.graph.edges = next.graph.edges.map((edge) => {
          if (edge.id !== edgeId) return edge
          return {
            ...edge,
            ...patch,
            w: patch.w ? { ...edge.w, ...patch.w } : edge.w,
          }
        })
        return sanitizePresetParameters(next)
      }),
    removeGraphEdge: (edgeId: string) =>
      setParameters((prev) => {
        const next = cloneParametersState(prev)
        next.graph.edges = next.graph.edges.filter((edge) => edge.id !== edgeId)
        return sanitizePresetParameters(next)
      }),
    applyConfigPreset: (preset: ConfigPreset) => applyConfigPreset(preset),
  }
}
