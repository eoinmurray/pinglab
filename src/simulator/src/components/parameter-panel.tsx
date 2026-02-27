import ParameterSelect from '@/components/parameter-select'
import ParameterSlider from '@/components/parameter-slider'
import ParameterSwitch from '@/components/parameter-switch'
import { Button } from '@/components/ui/button'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import { useSimulationContext } from '@/context/simulation-context'
import {
  ChevronDown,
  Component,
  Gauge,
  GitBranch,
  type LucideIcon,
  Sparkles,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import {
  CONFIG_PARAMETER_DEFINITIONS,
  CONFIG_PRESET_OPTIONS,
  type ConfigPreset,
  type GraphEdgeKind,
  type GraphNodeState,
  type InputType,
  type MqifNumericKey,
} from '@/hooks/use-parameters'

type PanelSection =
  | 'Sim'
  | 'Nodes'
  | 'Edges'
  | 'Biophysics'
  | 'Execution'

const SECTION_ICONS: Record<PanelSection, LucideIcon> = {
  Sim: Gauge,
  Nodes: Component,
  Edges: GitBranch,
  Biophysics: Sparkles,
  Execution: Gauge,
}

const EDGE_KIND_OPTIONS: { value: GraphEdgeKind; label: string }[] = [
  { value: 'input', label: 'input' },
  { value: 'EE', label: 'EE' },
  { value: 'EI', label: 'EI' },
  { value: 'IE', label: 'IE' },
  { value: 'II', label: 'II' },
]

const BIOPHYSICS_KEYS = [
  'tau_ampa',
  'tau_gaba',
  't_ref_E',
  't_ref_I',
  'V_init',
  'E_L',
  'E_e',
  'E_i',
  'C_m_E',
  'g_L_E',
  'C_m_I',
  'g_L_I',
  'V_th',
  'V_reset',
  'g_L_heterogeneity_sd',
  'C_m_heterogeneity_sd',
  'V_th_heterogeneity_sd',
  't_ref_heterogeneity_sd',
] as const

const visibleInputKeysForMode = (mode: InputType) =>
  mode === 'ramp'
    ? ([
        'mean',
        'std',
        'seed',
      ] as const)
    : mode === 'sine'
      ? ([
          'mean',
          'std',
          'seed',
          'sine_freq_hz',
          'sine_amp',
          'sine_y_offset',
          'sine_phase',
          'sine_phase_offset_i',
        ] as const)
      : mode === 'external_spike_train'
        ? ([
            'seed',
            'lambda0_hz',
            'mod_depth',
            'envelope_freq_hz',
            'phase_rad',
            'w_in',
            'tau_in_ms',
          ] as const)
        : mode === 'tonic'
          ? ([
              'mean',
              'std',
              'seed',
            ] as const)
          : mode === 'pulse'
            ? ([
                'mean',
                'std',
                'seed',
                'pulse_t_ms',
                'pulse_width_ms',
                'pulse_amp_E',
                'pulse_amp_I',
              ] as const)
            : ([
                'mean',
                'std',
                'seed',
                'pulse_t_ms',
                'pulse_width_ms',
                'pulse_interval_ms',
                'pulse_amp_E',
                'pulse_amp_I',
              ] as const)

export default function ParameterPanel() {
  const {
    parameters,
    setConfigValue,
    setPerformanceMode,
    setDownsampleEnabled,
    setUseArrowTransport,
    setBurnInMs,
    setNeuronModel,
    setMqifValue,
    setInputProgramType,
    setInputProgramValue,
    setInputProgramTargetedSubsetEnabled,
    setInputProgramTargetPopulation,
    setInputProgramTargetStrategy,
    setInputProgramTargetFraction,
    setInputProgramTargetSeed,
    addGraphNode,
    updateGraphNode,
    removeGraphNode,
    addGraphEdge,
    updateGraphEdge,
    removeGraphEdge,
    applyConfigPreset,
    collapsibleOpen,
    setCollapsibleOpen,
  } = useSimulationContext()
  const [selectedPreset, setSelectedPreset] = useState<ConfigPreset>('ping')
  const neuronModel = parameters.neuronModel.value

  useEffect(() => {
    applyConfigPreset('ping')
    // Intentionally one-shot on mount so initial UI state matches selected preset.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const mqifKeys: MqifNumericKey[] = ['a', 'Vr', 'w_a', 'w_Vr', 'w_tau']
  const graphNodeOptions = useMemo(
    () => parameters.graph.nodes.map((node) => ({ value: node.id, label: node.id })),
    [parameters.graph.nodes]
  )
  const renderSectionHeader = (title: PanelSection) => {
    const Icon = SECTION_ICONS[title]
    return (
      <CollapsibleTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          className="group h-6 w-full justify-between rounded-sm px-1.5 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground"
        >
          <span className="inline-flex items-center gap-1.5">
            <Icon className="size-3.5" />
            <span>{title}</span>
          </span>
          <ChevronDown className="size-3 transition-transform group-data-[state=open]:rotate-180" />
        </Button>
      </CollapsibleTrigger>
    )
  }

  return (
    <div className="pb-6">
      <div className="px-1.5 pb-1.5">
        <ParameterSelect
          label="Graph Preset"
          info="Apply a predefined graph spec preset."
          options={CONFIG_PRESET_OPTIONS}
          value={selectedPreset}
          onValueChange={(preset) => {
            setSelectedPreset(preset)
            applyConfigPreset(preset)
          }}
        />
      </div>

      <Collapsible
        open={collapsibleOpen.Sim ?? false}
        onOpenChange={(open) => setCollapsibleOpen('Sim', open)}
        className="space-y-1 data-[state=open]:mb-3"
      >
        {renderSectionHeader('Sim')}
        <CollapsibleContent className="space-y-2 px-1.5 pb-1.5 pt-0.5">
          <ParameterSlider
            label="sim.dt_ms"
            info={parameters.config.dt.info}
            min={parameters.config.dt.min}
            max={parameters.config.dt.max}
            step={parameters.config.dt.step}
            defaultValue={parameters.config.dt.defaultValue}
            value={parameters.config.dt.value}
            onValueChange={(value) => setConfigValue('dt', value)}
          />
          <ParameterSlider
            label="sim.T_ms"
            info={parameters.config.T.info}
            min={parameters.config.T.min}
            max={parameters.config.T.max}
            step={parameters.config.T.step}
            defaultValue={parameters.config.T.defaultValue}
            value={parameters.config.T.value}
            onValueChange={(value) => setConfigValue('T', value)}
          />
          <ParameterSlider
            label="sim.seed"
            info={parameters.config.seed.info}
            min={parameters.config.seed.min}
            max={parameters.config.seed.max}
            step={parameters.config.seed.step}
            defaultValue={parameters.config.seed.defaultValue}
            value={parameters.config.seed.value}
            onValueChange={(value) => setConfigValue('seed', value)}
          />
          <ParameterSelect
            label="sim.neuron_model"
            info={parameters.neuronModel.info}
            options={parameters.neuronModel.options}
            defaultValue={parameters.neuronModel.defaultValue}
            value={parameters.neuronModel.value}
            onValueChange={setNeuronModel}
          />
        </CollapsibleContent>
      </Collapsible>

      <Collapsible
        open={collapsibleOpen.Nodes ?? false}
        onOpenChange={(open) => setCollapsibleOpen('Nodes', open)}
        className="space-y-1 data-[state=open]:mb-3"
      >
        {renderSectionHeader('Nodes')}
        <CollapsibleContent className="space-y-2 px-1.5 pb-1.5 pt-0.5">
          <div className="flex flex-wrap gap-2">
            <Button size="xs" type="button" variant="outline" onClick={() => addGraphNode('population')}>
              Add Population Node
            </Button>
            <Button size="xs" type="button" variant="outline" onClick={() => addGraphNode('input')}>
              Add Input Node
            </Button>
          </div>

          {parameters.graph.nodes.map((node) => (
            <div
              key={node.id}
              className="space-y-1.5 rounded-sm border border-border/70 bg-muted/30 px-1.5 pt-1.5 pb-2.5"
            >
              {(() => {
                const canRemoveInput =
                  node.kind !== 'input' ||
                  parameters.graph.nodes.filter((entry) => entry.kind === 'input').length > 1
                const canRemoveEPopulation =
                  !(node.kind === 'population' && node.type === 'E') ||
                  parameters.graph.nodes.filter(
                    (entry) => entry.kind === 'population' && entry.type === 'E'
                  ).length > 1
                const canRemove = canRemoveInput && canRemoveEPopulation
                return (
                  <>
              <div className="flex items-center justify-between gap-2">
                <div className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                  Node: {node.id}
                </div>
                <Button
                  size="xs"
                  type="button"
                  variant="ghost"
                  disabled={!canRemove}
                  onClick={() => removeGraphNode(node.id)}
                >
                  Remove
                </Button>
              </div>

              <label className="text-[10px] text-muted-foreground">nodes.{node.id}.id</label>
              <input
                className="h-6 w-full rounded-sm border border-border/70 bg-background px-2 text-[11px]"
                value={node.id}
                onChange={(event) =>
                  updateGraphNode(node.id, { id: event.target.value.trim() || node.id })
                }
              />

              {node.kind === 'input' ? (
                <>
                  <ParameterSelect
                    label={`nodes.${node.id}.type`}
                    info={parameters.inputs.inputType.info}
                    options={parameters.inputs.inputType.options}
                    value={String(node.type)}
                    onValueChange={(value) =>
                      setInputProgramType(node.id, value as InputType)
                    }
                  />
                  {(() => {
                    const inputProgram = parameters.graph.inputPrograms[node.id]
                    const inputMode =
                      inputProgram?.mode ?? (node.type as InputType)
                    const visibleInputKeys = visibleInputKeysForMode(inputMode)
                    const showTargetSubsetControls =
                      inputMode === 'pulse' || inputMode === 'pulses'
                    return (
                      <>
                        {visibleInputKeys.map((key) => {
                          const parameter = parameters.inputs.values[key]
                          return (
                            <ParameterSlider
                              key={`${node.id}:${key}`}
                              label={`inputs.${node.id}.${key}`}
                              info={parameter.info}
                              min={parameter.min}
                              max={parameter.max}
                              step={parameter.step}
                              defaultValue={parameter.defaultValue}
                              value={inputProgram?.values[key] ?? parameter.defaultValue}
                              onValueChange={(nextValue) =>
                                setInputProgramValue(node.id, key, nextValue)
                              }
                            />
                          )
                        })}
                        {showTargetSubsetControls ? (
                          <>
                            <ParameterSwitch
                              label={`inputs.${node.id}.targeted_subset_enabled`}
                              info={parameters.inputs.targetedSubsetEnabled.info}
                              defaultChecked={
                                parameters.inputs.targetedSubsetEnabled.defaultValue
                              }
                              checked={inputProgram?.targeted_subset_enabled ?? false}
                              onCheckedChange={(checked) =>
                                setInputProgramTargetedSubsetEnabled(node.id, checked)
                              }
                            />
                            {inputProgram?.targeted_subset_enabled ? (
                              <>
                                <ParameterSelect
                                  label={`inputs.${node.id}.target_population`}
                                  info={parameters.inputs.targetPopulation.info}
                                  options={parameters.inputs.targetPopulation.options}
                                  defaultValue={
                                    parameters.inputs.targetPopulation.defaultValue
                                  }
                                  value={inputProgram.target_population}
                                  onValueChange={(value) =>
                                    setInputProgramTargetPopulation(
                                      node.id,
                                      value as 'all' | 'e' | 'i'
                                    )
                                  }
                                />
                                <ParameterSelect
                                  label={`inputs.${node.id}.target_strategy`}
                                  info={parameters.inputs.targetStrategy.info}
                                  options={parameters.inputs.targetStrategy.options}
                                  defaultValue={
                                    parameters.inputs.targetStrategy.defaultValue
                                  }
                                  value={inputProgram.target_strategy}
                                  onValueChange={(value) =>
                                    setInputProgramTargetStrategy(
                                      node.id,
                                      value as 'random' | 'first'
                                    )
                                  }
                                />
                                <ParameterSlider
                                  label={`inputs.${node.id}.target_fraction`}
                                  info={parameters.inputs.targetFraction.info}
                                  min={parameters.inputs.targetFraction.min}
                                  max={parameters.inputs.targetFraction.max}
                                  step={parameters.inputs.targetFraction.step}
                                  defaultValue={
                                    parameters.inputs.targetFraction.defaultValue
                                  }
                                  value={inputProgram.target_fraction}
                                  onValueChange={(value) =>
                                    setInputProgramTargetFraction(node.id, value)
                                  }
                                />
                                <ParameterSlider
                                  label={`inputs.${node.id}.target_seed`}
                                  info={parameters.inputs.targetSeed.info}
                                  min={parameters.inputs.targetSeed.min}
                                  max={parameters.inputs.targetSeed.max}
                                  step={parameters.inputs.targetSeed.step}
                                  defaultValue={
                                    parameters.inputs.targetSeed.defaultValue
                                  }
                                  value={inputProgram.target_seed}
                                  onValueChange={(value) =>
                                    setInputProgramTargetSeed(node.id, value)
                                  }
                                />
                              </>
                            ) : null}
                          </>
                        ) : null}
                      </>
                    )
                  })()}
                </>
              ) : (
                <>
                  <ParameterSelect
                    label={`nodes.${node.id}.type`}
                    info="Population type."
                    options={[
                      { value: 'E', label: 'E' },
                      { value: 'I', label: 'I' },
                    ]}
                    value={String(node.type)}
                    onValueChange={(value) =>
                      updateGraphNode(node.id, { type: value as GraphNodeState['type'] })
                    }
                  />
                  <ParameterSlider
                    label={`nodes.${node.id}.size`}
                    info="Population size."
                    min={0}
                    max={800}
                    step={1}
                    defaultValue={node.size}
                    value={node.size}
                    onValueChange={(value) => updateGraphNode(node.id, { size: value })}
                  />
                </>
              )}
                  </>
                )
              })()}
            </div>
          ))}
        </CollapsibleContent>
      </Collapsible>

      <Collapsible
        open={collapsibleOpen.Edges ?? false}
        onOpenChange={(open) => setCollapsibleOpen('Edges', open)}
        className="space-y-1 data-[state=open]:mb-3"
      >
        {renderSectionHeader('Edges')}
        <CollapsibleContent className="space-y-2 px-1.5 pb-1.5 pt-0.5">
          <Button size="xs" type="button" variant="outline" onClick={addGraphEdge}>
            Add Edge
          </Button>

          {parameters.graph.edges.map((edge) => (
            <div
              key={edge.id}
              className="space-y-1.5 rounded-sm border border-border/70 bg-muted/30 px-1.5 pt-1.5 pb-2.5"
            >
              <div className="flex items-center justify-between gap-2">
                <div className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                  Edge: {edge.id}
                </div>
                <Button
                  size="xs"
                  type="button"
                  variant="ghost"
                  onClick={() => removeGraphEdge(edge.id)}
                >
                  Remove
                </Button>
              </div>

              <label className="text-[10px] text-muted-foreground">edges.{edge.id}.id</label>
              <input
                className="h-6 w-full rounded-sm border border-border/70 bg-background px-2 text-[11px]"
                value={edge.id}
                onChange={(event) =>
                  updateGraphEdge(edge.id, { id: event.target.value.trim() || edge.id })
                }
              />

              <ParameterSelect
                label={`edges.${edge.id}.from`}
                info="Source node."
                options={graphNodeOptions}
                value={edge.from}
                onValueChange={(value) => updateGraphEdge(edge.id, { from: value })}
              />
              <ParameterSelect
                label={`edges.${edge.id}.to`}
                info="Target node."
                options={graphNodeOptions}
                value={edge.to}
                onValueChange={(value) => updateGraphEdge(edge.id, { to: value })}
              />
              <ParameterSelect
                label={`edges.${edge.id}.kind`}
                info="Edge kind."
                options={EDGE_KIND_OPTIONS}
                value={edge.kind}
                onValueChange={(value) =>
                  updateGraphEdge(edge.id, { kind: value as GraphEdgeKind })
                }
              />
              <ParameterSlider
                label={`edges.${edge.id}.w.mean`}
                info="Weight mean."
                min={0}
                max={1}
                step={0.0002}
                defaultValue={edge.w.mean}
                value={edge.w.mean}
                onValueChange={(value) => updateGraphEdge(edge.id, { w: { mean: value } })}
              />
              <ParameterSlider
                label={`edges.${edge.id}.w.std`}
                info="Weight std."
                min={0}
                max={0.5}
                step={0.0002}
                defaultValue={edge.w.std}
                value={edge.w.std}
                onValueChange={(value) => updateGraphEdge(edge.id, { w: { std: value } })}
              />
              <ParameterSlider
                label={`edges.${edge.id}.delay_ms`}
                info="Edge delay in ms."
                min={0.01}
                max={5}
                step={0.01}
                defaultValue={edge.delay_ms}
                value={edge.delay_ms}
                onValueChange={(value) => updateGraphEdge(edge.id, { delay_ms: value })}
              />
            </div>
          ))}
        </CollapsibleContent>
      </Collapsible>

      <Collapsible
        open={collapsibleOpen.Biophysics ?? false}
        onOpenChange={(open) => setCollapsibleOpen('Biophysics', open)}
        className="space-y-1 data-[state=open]:mb-3"
      >
        {renderSectionHeader('Biophysics')}
        <CollapsibleContent className="space-y-2 px-1.5 pb-1.5 pt-0.5">
          {BIOPHYSICS_KEYS.map((key) => {
            const def = CONFIG_PARAMETER_DEFINITIONS.find((item) => item.key === key)
            if (!def) return null
            const param = parameters.config[key]
            return (
              <ParameterSlider
                key={key}
                label={`biophysics.${key}`}
                info={param.info}
                min={param.min}
                max={param.max}
                step={param.step}
                defaultValue={param.defaultValue}
                value={param.value}
                onValueChange={(value) => setConfigValue(key, value)}
              />
            )
          })}
          {neuronModel === 'mqif' ? (
            <div className="space-y-2 rounded-sm border border-dashed border-border/70 px-1.5 py-1.5">
              <div className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                MQIF
              </div>
              {mqifKeys.map((key) => {
                const parameter = parameters.mqif[key]
                return (
                  <ParameterSlider
                    key={key}
                    label={`biophysics.mqif_${key}`}
                    info={parameter.info}
                    min={parameter.min}
                    max={parameter.max}
                    step={parameter.step}
                    defaultValue={parameter.defaultValue}
                    value={parameter.value}
                    onValueChange={(value) => setMqifValue(key, value)}
                  />
                )
              })}
            </div>
          ) : null}
        </CollapsibleContent>
      </Collapsible>

      <Collapsible
        open={collapsibleOpen.Execution ?? false}
        onOpenChange={(open) => setCollapsibleOpen('Execution', open)}
        className="space-y-1 data-[state=open]:mb-3"
      >
        {renderSectionHeader('Execution')}
        <CollapsibleContent className="space-y-2 px-1.5 pb-1.5 pt-0.5">
          <ParameterSlider
            label="execution.burn_in_ms"
            info={parameters.burnInMs.info}
            min={parameters.burnInMs.min}
            max={parameters.burnInMs.max}
            step={parameters.burnInMs.step}
            defaultValue={parameters.burnInMs.defaultValue}
            value={parameters.burnInMs.value}
            onValueChange={setBurnInMs}
          />
          <ParameterSwitch
            label="execution.performance_mode"
            info={parameters.performanceMode.info}
            defaultChecked={parameters.performanceMode.defaultValue}
            checked={parameters.performanceMode.value}
            onCheckedChange={setPerformanceMode}
          />
          <ParameterSwitch
            label="execution.max_spikes (enabled)"
            info={parameters.downsampleEnabled.info}
            defaultChecked={parameters.downsampleEnabled.defaultValue}
            checked={parameters.downsampleEnabled.value}
            onCheckedChange={setDownsampleEnabled}
          />
          <ParameterSwitch
            label="client.use_arrow_transport"
            info={parameters.useArrowTransport.info}
            defaultChecked={parameters.useArrowTransport.defaultValue}
            checked={parameters.useArrowTransport.value}
            onCheckedChange={setUseArrowTransport}
          />
        </CollapsibleContent>
      </Collapsible>
    </div>
  )
}
