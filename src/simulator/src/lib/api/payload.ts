import type { ParametersState } from '@/hooks/use-parameters'
import type { RunRequest, WeightsRequest } from '@/lib/api/types'

function buildGraphRequest(parameters: ParametersState): Record<string, unknown> {
  const { config, performanceMode, downsampleEnabled, burnInMs, graph } = parameters
  const inputNodes = graph.nodes.filter((node) => node.kind === 'input')

  return {
    schema_version: 'pinglab-graph.v1',
    sim: {
      dt_ms: config.dt.value,
      T_ms: config.T.value,
      seed: Math.round(config.seed.value),
      neuron_model: parameters.neuronModel.value,
    },
    execution: {
      performance_mode: performanceMode.value,
      max_spikes: downsampleEnabled.value ? 30000 : null,
      burn_in_ms: burnInMs.value,
    },
    biophysics: {
      V_init: config.V_init.value,
      E_L: config.E_L.value,
      E_e: config.E_e.value,
      E_i: config.E_i.value,
      C_m_E: config.C_m_E.value,
      g_L_E: config.g_L_E.value,
      C_m_I: config.C_m_I.value,
      g_L_I: config.g_L_I.value,
      V_th: config.V_th.value,
      V_reset: config.V_reset.value,
      t_ref_E: config.t_ref_E.value,
      t_ref_I: config.t_ref_I.value,
      tau_ampa: config.tau_ampa.value,
      tau_gaba: config.tau_gaba.value,
      mqif_a: parameters.neuronModel.value === 'mqif' ? [parameters.mqif.a.value] : undefined,
      mqif_Vr: parameters.neuronModel.value === 'mqif' ? [parameters.mqif.Vr.value] : undefined,
      mqif_w_a:
        parameters.neuronModel.value === 'mqif' ? [parameters.mqif.w_a.value] : undefined,
      mqif_w_Vr:
        parameters.neuronModel.value === 'mqif' ? [parameters.mqif.w_Vr.value] : undefined,
      mqif_w_tau:
        parameters.neuronModel.value === 'mqif' ? [parameters.mqif.w_tau.value] : undefined,
      g_L_heterogeneity_sd: config.g_L_heterogeneity_sd.value,
      C_m_heterogeneity_sd: config.C_m_heterogeneity_sd.value,
      V_th_heterogeneity_sd: config.V_th_heterogeneity_sd.value,
      t_ref_heterogeneity_sd: config.t_ref_heterogeneity_sd.value,
    },
    nodes: graph.nodes.map((node) => ({
      id: node.id,
      kind: node.kind,
      type: node.type,
      size:
        node.kind === 'population'
          ? Math.max(0, Math.round(node.size))
          : 0,
    })),
    edges: graph.edges.map((edge) => ({
      id: edge.id,
      from: edge.from,
      to: edge.to,
      kind: edge.kind,
      w: { mean: edge.w.mean, std: edge.w.std },
      delay_ms: edge.delay_ms,
      ...(edge.kind === 'input' ? { target: { mode: 'all' } } : {}),
    })),
    inputs: Object.fromEntries(
      inputNodes.map((node) => [
        node.id,
        (() => {
          const program = graph.inputPrograms[node.id]
          return {
            mode: program?.mode ?? node.type,
            mean: program?.values.mean ?? 0,
            std: program?.values.std ?? 0,
            I_E_start: program?.values.I_E_start ?? 0,
            I_E_end: program?.values.I_E_end ?? 0,
            I_I_start: program?.values.I_I_start ?? 0,
            I_I_end: program?.values.I_I_end ?? 0,
            seed: Math.round(program?.values.seed ?? 0),
            sine_freq_hz: program?.values.sine_freq_hz ?? 0,
            sine_amp: program?.values.sine_amp ?? 0,
            sine_y_offset: program?.values.sine_y_offset ?? 0,
            sine_phase: program?.values.sine_phase ?? 0,
            sine_phase_offset_i: program?.values.sine_phase_offset_i ?? 0,
            lambda0_hz: program?.values.lambda0_hz ?? 0,
            mod_depth: program?.values.mod_depth ?? 0,
            envelope_freq_hz: program?.values.envelope_freq_hz ?? 0,
            phase_rad: program?.values.phase_rad ?? 0,
            w_in: program?.values.w_in ?? 0,
            tau_in_ms: program?.values.tau_in_ms ?? 3,
            pulse_t_ms: program?.values.pulse_t_ms ?? 0,
            pulse_width_ms: program?.values.pulse_width_ms ?? 1,
            pulse_interval_ms: program?.values.pulse_interval_ms ?? 10,
            pulse_amp_E: program?.values.pulse_amp_E ?? 0,
            pulse_amp_I: program?.values.pulse_amp_I ?? 0,
          }
        })(),
      ])
    ),
    constraints: {
      nonnegative_input: true,
      nonnegative_weights: true,
    },
  }
}

export function buildRunRequest(parameters: ParametersState): RunRequest {
  return {
    graph: buildGraphRequest(parameters),
  }
}

export function buildWeightsRequest(parameters: ParametersState): WeightsRequest {
  const run = buildRunRequest(parameters)
  return {
    graph: run.graph,
  }
}
