
import numpy as np
import sys
from tqdm import tqdm

from pinglab.lib import lif_step, decay_exponential
from pinglab.types import Spikes, InstrumentsResults, NetworkConfig, NetworkResult
from pinglab.run.apply_heterogeneity import apply_heterogeneity


def run_network(config: NetworkConfig) -> NetworkResult:
    v = config # v for validated

    # Validate that external_input is provided
    if v.external_input is None:
        raise ValueError(
            "external_input is required for run_network. "
            "NetworkConfig must include external_input array with shape (num_steps, N_E + N_I) or (num_steps,)"
        )

    external_input = v.external_input

    # Numerical stability check for Euler method
    # Requirement: dt << tau (typically dt < tau/5 for reasonable accuracy)
    tau_mem_E = v.C_m_E / v.g_L_E
    tau_mem_I = v.C_m_I / v.g_L_I
    tau_min = min(tau_mem_E, tau_mem_I)

    if v.dt > tau_min / 5:
        raise ValueError(
            f"Time step dt={v.dt}ms is too large for numerical stability. "
            f"Minimum membrane time constant is tau_min={tau_min:.2f}ms. "
            f"Require dt < tau_min/5 = {tau_min/5:.2f}ms"
        )

    if v.dt > tau_min / 10:
        print(
            f"WARNING: dt={v.dt}ms is large relative to tau_min={tau_min:.2f}ms. "
            f"Consider dt < {tau_min/10:.2f}ms for better accuracy."
        )

    rng = np.random.RandomState(v.seed)
    N = v.N_E + v.N_I
    num_steps = int(np.ceil(v.T / v.dt))
    # Synaptic delays in steps
    delay_ei_steps = max(1, int(np.round(v.delay_ei / v.dt)))
    delay_ie_steps = max(1, int(np.round(v.delay_ie / v.dt)))
    delay_ee_steps = max(1, int(np.round(v.delay_ee / v.dt)))
    delay_ii_steps = max(1, int(np.round(v.delay_ii / v.dt)))

    # Generate heterogeneous neural parameters
    V_th_arr, g_L_arr, C_m_arr, t_ref_arr = apply_heterogeneity(
        rng, v.N_E, v.N_I,
        v.V_th_heterogeneity_sd, v.g_L_heterogeneity_sd,
        v.C_m_heterogeneity_sd, v.t_ref_heterogeneity_sd,
        v.g_L_E, v.g_L_I, v.C_m_E, v.C_m_I, v.V_th, v.t_ref_E, v.t_ref_I
    )

    # Refractory period in steps (per-neuron)
    ref_steps_arr = np.round(t_ref_arr / v.dt).astype(int)

    # State variables
    V = np.full(N, v.V_init)
    g_e = np.zeros(N)
    g_i = np.zeros(N)
    refractory_countdown = np.zeros(N, dtype=int)
    # Ring buffers for synaptic events
    buf_len = max(delay_ei_steps, delay_ie_steps, delay_ee_steps, delay_ii_steps) + 1
    buffer_e_to_i = np.zeros(buf_len, dtype=int) # E->I buffer
    buffer_i_to_e = np.zeros(buf_len, dtype=int) # I->E buffer
    buffer_e_to_e = np.zeros(buf_len, dtype=int) # E->E buffer
    buffer_i_to_i = np.zeros(buf_len, dtype=int) # I->I buffer
    buf_idx = 0
    # Determine scaling factors
    if v.connectivity_scaling == "one_over_N_src":
        scale_e_to_i = 1.0 / v.N_E
        scale_i_to_e = 1.0 / v.N_I
        scale_e_to_e = 1.0 / v.N_E
        scale_i_to_i = 1.0 / v.N_I
    else:
        scale_e_to_i = 1.0
        scale_i_to_e = 1.0
        scale_e_to_e = 1.0
        scale_i_to_i = 1.0
    # Spike recording
    spike_times = []
    spike_ids = []
    spike_types = []
    # Tonic input tracking (mean per population per timestep)
    tonic_input_times = []
    tonic_input_E = []
    tonic_input_I = []

    # Instrument recording setup
    instrument_recording = v.instruments is not None
    if instrument_recording:
        instrument_neuron_ids = np.array(v.instruments.neuron_ids) if v.instruments.neuron_ids is not None else None
        instrument_downsample = v.instruments.downsample
        instrument_variables = v.instruments.variables
        instrument_population_means = v.instruments.population_means
        # Initialize recording arrays
        instrument_times = []
        instrument_V = [] if 'V' in instrument_variables and instrument_neuron_ids is not None else None
        instrument_g_e = [] if 'g_e' in instrument_variables and instrument_neuron_ids is not None else None
        instrument_g_i = [] if 'g_i' in instrument_variables and instrument_neuron_ids is not None else None
        # Population mean arrays
        instrument_V_mean_E = [] if 'V' in instrument_variables and instrument_population_means else None
        instrument_V_mean_I = [] if 'V' in instrument_variables and instrument_population_means else None
        instrument_g_e_mean_E = [] if 'g_e' in instrument_variables and instrument_population_means else None
        instrument_g_e_mean_I = [] if 'g_e' in instrument_variables and instrument_population_means else None
        instrument_g_i_mean_E = [] if 'g_i' in instrument_variables and instrument_population_means else None
        instrument_g_i_mean_I = [] if 'g_i' in instrument_variables and instrument_population_means else None
        instrument_step_counter = 0
    # Simulation loop with progress bar
    for step in tqdm(range(num_steps), desc="Simulating", unit=" steps", file=sys.stdout, leave=False, dynamic_ncols=True):
        t = step * v.dt
        # Apply delayed synaptic events
        nE_to_i = buffer_e_to_i[buf_idx]
        nI_to_e = buffer_i_to_e[buf_idx]
        nE_to_e = buffer_e_to_e[buf_idx]
        nI_to_i = buffer_i_to_i[buf_idx]

        if nE_to_i:
            # E spikes cause excitatory conductance to inhibitory neurons
            g_e[v.N_E:] += nE_to_i * v.g_ei * scale_e_to_i * v.p_ei
        if nI_to_e:
            # I spikes cause inhibitory conductance to excitatory neurons
            g_i[:v.N_E] += nI_to_e * v.g_ie * scale_i_to_e * v.p_ie
        # Intra-population connections
        if nE_to_e:
            g_e[:v.N_E] += nE_to_e * v.g_ee * scale_e_to_e * v.p_ee
        if nI_to_i:
            g_i[v.N_E:] += nI_to_i * v.g_ii * scale_i_to_i * v.p_ii

        # Clear current buffer slots
        buffer_e_to_i[buf_idx] = 0
        buffer_i_to_e[buf_idx] = 0
        buffer_e_to_e[buf_idx] = 0
        buffer_i_to_i[buf_idx] = 0

        # Use external input (required)
        if external_input.ndim == 1:
            # Scalar time series - broadcast to all neurons
            I_ext = np.full(N, external_input[step])
        else:
            # Per-neuron time series
            I_ext = external_input[step, :]

        # Record input (mean per population)
        tonic_input_times.append(t)
        tonic_input_E.append(np.mean(I_ext[:v.N_E]))
        tonic_input_I.append(np.mean(I_ext[v.N_E:]))

        # Synaptic conductance decay (AFTER adding new conductances)
        g_e = decay_exponential(g_e, v.tau_ampa, v.dt)
        g_i = decay_exponential(g_i, v.tau_gaba, v.dt)

        # Decrement refractory countdown
        refractory_countdown[refractory_countdown > 0] -= 1

        # Membrane update (only for non-refractory neurons)
        # Create a mask for neurons that can spike
        can_spike = refractory_countdown == 0

        V, spiked = lif_step(
            V,
            g_e,
            g_i,
            I_ext,
            v.dt,
            E_L=v.E_L,
            E_e=v.E_e,
            E_i=v.E_i,
            C_m=C_m_arr,
            g_L=g_L_arr,
            V_th=V_th_arr,
            V_reset=v.V_reset,
            can_spike=can_spike,
        )

        # Clear spikes for neurons in refractory period
        spiked = spiked & can_spike

        # Record spikes and schedule synaptic events
        if spiked.any():
            idxs = np.nonzero(spiked)[0]
            # Set refractory period for spiked neurons (per-neuron)
            refractory_countdown[idxs] = ref_steps_arr[idxs]
            # Reset voltage for spiked neurons
            V[idxs] = v.V_reset
            # Record all spikes (filtering by burn_in_ms happens during analysis)
            for idx in idxs:
                spike_times.append(t)
                spike_ids.append(int(idx))
                spike_types.append(np.uint8(0 if idx < v.N_E else 1)) # 0=E,1=I
            # Count population spikes
            nE_spikes = int((idxs < v.N_E).sum())
            nI_spikes = int((idxs >= v.N_E).sum())
            # Schedule events at future buffer position
            tgt_ei = (buf_idx + delay_ei_steps) % buf_len
            tgt_ie = (buf_idx + delay_ie_steps) % buf_len
            tgt_ee = (buf_idx + delay_ee_steps) % buf_len
            tgt_ii = (buf_idx + delay_ii_steps) % buf_len
            buffer_e_to_i[tgt_ei] += nE_spikes
            buffer_i_to_e[tgt_ie] += nI_spikes
            buffer_e_to_e[tgt_ee] += nE_spikes
            buffer_i_to_i[tgt_ii] += nI_spikes

        # Instrument recording
        if instrument_recording:
            if instrument_step_counter % instrument_downsample == 0:
                instrument_times.append(t)
                # Individual neuron recordings
                if instrument_V is not None:
                    instrument_V.append(V[instrument_neuron_ids].copy())
                if instrument_g_e is not None:
                    instrument_g_e.append(g_e[instrument_neuron_ids].copy())
                if instrument_g_i is not None:
                    instrument_g_i.append(g_i[instrument_neuron_ids].copy())
                # Population mean recordings
                if instrument_V_mean_E is not None:
                    instrument_V_mean_E.append(np.mean(V[:v.N_E]))
                if instrument_V_mean_I is not None:
                    instrument_V_mean_I.append(np.mean(V[v.N_E:]))
                if instrument_g_e_mean_E is not None:
                    instrument_g_e_mean_E.append(np.mean(g_e[:v.N_E]))
                if instrument_g_e_mean_I is not None:
                    instrument_g_e_mean_I.append(np.mean(g_e[v.N_E:]))
                if instrument_g_i_mean_E is not None:
                    instrument_g_i_mean_E.append(np.mean(g_i[:v.N_E]))
                if instrument_g_i_mean_I is not None:
                    instrument_g_i_mean_I.append(np.mean(g_i[v.N_E:]))
            instrument_step_counter += 1

        # Advance buffer index
        buf_idx = (buf_idx + 1) % buf_len

    # Post-simulation validation checks
    if len(spike_times) > 0:
        # Check 1: Firing rate limits
        max_possible_rate_E = 1.0 / (v.t_ref_E / 1000.0)  # Convert ms to seconds
        max_possible_rate_I = 1.0 / (v.t_ref_I / 1000.0)  # Convert ms to seconds
        rate_E = len([s for s in spike_types if s == 0]) / (v.N_E * (v.T / 1000.0)) if v.N_E > 0 else 0
        rate_I = len([s for s in spike_types if s == 1]) / (v.N_I * (v.T / 1000.0)) if v.N_I > 0 else 0

        if rate_E > max_possible_rate_E * 1.01:  # Allow 1% tolerance
            print(f"WARNING: E firing rate {rate_E:.1f} Hz exceeds theoretical max {max_possible_rate_E:.1f} Hz")
        if rate_I > max_possible_rate_I * 1.01:
            print(f"WARNING: I firing rate {rate_I:.1f} Hz exceeds theoretical max {max_possible_rate_I:.1f} Hz")

        # Check 2: Sanity checks on spike counts
        n_total = len(spike_times)
        expected_max = v.N_E * v.N_I * (v.T / v.dt)  # Very loose upper bound
        if n_total > expected_max:
            print(f"WARNING: Spike count {n_total} seems unreasonably high")

    spikes = Spikes(
        times=np.array(spike_times),
        ids=np.array(spike_ids),
        types=np.array(spike_types)
    )

    # Build InstrumentsResults if recording was active
    instruments_result = None
    if instrument_recording:
        # Determine types for instrumented neurons (0=E, 1=I)
        instrument_types = np.array([0 if nid < v.N_E else 1 for nid in instrument_neuron_ids], dtype=np.uint8) if instrument_neuron_ids is not None else None
        instruments_result = InstrumentsResults(
            times=np.array(instrument_times),
            neuron_ids=instrument_neuron_ids if instrument_neuron_ids is not None else np.array([]),
            V=np.array(instrument_V) if instrument_V is not None else None,
            g_e=np.array(instrument_g_e) if instrument_g_e is not None else None,
            g_i=np.array(instrument_g_i) if instrument_g_i is not None else None,
            types=instrument_types,
            V_mean_E=np.array(instrument_V_mean_E) if instrument_V_mean_E is not None else None,
            V_mean_I=np.array(instrument_V_mean_I) if instrument_V_mean_I is not None else None,
            g_e_mean_E=np.array(instrument_g_e_mean_E) if instrument_g_e_mean_E is not None else None,
            g_e_mean_I=np.array(instrument_g_e_mean_I) if instrument_g_e_mean_I is not None else None,
            g_i_mean_E=np.array(instrument_g_i_mean_E) if instrument_g_i_mean_E is not None else None,
            g_i_mean_I=np.array(instrument_g_i_mean_I) if instrument_g_i_mean_I is not None else None,
        )

    return NetworkResult(
        spikes=spikes,
        instruments=instruments_result,
    )