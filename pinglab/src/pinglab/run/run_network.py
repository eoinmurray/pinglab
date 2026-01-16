import warnings

import numpy as np

from pinglab.lib import (
    lif_step,
    hh_step,
    hh_init_gating,
    adex_step,
    cs_step,
    cs_init_gating,
    fhn_step,
    izh_step,
    izh_init_u,
    mqif_step,
    qif_step,
    decay_exponential,
)
from pinglab.types import Spikes, InstrumentsResults, NetworkConfig, NetworkResult
from pinglab.run.apply_heterogeneity import apply_heterogeneity

# Numerical stability thresholds for Euler integration
# dt must be less than tau/DT_STABILITY_FACTOR for stability
DT_STABILITY_FACTOR = 5
# dt should be less than tau/DT_ACCURACY_FACTOR for good accuracy
DT_ACCURACY_FACTOR = 10
# Tolerance for firing rate validation (1% above theoretical max)
RATE_VALIDATION_TOLERANCE = 1.01


def run_network(config: NetworkConfig, external_input: np.ndarray) -> NetworkResult:
    """
    Run a conductance-based E/I network simulation.

    Implements either a leaky integrate-and-fire or Hodgkin-Huxley network with:
    - Excitatory (E) and inhibitory (I) populations
    - Exponentially decaying synaptic conductances (AMPA, GABA)
    - Synaptic delays via ring buffers
    - Optional per-neuron heterogeneity
    - Optional instrument recording (voltage, conductance traces)

    Parameters:
        config: Network configuration including neuron counts, time constants,
                coupling strengths, and simulation parameters
        external_input: External input current array of shape (num_steps, N) or (num_steps,)

    Returns:
        NetworkResult containing spike data and optional instrument recordings
    """
    v = config  # v for validated
    N = v.N_E + v.N_I
    num_steps = int(np.ceil(v.T / v.dt))

    # Validate external input shape
    if external_input.ndim == 1:
        if external_input.shape[0] != num_steps:
            raise ValueError(
                f"external_input has {external_input.shape[0]} steps, expected {num_steps}"
            )
    elif external_input.ndim == 2:
        if external_input.shape[0] != num_steps:
            raise ValueError(
                f"external_input has {external_input.shape[0]} steps, expected {num_steps}"
            )
        if external_input.shape[1] != N:
            raise ValueError(
                f"external_input has {external_input.shape[1]} neurons, expected {N}"
            )
    else:
        raise ValueError(
            f"external_input must be 1D or 2D, got {external_input.ndim}D"
        )

    # Numerical stability check for Euler method
    tau_mem_E = v.C_m_E / v.g_L_E
    tau_mem_I = v.C_m_I / v.g_L_I
    tau_min = min(tau_mem_E, tau_mem_I)

    if v.dt > tau_min / DT_STABILITY_FACTOR:
        raise ValueError(
            f"Time step dt={v.dt}ms is too large for numerical stability. "
            f"Minimum membrane time constant is tau_min={tau_min:.2f}ms. "
            f"Require dt < tau_min/{DT_STABILITY_FACTOR} = {tau_min/DT_STABILITY_FACTOR:.2f}ms"
        )

    if v.dt > tau_min / DT_ACCURACY_FACTOR:
        warnings.warn(
            f"dt={v.dt}ms is large relative to tau_min={tau_min:.2f}ms. "
            f"Consider dt < {tau_min/DT_ACCURACY_FACTOR:.2f}ms for better accuracy.",
            UserWarning,
            stacklevel=2,
        )

    rng = np.random.RandomState(v.seed)
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
    m = h = n = None
    cs_a = cs_b = None
    adex_w = None
    fhn_w = None
    izh_u = None
    mqif_a_terms = None
    mqif_vr_terms = None
    if v.neuron_model == "hh":
        m, h, n = hh_init_gating(V)
    elif v.neuron_model == "connor_stevens":
        m, h, n, cs_a, cs_b = cs_init_gating(V)
    elif v.neuron_model == "adex":
        adex_w = np.zeros(N)
    elif v.neuron_model == "fitzhugh":
        fhn_w = np.zeros(N)
    elif v.neuron_model == "izhikevich":
        izh_u = izh_init_u(V, v.izh_b)
    elif v.neuron_model == "mqif":
        mqif_a_terms = np.asarray(v.mqif_a, dtype=float)
        mqif_vr_terms = np.asarray(v.mqif_Vr, dtype=float)
        if mqif_a_terms.size != mqif_vr_terms.size:
            raise ValueError("mqif_a and mqif_Vr must have the same length")
    # Ring buffers for synaptic events
    buf_len = max(delay_ei_steps, delay_ie_steps, delay_ee_steps, delay_ii_steps) + 1
    buffer_e_to_i = np.zeros(buf_len, dtype=int) # E->I buffer
    buffer_i_to_e = np.zeros(buf_len, dtype=int) # I->E buffer
    buffer_e_to_e = np.zeros(buf_len, dtype=int) # E->E buffer
    buffer_i_to_i = np.zeros(buf_len, dtype=int) # I->I buffer
    buf_idx = 0
    # Determine scaling factors
    if v.connectivity_scaling == "one_over_N_src":
        scale_e_to_i = 1.0 / v.N_E if v.N_E > 0 else 0.0
        scale_i_to_e = 1.0 / v.N_I if v.N_I > 0 else 0.0
        scale_e_to_e = 1.0 / v.N_E if v.N_E > 0 else 0.0
        scale_i_to_i = 1.0 / v.N_I if v.N_I > 0 else 0.0
    else:
        scale_e_to_i = 1.0
        scale_i_to_e = 1.0
        scale_e_to_e = 1.0
        scale_i_to_i = 1.0
    # Spike recording
    spike_times = []
    spike_ids = []
    spike_types = []

    # Instrument recording setup
    instrument_recording = v.instruments is not None
    # Initialize all instrument variables unconditionally to avoid "possibly unbound" errors
    instrument_neuron_ids = None
    instrument_downsample = 1
    instrument_variables = []
    instrument_population_means = False
    instrument_times = []
    instrument_V = None
    instrument_g_e = None
    instrument_g_i = None
    instrument_V_mean_E = None
    instrument_V_mean_I = None
    instrument_g_e_mean_E = None
    instrument_g_e_mean_I = None
    instrument_g_i_mean_E = None
    instrument_g_i_mean_I = None
    instrument_step_counter = 0
    
    if instrument_recording and v.instruments is not None:
        instrument_all_neurons = v.instruments.all_neurons
        if instrument_all_neurons:
            instrument_neuron_ids = np.arange(N)
        else:
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

    for step in range(num_steps):
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

        # Synaptic conductance decay (AFTER adding new conductances)
        g_e = decay_exponential(g_e, v.tau_ampa, v.dt)
        g_i = decay_exponential(g_i, v.tau_gaba, v.dt)

        # Decrement refractory countdown
        refractory_countdown[refractory_countdown > 0] -= 1

        # Membrane update (only for non-refractory neurons)
        # Create a mask for neurons that can spike
        can_spike = refractory_countdown == 0

        if v.neuron_model == "lif":
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
        elif v.neuron_model == "hh":
            V_prev = V.copy()
            V, m, h, n = hh_step(
                V,
                m,
                h,
                n,
                g_e,
                g_i,
                I_ext,
                v.dt,
                C_m=C_m_arr,
                g_L=g_L_arr,
                g_Na=v.g_Na,
                g_K=v.g_K,
                E_L=v.E_L,
                E_Na=v.E_Na,
                E_K=v.E_K,
                E_e=v.E_e,
                E_i=v.E_i,
            )
            spiked = (V_prev < V_th_arr) & (V >= V_th_arr) & can_spike
        elif v.neuron_model == "adex":
            V, adex_w, spiked = adex_step(
                V,
                adex_w,
                g_e,
                g_i,
                I_ext,
                v.dt,
                C_m=C_m_arr,
                g_L=g_L_arr,
                E_L=v.E_L,
                E_e=v.E_e,
                E_i=v.E_i,
                V_T=v.adex_V_T,
                Delta_T=v.adex_delta_T,
                tau_w=v.adex_tau_w,
                a=v.adex_a,
                b=v.adex_b,
                V_reset=v.V_reset,
                V_peak=v.adex_V_peak,
                can_spike=can_spike,
            )
        elif v.neuron_model == "connor_stevens":
            V_prev = V.copy()
            V, m, h, n, cs_a, cs_b = cs_step(
                V,
                m,
                h,
                n,
                cs_a,
                cs_b,
                g_e,
                g_i,
                I_ext,
                v.dt,
                C_m=C_m_arr,
                g_L=g_L_arr,
                g_Na=v.g_Na,
                g_K=v.g_K,
                g_A=v.g_A,
                E_L=v.E_L,
                E_Na=v.E_Na,
                E_K=v.E_K,
                E_e=v.E_e,
                E_i=v.E_i,
            )
            spiked = (V_prev < V_th_arr) & (V >= V_th_arr) & can_spike
        elif v.neuron_model == "fitzhugh":
            V_prev = V.copy()
            V, fhn_w = fhn_step(
                V,
                fhn_w,
                g_e,
                g_i,
                I_ext,
                v.dt,
                a=v.fhn_a,
                b=v.fhn_b,
                tau_w=v.fhn_tau_w,
                E_e=v.E_e,
                E_i=v.E_i,
            )
            spiked = (V_prev < V_th_arr) & (V >= V_th_arr) & can_spike
        elif v.neuron_model == "mqif":
            V, spiked = mqif_step(
                V,
                g_e,
                g_i,
                I_ext,
                v.dt,
                C_m=C_m_arr,
                g_L=g_L_arr,
                E_L=v.E_L,
                E_e=v.E_e,
                E_i=v.E_i,
                a_terms=mqif_a_terms,
                V_r_terms=mqif_vr_terms,
                V_th=V_th_arr,
                V_reset=v.V_reset,
                can_spike=can_spike,
            )
        elif v.neuron_model == "qif":
            V, spiked = qif_step(
                V,
                g_e,
                g_i,
                I_ext,
                v.dt,
                C_m=C_m_arr,
                g_L=g_L_arr,
                E_L=v.E_L,
                E_e=v.E_e,
                E_i=v.E_i,
                a=v.qif_a,
                V_r=v.qif_Vr,
                V_t=v.qif_Vt,
                V_th=V_th_arr,
                V_reset=v.V_reset,
                can_spike=can_spike,
            )
        elif v.neuron_model == "izhikevich":
            V, izh_u, spiked = izh_step(
                V,
                izh_u,
                g_e,
                g_i,
                I_ext,
                v.dt,
                a=v.izh_a,
                b=v.izh_b,
                c=v.izh_c,
                d=v.izh_d,
                V_th=V_th_arr,
                E_e=v.E_e,
                E_i=v.E_i,
                can_spike=can_spike,
            )
        else:
            raise ValueError(f"Unsupported neuron_model: {v.neuron_model}")

        # Record spikes and schedule synaptic events
        if spiked.any():
            idxs = np.nonzero(spiked)[0]
            # Set refractory period for spiked neurons (per-neuron)
            refractory_countdown[idxs] = ref_steps_arr[idxs]
            # Reset voltage only for integrate-and-fire style neurons
            if v.neuron_model in {"lif", "adex", "mqif", "qif"}:
                V[idxs] = v.V_reset
            # Record all spikes (filtering by burn_in_ms happens during analysis)
            # Spike types: 0=E (excitatory), 1=I (inhibitory)
            for idx in idxs:
                spike_times.append(t)
                spike_ids.append(idx)
                spike_types.append(0 if idx < v.N_E else 1)
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

        if rate_E > max_possible_rate_E * RATE_VALIDATION_TOLERANCE:
            warnings.warn(
                f"E firing rate {rate_E:.1f} Hz exceeds theoretical max {max_possible_rate_E:.1f} Hz",
                UserWarning,
                stacklevel=2,
            )
        if rate_I > max_possible_rate_I * RATE_VALIDATION_TOLERANCE:
            warnings.warn(
                f"I firing rate {rate_I:.1f} Hz exceeds theoretical max {max_possible_rate_I:.1f} Hz",
                UserWarning,
                stacklevel=2,
            )

        # Check 2: Sanity checks on spike counts
        # Maximum possible spikes: each neuron fires at most once per refractory period
        n_total = len(spike_times)
        min_ref_steps = min(ref_steps_arr)
        max_spikes_per_neuron = num_steps / max(1, min_ref_steps)
        expected_max = N * max_spikes_per_neuron
        if n_total > expected_max * RATE_VALIDATION_TOLERANCE:
            warnings.warn(
                f"Spike count {n_total} exceeds theoretical max {expected_max:.0f}",
                UserWarning,
                stacklevel=2,
            )

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
