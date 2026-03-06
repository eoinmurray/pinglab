"""E-prop (eligibility propagation) for conductance-based LIF networks.

Replaces BPTT with forward-running eligibility traces. The (E_syn - V) driving
force appears additively in each trace update rather than multiplicatively
through time, avoiding the gradient explosion that plagues BPTT in
conductance-based models.

Reference: Bellec et al., "A solution to the learning dilemma for recurrent
networks of spiking neurons", Nature Communications 11, 3625 (2020).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def run_batch_eprop(
    runtime: Any,
    images: "torch.Tensor",
    *,
    T_steps: int,
    n_total: int,
    n_input: int,
    out_start: int,
    out_stop: int,
    input_scale: float,
    sim_state: Any,
    burn_in_steps: int = 0,
    readout_alpha: float = 0.01,
    encoding: str = "poisson",
    pop_idx: dict[str, dict[str, int]],
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Run a batch through the SNN with e-prop eligibility traces.

    Instead of loss.backward(), this function computes gradients analytically
    via eligibility traces and writes them directly into .grad fields.

    Args:
        runtime: Compiled network runtime.
        images: Image batch [B, ...].
        pop_idx: Population index dict with E_in, E_hid, E_out start/stop.
        (other args same as run_batch)

    Returns:
        (loss, logits, spikes_E) where loss is scalar, logits is [B, 10],
        spikes_E is [B, T, N_E] for firing rate tracking.
    """
    import torch
    import torch.nn.functional as F

    from pinglab.io.training import encode_poisson, encode_rate_to_tonic
    from pinglab.backends.pytorch.simulate_network import (
        reset_simulation_state,
        apply_delayed_events,
        emit_and_schedule_spikes,
        lif_step,
    )

    if encoding == "poisson":
        encode_fn = encode_poisson
    else:
        encode_fn = encode_rate_to_tonic

    cfg = runtime.config
    w = runtime.weights
    dev = sim_state.V.device

    # Population slices in E-neuron space
    in_start = int(pop_idx["E_in"]["start"])
    in_stop = int(pop_idx["E_in"]["stop"])
    hid_start = int(pop_idx["E_hid"]["start"])
    hid_stop = int(pop_idx["E_hid"]["stop"])
    # out_start, out_stop already passed as args

    n_in = in_stop - in_start
    n_hid = hid_stop - hid_start
    n_out = out_stop - out_start
    N_E = sim_state.N_E
    N_I = sim_state.N_I

    # Biophysical constants
    dt = float(cfg.dt)
    E_L = float(cfg.E_L)
    E_e = float(cfg.E_e)
    E_i = float(cfg.E_i)
    V_reset = float(cfg.V_reset)
    V_floor = getattr(cfg, "V_floor", None)
    if V_floor is not None:
        V_floor = float(V_floor)

    # Membrane leak factor: α = 1 - dt·g_L/C_m (per neuron)
    # g_L_arr and C_m_arr are [N_total], V_th_arr is [N_total]
    g_L_arr = sim_state.g_L_arr  # [N]
    C_m_arr = sim_state.C_m_arr  # [N]
    V_th_arr = sim_state.V_th_arr  # [N]

    # Per-neuron leak factor (only need E and I separately for traces)
    alpha_all = 1.0 - dt * g_L_arr / C_m_arr  # [N]
    alpha_E = alpha_all[:N_E]  # [N_E]
    alpha_I = alpha_all[N_E:]  # [N_I]

    dt_over_Cm_E = dt / C_m_arr[:N_E]  # [N_E]
    dt_over_Cm_I = dt / C_m_arr[N_E:]  # [N_I]

    V_th_E = V_th_arr[:N_E]  # [N_E]
    V_th_I = V_th_arr[N_E:]  # [N_I]

    # ── Encode input ──────────────────────────────────────────────────────
    B_actual = images.shape[0]
    ext_batch = torch.stack(
        [encode_fn(img, T_steps=T_steps, n_total=n_total, n_input=n_input, scale=input_scale)
         for img in images],
        dim=0,
    ).to(device=dev)  # [B, T, N]

    B_state = sim_state.batch_size
    if B_actual < B_state:
        pad = torch.zeros(B_state - B_actual, T_steps, n_total,
                          dtype=ext_batch.dtype, device=dev)
        ext_batch = torch.cat([ext_batch, pad], dim=0)

    B = B_state

    # ── Reset simulation state ────────────────────────────────────────────
    reset_simulation_state(sim_state, runtime)
    sim_state.external_input = ext_batch
    sim_state.detach_spikes = True  # e-prop doesn't need BPTT through buffers

    # ── Initialize eligibility vectors (zeros) ────────────────────────────
    # eps_in_hid: [B, n_hid, n_in] — how V_hid depends on W_ee[hid, in]
    eps_in_hid = torch.zeros(B, n_hid, n_in, device=dev)
    # eps_hid_out: [B, n_out, n_hid] — how V_out depends on W_ee[out, hid]
    eps_hid_out = torch.zeros(B, n_out, n_hid, device=dev)
    # eps_ei: [B, N_I, n_hid] — how V_I depends on W_ei[:, hid]
    eps_ei = torch.zeros(B, N_I, n_hid, device=dev)
    # eps_ie: [B, n_hid, N_I] — how V_hid depends on W_ie[hid, :]
    eps_ie = torch.zeros(B, n_hid, N_I, device=dev)

    # Accumulated eligibility traces (sum of ψ·ε over time)
    E_in_hid = torch.zeros(B, n_hid, n_in, device=dev)
    E_hid_out = torch.zeros(B, n_out, n_hid, device=dev)
    E_ei = torch.zeros(B, N_I, n_hid, device=dev)
    E_ie = torch.zeros(B, n_hid, N_I, device=dev)

    # Readout accumulators
    spike_counts = torch.zeros(B, n_out, device=dev)
    voltage_sum = torch.zeros(B, n_out, device=dev)

    # For firing rate tracking
    all_spikes_E = []
    all_spikes_I = []

    # ── Forward loop (no autograd) ────────────────────────────────────────
    with torch.no_grad():
        for t in range(T_steps):
            # Read delayed spikes BEFORE integrate_step clears them
            # These are the spikes that arrived at this timestep after delay
            ee_idx = sim_state.buf_idx
            ie_idx = sim_state.buf_idx
            delayed_E = sim_state.buffer_e_to_e[ee_idx].clone()  # [B, N_E]
            delayed_I = sim_state.buffer_i_to_e[ie_idx].clone() if N_I > 0 else None  # [B, N_I]

            # LIF step (hard threshold)
            spiked = _integrate_step_eprop(sim_state, runtime, step=t)
            # spiked: [B, N] bool

            spiked_E = spiked[:, :N_E].float()  # [B, N_E]
            all_spikes_E.append(spiked_E[:B_actual].clone())
            if N_I > 0:
                spiked_I = spiked[:, N_E:].float()  # [B, N_I]
                all_spikes_I.append(spiked_I[:B_actual].clone())

            # ── Surrogate derivative ψ for all neurons ────────────────────
            V_E = sim_state.V[:, :N_E]  # [B, N_E]
            V_I = sim_state.V[:, N_E:]  # [B, N_I]
            psi_E = 1.0 / (1.0 + (V_E - V_th_E).abs()) ** 2  # [B, N_E]
            psi_I = 1.0 / (1.0 + (V_I - V_th_I).abs()) ** 2  # [B, N_I]

            # ── Update eligibility vectors ────────────────────────────────
            # For W_ee[hid, in]: post=hid, pre=in
            # ε(t+1) = α_hid · ε(t) + (dt/C_m_hid) · (E_e - V_hid) · delayed_E_in
            V_hid = V_E[:, hid_start:hid_stop]  # [B, n_hid]
            alpha_hid = alpha_E[hid_start:hid_stop]  # [n_hid]
            dt_Cm_hid = dt_over_Cm_E[hid_start:hid_stop]  # [n_hid]

            delayed_in = delayed_E[:, in_start:in_stop]  # [B, n_in]
            driving_e_hid = (E_e - V_hid) * dt_Cm_hid  # [B, n_hid]
            eps_in_hid = alpha_hid.unsqueeze(0).unsqueeze(2) * eps_in_hid + \
                driving_e_hid.unsqueeze(2) * delayed_in.unsqueeze(1)
            # [B, n_hid, n_in]

            # For W_ee[out, hid]: post=out, pre=hid
            V_out = V_E[:, out_start:out_stop]  # [B, n_out]
            alpha_out = alpha_E[out_start:out_stop]  # [n_out]
            dt_Cm_out = dt_over_Cm_E[out_start:out_stop]  # [n_out]

            delayed_hid = delayed_E[:, hid_start:hid_stop]  # [B, n_hid]
            driving_e_out = (E_e - V_out) * dt_Cm_out  # [B, n_out]
            eps_hid_out = alpha_out.unsqueeze(0).unsqueeze(2) * eps_hid_out + \
                driving_e_out.unsqueeze(2) * delayed_hid.unsqueeze(1)
            # [B, n_out, n_hid]

            # For W_ei[:, hid]: post=I, pre=E_hid (excitatory synapse → E_e)
            alpha_I_vec = alpha_I  # [N_I]
            dt_Cm_I_vec = dt_over_Cm_I  # [N_I]
            driving_e_I = (E_e - V_I) * dt_Cm_I_vec  # [B, N_I]
            # Use pre-cloned delayed_E (buffer_e_to_i was cleared by integrate step)
            delayed_E_for_I = delayed_E[:, hid_start:hid_stop]  # [B, n_hid]
            eps_ei = alpha_I_vec.unsqueeze(0).unsqueeze(2) * eps_ei + \
                driving_e_I.unsqueeze(2) * delayed_E_for_I.unsqueeze(1)
            # [B, N_I, n_hid]

            # For W_ie[hid, :]: post=E_hid, pre=I (inhibitory synapse → E_i)
            if delayed_I is not None:
                driving_i_hid = (E_i - V_hid) * dt_Cm_hid  # [B, n_hid]
                eps_ie = alpha_hid.unsqueeze(0).unsqueeze(2) * eps_ie + \
                    driving_i_hid.unsqueeze(2) * delayed_I.unsqueeze(1)
                # [B, n_hid, N_I]

            # ── Accumulate e_ij(t) = ψ_j(t) · ε_ij(t) ──────────────────
            psi_hid = psi_E[:, hid_start:hid_stop]  # [B, n_hid]
            psi_out = psi_E[:, out_start:out_stop]   # [B, n_out]

            if t >= burn_in_steps:
                E_in_hid += psi_hid.unsqueeze(2) * eps_in_hid
                E_hid_out += psi_out.unsqueeze(2) * eps_hid_out
                E_ei += psi_I.unsqueeze(2) * eps_ei
                E_ie += psi_hid.unsqueeze(2) * eps_ie

            # ── Emit spikes, advance buffer ───────────────────────────────
            emit_and_schedule_spikes(sim_state, spiked)
            sim_state.buf_idx = (sim_state.buf_idx + 1) % sim_state.buf_len

            # ── Accumulate readout ────────────────────────────────────────
            if t >= burn_in_steps:
                spike_counts += spiked_E[:, out_start:out_stop]
                voltage_sum += sim_state.V[:, out_start:out_stop]

    # ── Compute logits and loss ───────────────────────────────────────────
    T_active = max(T_steps - burn_in_steps, 1)
    logits = spike_counts + readout_alpha * voltage_sum / T_active  # [B, 10]
    logits = logits[:B_actual]

    # Stack spikes for return
    spikes_E = torch.stack(all_spikes_E, dim=1)  # [B_actual, T, N_E]
    spikes_I = torch.stack(all_spikes_I, dim=1) if all_spikes_I else None  # [B_actual, T, N_I]

    return logits, spikes_E, spikes_I, (
        E_in_hid[:B_actual], E_hid_out[:B_actual],
        E_ei[:B_actual], E_ie[:B_actual],
    )


def compute_eprop_gradients(
    runtime: Any,
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    traces: tuple,
    *,
    pop_idx: dict[str, dict[str, int]],
    spikes_E: "torch.Tensor | None" = None,
    dt_ms: float = 1.0,
    rate_reg: float = 0.0,
    rate_target_hz: float = 10.0,
) -> "torch.Tensor":
    """Compute e-prop learning signals and write gradients into .grad fields.

    Args:
        runtime: Network runtime with weight matrices.
        logits: [B, 10] output logits.
        labels: [B] integer labels.
        traces: Tuple of (E_in_hid, E_hid_out, E_ei, E_ie) accumulated traces.
        spikes_E: [B, T, N_E] spike tensor (needed if rate_reg > 0).
        dt_ms: Timestep in ms (for Hz conversion).
        rate_reg: Firing rate regularization strength (0 = off).
        rate_target_hz: Target firing rate in Hz.

    Returns:
        Scalar loss value.
    """
    import torch
    import torch.nn.functional as F

    w = runtime.weights
    B = logits.shape[0]

    hid_start = int(pop_idx["E_hid"]["start"])
    hid_stop = int(pop_idx["E_hid"]["stop"])
    out_start = int(pop_idx["E_out"]["start"])
    out_stop = int(pop_idx["E_out"]["stop"])

    E_in_hid, E_hid_out, E_ei, E_ie = traces

    # Cross-entropy loss (for logging)
    loss = F.cross_entropy(logits, labels)

    # Analytical gradient of cross-entropy w.r.t. logits
    probs = F.softmax(logits, dim=1)  # [B, 10]
    one_hot = F.one_hot(labels, num_classes=logits.shape[1]).float()  # [B, 10]
    L_out = probs - one_hot  # [B, 10] learning signal for output neurons

    # ── Firing rate regularization ─────────────────────────────────────────
    # L_j^rate = 2λ(r_j - r_target) added to the learning signal
    if rate_reg > 0.0 and spikes_E is not None:
        T = spikes_E.shape[1]
        rate_target = rate_target_hz * dt_ms / 1000.0  # convert Hz to spikes/step
        # Mean firing probability per neuron per sample: [B, N_E]
        mean_rate = spikes_E.mean(dim=1)  # [B, N_E]
        rate_error = 2.0 * rate_reg * (mean_rate - rate_target)  # [B, N_E]
        # Add rate regularization to output and hidden learning signals
        L_out = L_out + rate_error[:, out_start:out_stop]
        # L_rate_hid will be added after L_hid is computed from classification
        L_rate_hid = rate_error[:, hid_start:hid_stop]  # [B, n_hid]
        # Add rate penalty to logged loss
        rate_loss = rate_reg * ((mean_rate - rate_target) ** 2).mean()
        loss = loss + rate_loss
    else:
        L_rate_hid = None

    # Symmetric e-prop: propagate learning signal to hidden layers
    # W_ee[out_sl, hid_sl] maps E_hid → E_out
    W_hid_out = w.W_ee[out_start:out_stop, hid_start:hid_stop]  # [n_out, n_hid]
    L_hid = L_out @ W_hid_out  # [B, n_hid]

    # Add direct rate regularization to hidden learning signal
    if L_rate_hid is not None:
        L_hid = L_hid + L_rate_hid

    # Learning signal for I neurons
    W_i_ehid = w.W_ie[hid_start:hid_stop, :]  # [n_hid, N_I]
    L_I = L_hid @ W_i_ehid  # [B, N_I]

    # ── Write gradients ───────────────────────────────────────────────────
    # grad_W = (1/B) * Σ_b L_j^b · E_ij^b
    # For W_ee: need to write into the correct sub-blocks

    # Initialize full gradient tensors
    grad_W_ee = torch.zeros_like(w.W_ee)

    # W_ee[hid, in]: grad = einsum('bj,bji->ji', L_hid, E_in_hid) / B
    grad_W_ee[hid_start:hid_stop, :int(pop_idx["E_in"]["stop"])] = \
        torch.einsum('bj,bji->ji', L_hid, E_in_hid) / B

    # W_ee[out, hid]: grad = einsum('bj,bji->ji', L_out, E_hid_out) / B
    grad_W_ee[out_start:out_stop, hid_start:hid_stop] = \
        torch.einsum('bj,bji->ji', L_out, E_hid_out) / B

    w.W_ee.grad = grad_W_ee

    # W_ei[I, E_hid]: grad for the E_hid→I subblock
    if w.W_ei is not None:
        grad_W_ei = torch.zeros_like(w.W_ei)
        grad_W_ei[:, hid_start:hid_stop] = \
            torch.einsum('bj,bji->ji', L_I, E_ei) / B
        w.W_ei.grad = grad_W_ei

    # W_ie[E_hid, I]: grad for the I→E_hid subblock
    if w.W_ie is not None:
        grad_W_ie = torch.zeros_like(w.W_ie)
        grad_W_ie[hid_start:hid_stop, :] = \
            torch.einsum('bj,bji->ji', L_hid, E_ie) / B
        w.W_ie.grad = grad_W_ie

    return loss


def _integrate_step_eprop(
    state: Any,
    runtime: Any,
    *,
    step: int,
) -> "torch.Tensor":
    """Single LIF integration step using hard threshold (no surrogate needed).

    Replicates integrate_step but uses lif_step directly.
    """
    import torch

    from pinglab.backends.pytorch.simulate_network import (
        apply_delayed_events,
        lif_step,
    )

    assert state.V is not None and state.g_e is not None and state.g_i is not None

    apply_delayed_events(state, runtime)

    state.g_e = state.g_e * state.decay_e
    state.g_i = state.g_i * state.decay_i

    state.refractory_countdown = (state.refractory_countdown - 1).clamp(min=0)
    can_spike = state.refractory_countdown == 0

    I_ext = state.external_input[:, step, :]

    V_floor = getattr(runtime.config, "V_floor", None)
    V_new, spiked = lif_step(
        state.V,
        state.g_e,
        state.g_i,
        I_ext,
        state.dt,
        E_L=float(runtime.config.E_L),
        E_e=float(runtime.config.E_e),
        E_i=float(runtime.config.E_i),
        C_m=state.C_m_arr,
        g_L=state.g_L_arr,
        V_th=state.V_th_arr,
        V_reset=float(runtime.config.V_reset),
        can_spike=can_spike,
        V_floor=float(V_floor) if V_floor is not None else None,
    )
    state.V = V_new

    spiked_bool = spiked.detach().bool()
    ref_expanded = state.ref_steps_arr.unsqueeze(0).expand_as(state.refractory_countdown)
    state.refractory_countdown = state.refractory_countdown.clone()
    state.refractory_countdown[spiked_bool] = ref_expanded[spiked_bool]

    return spiked


def train_epoch_eprop(
    dataloader: Any,
    optimizer: Any,
    runtime: Any,
    *,
    n_total_samples: int,
    device: str = "cpu",
    max_grad_norm: float | None = None,
    batch_kwargs: dict,
    pop_idx: dict,
    iter_rates: dict | None = None,
    layer_slices: dict | None = None,
    burn_in_steps: int = 0,
    T_sec: float = 0.1,
    grad_norms: dict | None = None,
    rate_reg: float = 0.0,
    rate_target_hz: float = 10.0,
    dt_ms: float = 1.0,
) -> "tuple[float, list[float], list[float]]":
    """One training epoch using e-prop.

    Similar to train_epoch but calls run_batch_eprop + compute_eprop_gradients
    instead of forward + loss.backward().
    """
    import torch

    total_loss = 0.0
    n_batches = 0
    samples_done = 0
    iter_losses: list[float] = []
    iter_accs: list[float] = []
    epoch_start = time.perf_counter()
    log_every = max(1, len(dataloader) // 10)

    for batch_idx, (X, y) in enumerate(dataloader):
        y = y.to(device)
        batch_start = time.perf_counter()

        # Forward pass with e-prop trace accumulation
        logits, spikes_E, spikes_I, traces = run_batch_eprop(
            runtime, X, pop_idx=pop_idx, **batch_kwargs,
        )

        # Compute gradients via e-prop learning signals
        loss = compute_eprop_gradients(
            runtime, logits, y, traces, pop_idx=pop_idx,
            spikes_E=spikes_E, dt_ms=dt_ms,
            rate_reg=rate_reg, rate_target_hz=rate_target_hz,
        )

        # Track gradient norms before clipping
        if grad_norms is not None:
            w = runtime.weights
            for name, param in [("W_ee", w.W_ee), ("W_ei", w.W_ei), ("W_ie", w.W_ie)]:
                if param is not None and param.grad is not None and name in grad_norms:
                    grad_norms[name].append(param.grad.norm().item())

        # Clip and step
        if max_grad_norm is not None:
            all_params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
            if all_params:
                torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        batch_loss = loss.item()
        batch_acc = (logits.detach().argmax(1) == y).float().mean().item()
        iter_losses.append(batch_loss)
        iter_accs.append(batch_acc)
        total_loss += batch_loss
        n_batches += 1
        samples_done += X.shape[0]

        # Track firing rates
        if iter_rates is not None and layer_slices is not None:
            with torch.no_grad():
                for name, (s, e) in layer_slices.items():
                    rate = spikes_E[:, burn_in_steps:, s:e].sum(dim=1).mean().item() / T_sec
                    iter_rates[name].append(rate)
                if "I_global" in iter_rates and spikes_I is not None:
                    rate = spikes_I[:, burn_in_steps:, :].sum(dim=1).mean().item() / T_sec
                    iter_rates["I_global"].append(rate)

        if batch_idx % log_every == 0 or batch_idx == len(dataloader) - 1:
            elapsed = time.perf_counter() - epoch_start
            sps = samples_done / elapsed if elapsed > 0 else 0.0
            remaining = (n_total_samples - samples_done) / sps if sps > 0 else float("inf")
            pct = 100 * samples_done / n_total_samples
            batch_ms = 1000 * (time.perf_counter() - batch_start) / X.shape[0]
            print(
                f"  [{pct:5.1f}%] batch {batch_idx:>4d}/{len(dataloader)-1}"
                f"  loss: {batch_loss:.4f}  acc: {100*batch_acc:.1f}%"
                f"  {sps:.1f} samp/s  {batch_ms:.0f} ms/samp"
                f"  ETA: {remaining:.0f}s",
                flush=True,
            )

    return total_loss / max(n_batches, 1), iter_losses, iter_accs
