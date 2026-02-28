"""Readable PyTorch network simulator (first pass)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from pinglab.backends.types import Spikes

if TYPE_CHECKING:
    import torch


def _as_tensor(value: float | "torch.Tensor", ref: "torch.Tensor") -> "torch.Tensor":
    if hasattr(value, "shape"):
        return value  # type: ignore[return-value]
    import torch

    return torch.full_like(ref, float(value))


def lif_step(
    V: "torch.Tensor",
    g_e: "torch.Tensor",
    g_i: "torch.Tensor",
    I_ext: "torch.Tensor",
    dt: float,
    *,
    E_L: float,
    E_e: float,
    E_i: float,
    C_m: float | "torch.Tensor",
    g_L: float | "torch.Tensor",
    V_th: float | "torch.Tensor",
    V_reset: float,
    can_spike: "torch.Tensor | None" = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch backend requires torch to be installed") from exc

    if can_spike is None:
        can_spike = torch.ones_like(V, dtype=torch.bool)

    C_m_t = _as_tensor(C_m, V)
    g_L_t = _as_tensor(g_L, V)
    V_th_t = _as_tensor(V_th, V)

    dVdt = (g_L_t * (E_L - V) + g_e * (E_e - V) + g_i * (E_i - V) + I_ext) / C_m_t
    V_new = V + float(dt) * dVdt
    spiked = (V_new >= V_th_t) & can_spike

    if bool(torch.any(spiked)):
        V_new = V_new.clone()
        V_new[spiked] = float(V_reset)

    V_new = V_new.clone()
    V_new[~can_spike] = float(V_reset)
    return V_new, spiked


@dataclass
class SimulationState:
    # Global simulation shape/config
    N: int
    N_E: int
    N_I: int
    num_steps: int
    dt: float
    decay_e: float
    decay_i: float
    dtype: str = "float32"

    # Core per-neuron mutable state
    V: "torch.Tensor | None" = None
    g_e: "torch.Tensor | None" = None
    g_i: "torch.Tensor | None" = None
    refractory_countdown: "torch.Tensor | None" = None
    V_th_arr: "torch.Tensor | None" = None
    g_L_arr: "torch.Tensor | None" = None
    C_m_arr: "torch.Tensor | None" = None
    ref_steps_arr: "torch.Tensor | None" = None

    # Inputs and current-step working values
    external_input: "torch.Tensor | None" = None
    I_ext: "torch.Tensor | None" = None
    can_spike: "torch.Tensor | None" = None
    spiked: "torch.Tensor | None" = None
    idxs: "torch.Tensor | None" = None

    # Spike accumulation (python lists for easy append)
    spike_times: list[float] = field(default_factory=list)
    spike_ids: list[int] = field(default_factory=list)
    spike_types: list[int] = field(default_factory=list)

    # Per-step neuron traces (shape: [num_steps, N])
    voltage_trace: np.ndarray | None = None
    syn_current_trace: np.ndarray | None = None
    ext_current_trace: np.ndarray | None = None
    total_current_trace: np.ndarray | None = None

    # Delay/buffer path
    delay_ei_steps: int = 1
    delay_ie_steps: int = 1
    delay_ee_steps: int = 1
    delay_ii_steps: int = 1
    buf_len: int = 2
    buf_idx: int = 0
    buffer_e_to_i: "torch.Tensor | None" = None
    buffer_e_to_e: "torch.Tensor | None" = None
    buffer_i_to_e: "torch.Tensor | None" = None
    buffer_i_to_i: "torch.Tensor | None" = None
    use_delay_overrides: bool = False
    buffer_g_e: "torch.Tensor | None" = None
    buffer_g_i: "torch.Tensor | None" = None
    ee_targets: list[Any] = field(default_factory=list)
    ee_weights: list[Any] = field(default_factory=list)
    ee_delays: list[Any] = field(default_factory=list)
    ei_targets: list[Any] = field(default_factory=list)
    ei_weights: list[Any] = field(default_factory=list)
    ei_delays: list[Any] = field(default_factory=list)
    ie_targets: list[Any] = field(default_factory=list)
    ie_weights: list[Any] = field(default_factory=list)
    ie_delays: list[Any] = field(default_factory=list)
    ii_targets: list[Any] = field(default_factory=list)
    ii_weights: list[Any] = field(default_factory=list)
    ii_delays: list[Any] = field(default_factory=list)

    # Bookkeeping
    step: int = 0
    t_ms: float = 0.0


@dataclass(frozen=True)
class SimulationResult:
    spikes: Spikes
    t_ms: np.ndarray
    voltage_trace: np.ndarray
    syn_current_trace: np.ndarray
    ext_current_trace: np.ndarray
    total_current_trace: np.ndarray


def validate_runtime(runtime: Any) -> None:
    required = ("config", "external_input", "weights", "model")
    for key in required:
        if not hasattr(runtime, key):
            raise ValueError(f"runtime is missing required field '{key}'")


def _device_of(runtime: Any) -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch backend requires torch to be installed") from exc
    if getattr(runtime, "device", None):
        return torch.device(str(runtime.device))
    return torch.device("cpu")


def _build_heterogeneity_arrays(cfg: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_e = int(cfg.N_E)
    n_i = int(cfg.N_I)
    n_total = n_e + n_i
    rng = np.random.RandomState(int(cfg.seed))

    v_th = np.full(n_total, float(cfg.V_th), dtype=np.float32)
    if float(cfg.V_th_heterogeneity_sd) > 0.0:
        v_th_het_e = rng.normal(0.0, float(cfg.V_th_heterogeneity_sd), n_e)
        v_th_het_i = rng.normal(0.0, float(cfg.V_th_heterogeneity_sd), n_i)
        v_th += np.concatenate([v_th_het_e, v_th_het_i]).astype(np.float32, copy=False)

    g_l = np.concatenate(
        [np.full(n_e, float(cfg.g_L_E), dtype=np.float32), np.full(n_i, float(cfg.g_L_I), dtype=np.float32)]
    )
    if float(cfg.g_L_heterogeneity_sd) > 0.0:
        g_l_het_e = rng.normal(0.0, float(cfg.g_L_heterogeneity_sd), n_e)
        g_l_het_i = rng.normal(0.0, float(cfg.g_L_heterogeneity_sd), n_i)
        g_l *= (1.0 + np.concatenate([g_l_het_e, g_l_het_i]).astype(np.float32, copy=False))
        g_l = np.clip(g_l, 0.01, None)

    c_m = np.concatenate(
        [np.full(n_e, float(cfg.C_m_E), dtype=np.float32), np.full(n_i, float(cfg.C_m_I), dtype=np.float32)]
    )
    if float(cfg.C_m_heterogeneity_sd) > 0.0:
        c_m_het_e = rng.normal(0.0, float(cfg.C_m_heterogeneity_sd), n_e)
        c_m_het_i = rng.normal(0.0, float(cfg.C_m_heterogeneity_sd), n_i)
        c_m *= (1.0 + np.concatenate([c_m_het_e, c_m_het_i]).astype(np.float32, copy=False))
        c_m = np.clip(c_m, 0.01, None)

    t_ref = np.concatenate(
        [np.full(n_e, float(cfg.t_ref_E), dtype=np.float32), np.full(n_i, float(cfg.t_ref_I), dtype=np.float32)]
    )
    if float(cfg.t_ref_heterogeneity_sd) > 0.0:
        t_ref_het_e = rng.normal(0.0, float(cfg.t_ref_heterogeneity_sd), n_e)
        t_ref_het_i = rng.normal(0.0, float(cfg.t_ref_heterogeneity_sd), n_i)
        t_ref += np.concatenate([t_ref_het_e, t_ref_het_i]).astype(np.float32, copy=False)
        t_ref = np.clip(t_ref, 0.1, None)

    return v_th, g_l, c_m, t_ref


def _has_delay_overrides(weights: Any) -> bool:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch backend requires torch to be installed") from exc
    for name in ("D_ee", "D_ei", "D_ie", "D_ii"):
        delay_mat = getattr(weights, name, None)
        if delay_mat is None:
            continue
        if bool(torch.isfinite(delay_mat).any()):
            return True
    return False


def _delay_steps(ms: float, dt: float) -> int:
    return max(1, int(round(float(ms) / float(dt))))


def _build_outgoing_edge_lists(
    *,
    W: "torch.Tensor",
    D: "torch.Tensor | None",
    dt: float,
    default_delay_ms: float,
    target_offset: int,
    device: Any,
) -> tuple[list[Any], list[Any], list[Any], int]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch backend requires torch to be installed") from exc

    W_cpu = W.detach().cpu().numpy()
    D_cpu = D.detach().cpu().numpy() if D is not None else None
    n_pre = int(W_cpu.shape[1])
    targets: list[Any] = []
    weights: list[Any] = []
    delays: list[Any] = []
    max_delay_steps = 1

    for pre in range(n_pre):
        nz = np.nonzero(W_cpu[:, pre])[0]
        if nz.size == 0:
            targets.append(torch.zeros((0,), dtype=torch.long, device=device))
            weights.append(torch.zeros((0,), dtype=torch.float32, device=device))
            delays.append(torch.zeros((0,), dtype=torch.long, device=device))
            continue

        delay_steps = np.empty(nz.size, dtype=np.int64)
        for i, row in enumerate(nz):
            delay_ms = float(default_delay_ms)
            if D_cpu is not None:
                value = float(D_cpu[row, pre])
                if np.isfinite(value):
                    delay_ms = value
            steps = _delay_steps(delay_ms, dt)
            delay_steps[i] = steps
            if steps > max_delay_steps:
                max_delay_steps = steps

        targets.append(
            torch.as_tensor(nz + int(target_offset), dtype=torch.long, device=device)
        )
        weights.append(
            torch.as_tensor(W_cpu[nz, pre], dtype=torch.float32, device=device)
        )
        delays.append(
            torch.as_tensor(delay_steps, dtype=torch.long, device=device)
        )

    return targets, weights, delays, max_delay_steps


def build_initial_state(runtime: Any) -> SimulationState:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch backend requires torch to be installed") from exc

    cfg = runtime.config
    dev = _device_of(runtime)
    n_e = int(cfg.N_E)
    n_i = int(cfg.N_I)
    n_total = n_e + n_i
    num_steps = int(np.ceil(float(cfg.T) / float(cfg.dt)))
    dt = float(cfg.dt)

    decay_e = float(np.exp(-dt / float(cfg.tau_ampa)))
    decay_i = float(np.exp(-dt / float(cfg.tau_gaba)))

    V = torch.full((n_total,), float(cfg.V_init), dtype=torch.float32, device=dev)
    g_e = torch.zeros((n_total,), dtype=torch.float32, device=dev)
    g_i = torch.zeros((n_total,), dtype=torch.float32, device=dev)
    refractory_countdown = torch.zeros((n_total,), dtype=torch.int64, device=dev)

    v_th_np, g_l_np, c_m_np, t_ref_ms = _build_heterogeneity_arrays(cfg)
    V_th_arr = torch.as_tensor(v_th_np, dtype=torch.float32, device=dev)
    g_L_arr = torch.as_tensor(g_l_np, dtype=torch.float32, device=dev)
    C_m_arr = torch.as_tensor(c_m_np, dtype=torch.float32, device=dev)
    ref_steps_arr = torch.as_tensor(
        np.maximum(np.round(t_ref_ms / dt).astype(np.int64), 1),
        dtype=torch.int64,
        device=dev,
    )

    use_delay_overrides = _has_delay_overrides(runtime.weights)

    delay_ei_steps = max(1, int(round(float(cfg.delay_ei) / dt)))
    delay_ie_steps = max(1, int(round(float(cfg.delay_ie) / dt)))
    delay_ee_steps = max(1, int(round(float(cfg.delay_ee) / dt)))
    delay_ii_steps = max(1, int(round(float(cfg.delay_ii) / dt)))
    ee_targets: list[Any] = []
    ee_weights: list[Any] = []
    ee_delays: list[Any] = []
    ei_targets: list[Any] = []
    ei_weights: list[Any] = []
    ei_delays: list[Any] = []
    ie_targets: list[Any] = []
    ie_weights: list[Any] = []
    ie_delays: list[Any] = []
    ii_targets: list[Any] = []
    ii_weights: list[Any] = []
    ii_delays: list[Any] = []
    buffer_g_e = None
    buffer_g_i = None
    buffer_e_to_i = None
    buffer_e_to_e = None
    buffer_i_to_e = None
    buffer_i_to_i = None

    if use_delay_overrides:
        ee_targets, ee_weights, ee_delays, max_ee = _build_outgoing_edge_lists(
            W=runtime.weights.W_ee,
            D=runtime.weights.D_ee,
            dt=dt,
            default_delay_ms=float(cfg.delay_ee),
            target_offset=0,
            device=dev,
        )
        ei_targets, ei_weights, ei_delays, max_ei = _build_outgoing_edge_lists(
            W=runtime.weights.W_ei,
            D=runtime.weights.D_ei,
            dt=dt,
            default_delay_ms=float(cfg.delay_ei),
            target_offset=n_e,
            device=dev,
        )
        ie_targets, ie_weights, ie_delays, max_ie = _build_outgoing_edge_lists(
            W=runtime.weights.W_ie,
            D=runtime.weights.D_ie,
            dt=dt,
            default_delay_ms=float(cfg.delay_ie),
            target_offset=0,
            device=dev,
        )
        ii_targets, ii_weights, ii_delays, max_ii = _build_outgoing_edge_lists(
            W=runtime.weights.W_ii,
            D=runtime.weights.D_ii,
            dt=dt,
            default_delay_ms=float(cfg.delay_ii),
            target_offset=n_e,
            device=dev,
        )
        buf_len = max(max_ee, max_ei, max_ie, max_ii) + 1
        buffer_g_e = torch.zeros((buf_len, n_total), dtype=torch.float32, device=dev)
        buffer_g_i = torch.zeros((buf_len, n_total), dtype=torch.float32, device=dev)
    else:
        buf_len = max(delay_ei_steps, delay_ie_steps, delay_ee_steps, delay_ii_steps) + 1
        buffer_e_to_i = torch.zeros((buf_len, n_e), dtype=torch.float32, device=dev)
        buffer_e_to_e = torch.zeros((buf_len, n_e), dtype=torch.float32, device=dev)
        buffer_i_to_e = torch.zeros((buf_len, n_i), dtype=torch.float32, device=dev)
        buffer_i_to_i = torch.zeros((buf_len, n_i), dtype=torch.float32, device=dev)

    ext = runtime.external_input
    if ext.ndim == 1:
        ext = ext[:, None].repeat(1, n_total)
    ext = ext.to(device=dev, dtype=torch.float32)

    return SimulationState(
        N=n_total,
        N_E=n_e,
        N_I=n_i,
        num_steps=num_steps,
        dt=dt,
        decay_e=decay_e,
        decay_i=decay_i,
        V=V,
        g_e=g_e,
        g_i=g_i,
        refractory_countdown=refractory_countdown,
        V_th_arr=V_th_arr,
        g_L_arr=g_L_arr,
        C_m_arr=C_m_arr,
        ref_steps_arr=ref_steps_arr,
        external_input=ext,
        delay_ei_steps=delay_ei_steps,
        delay_ie_steps=delay_ie_steps,
        delay_ee_steps=delay_ee_steps,
        delay_ii_steps=delay_ii_steps,
        buf_len=buf_len,
        buffer_e_to_i=buffer_e_to_i,
        buffer_e_to_e=buffer_e_to_e,
        buffer_i_to_e=buffer_i_to_e,
        buffer_i_to_i=buffer_i_to_i,
        use_delay_overrides=use_delay_overrides,
        buffer_g_e=buffer_g_e,
        buffer_g_i=buffer_g_i,
        ee_targets=ee_targets,
        ee_weights=ee_weights,
        ee_delays=ee_delays,
        ei_targets=ei_targets,
        ei_weights=ei_weights,
        ei_delays=ei_delays,
        ie_targets=ie_targets,
        ie_weights=ie_weights,
        ie_delays=ie_delays,
        ii_targets=ii_targets,
        ii_weights=ii_weights,
        ii_delays=ii_delays,
        voltage_trace=np.zeros((num_steps, n_total), dtype=np.float32),
        syn_current_trace=np.zeros((num_steps, n_total), dtype=np.float32),
        ext_current_trace=np.zeros((num_steps, n_total), dtype=np.float32),
        total_current_trace=np.zeros((num_steps, n_total), dtype=np.float32),
    )


def prepare_runtime_tensors(runtime: Any) -> SimulationState:
    validate_runtime(runtime)
    return build_initial_state(runtime)


def apply_delayed_events(state: SimulationState, runtime: Any) -> None:
    assert state.g_e is not None and state.g_i is not None
    if state.use_delay_overrides:
        assert state.buffer_g_e is not None and state.buffer_g_i is not None
        state.g_e += state.buffer_g_e[state.buf_idx]
        state.g_i += state.buffer_g_i[state.buf_idx]
        state.buffer_g_e[state.buf_idx].zero_()
        state.buffer_g_i[state.buf_idx].zero_()
        return
    assert state.buffer_e_to_e is not None and state.buffer_e_to_i is not None
    assert state.buffer_i_to_e is not None and state.buffer_i_to_i is not None
    w = runtime.weights

    spikes_ee = state.buffer_e_to_e[state.buf_idx]
    spikes_ei = state.buffer_e_to_i[state.buf_idx]
    spikes_ie = state.buffer_i_to_e[state.buf_idx]
    spikes_ii = state.buffer_i_to_i[state.buf_idx]

    # E->* contributes excitatory conductance g_e
    if state.N_E > 0:
        state.g_e[: state.N_E] = state.g_e[: state.N_E] + (w.W_ee @ spikes_ee)
        state.g_e[state.N_E :] = state.g_e[state.N_E :] + (w.W_ei @ spikes_ei)
    # I->* contributes inhibitory conductance g_i
    if state.N_I > 0:
        state.g_i[: state.N_E] = state.g_i[: state.N_E] + (w.W_ie @ spikes_ie)
        state.g_i[state.N_E :] = state.g_i[state.N_E :] + (w.W_ii @ spikes_ii)

    state.buffer_e_to_e[state.buf_idx].zero_()
    state.buffer_e_to_i[state.buf_idx].zero_()
    state.buffer_i_to_e[state.buf_idx].zero_()
    state.buffer_i_to_i[state.buf_idx].zero_()


def emit_and_schedule_spikes(state: SimulationState, spiked: "torch.Tensor") -> None:
    if state.use_delay_overrides:
        assert state.buffer_g_e is not None and state.buffer_g_i is not None
        spiked_e_idx = spiked[: state.N_E].nonzero(as_tuple=False).flatten().tolist()
        spiked_i_idx = spiked[state.N_E :].nonzero(as_tuple=False).flatten().tolist()

        for pre in spiked_e_idx:
            ee_t = state.ee_targets[pre]
            if ee_t.numel() > 0:
                ee_slots = (state.buf_idx + state.ee_delays[pre]) % state.buf_len
                state.buffer_g_e.index_put_(
                    (ee_slots, ee_t),
                    state.ee_weights[pre],
                    accumulate=True,
                )
            ei_t = state.ei_targets[pre]
            if ei_t.numel() > 0:
                ei_slots = (state.buf_idx + state.ei_delays[pre]) % state.buf_len
                state.buffer_g_e.index_put_(
                    (ei_slots, ei_t),
                    state.ei_weights[pre],
                    accumulate=True,
                )

        for pre in spiked_i_idx:
            ie_t = state.ie_targets[pre]
            if ie_t.numel() > 0:
                ie_slots = (state.buf_idx + state.ie_delays[pre]) % state.buf_len
                state.buffer_g_i.index_put_(
                    (ie_slots, ie_t),
                    state.ie_weights[pre],
                    accumulate=True,
                )
            ii_t = state.ii_targets[pre]
            if ii_t.numel() > 0:
                ii_slots = (state.buf_idx + state.ii_delays[pre]) % state.buf_len
                state.buffer_g_i.index_put_(
                    (ii_slots, ii_t),
                    state.ii_weights[pre],
                    accumulate=True,
                )
        return

    assert state.buffer_e_to_e is not None and state.buffer_e_to_i is not None
    assert state.buffer_i_to_e is not None and state.buffer_i_to_i is not None

    spiked_e = spiked[: state.N_E].to(dtype=state.buffer_e_to_e.dtype)
    spiked_i = spiked[state.N_E :].to(dtype=state.buffer_i_to_e.dtype)
    if bool(spiked_e.any()):
        tgt_ei = (state.buf_idx + state.delay_ei_steps) % state.buf_len
        tgt_ee = (state.buf_idx + state.delay_ee_steps) % state.buf_len
        state.buffer_e_to_i[tgt_ei] = spiked_e
        state.buffer_e_to_e[tgt_ee] = spiked_e
    if bool(spiked_i.any()):
        tgt_ie = (state.buf_idx + state.delay_ie_steps) % state.buf_len
        tgt_ii = (state.buf_idx + state.delay_ii_steps) % state.buf_len
        state.buffer_i_to_e[tgt_ie] = spiked_i
        state.buffer_i_to_i[tgt_ii] = spiked_i


def record_step(state: SimulationState, spiked: "torch.Tensor", t_ms: float) -> None:
    idxs = np.asarray(spiked.detach().cpu().nonzero(as_tuple=False).flatten().tolist(), dtype=int)
    for idx in idxs:
        state.spike_times.append(float(t_ms))
        state.spike_ids.append(int(idx))
        state.spike_types.append(0 if int(idx) < state.N_E else 1)


def integrate_step(state: SimulationState, runtime: Any, *, step: int) -> "torch.Tensor":
    assert state.V is not None and state.g_e is not None and state.g_i is not None
    assert state.refractory_countdown is not None
    assert state.external_input is not None
    assert state.C_m_arr is not None and state.g_L_arr is not None and state.V_th_arr is not None
    assert state.ref_steps_arr is not None

    apply_delayed_events(state, runtime)

    state.g_e *= state.decay_e
    state.g_i *= state.decay_i
    state.refractory_countdown = (state.refractory_countdown - 1).clamp(min=0)
    can_spike = state.refractory_countdown == 0
    I_ext = state.external_input[step]
    I_syn = state.g_e * (float(runtime.config.E_e) - state.V) + state.g_i * (
        float(runtime.config.E_i) - state.V
    )
    I_total = I_syn + I_ext

    # Persist full neuron traces for downstream analysis/plotting.
    assert state.voltage_trace is not None
    assert state.syn_current_trace is not None
    assert state.ext_current_trace is not None
    assert state.total_current_trace is not None
    state.syn_current_trace[step, :] = I_syn.detach().cpu().numpy()
    state.ext_current_trace[step, :] = I_ext.detach().cpu().numpy()
    state.total_current_trace[step, :] = I_total.detach().cpu().numpy()

    V_new, spiked = runtime.model(
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
    )
    state.V = V_new
    state.voltage_trace[step, :] = V_new.detach().cpu().numpy()
    if bool(spiked.any()):
        state.refractory_countdown = state.refractory_countdown.clone()
        state.refractory_countdown[spiked] = state.ref_steps_arr[spiked]
    return spiked


def _step_once(state: SimulationState, runtime: Any, *, step: int) -> None:
    t_ms = step * state.dt
    spiked = integrate_step(state, runtime, step=step)
    record_step(state, spiked, t_ms)
    emit_and_schedule_spikes(state, spiked)

    state.buf_idx = (state.buf_idx + 1) % state.buf_len
    state.step = step
    state.t_ms = t_ms


def simulate_network(runtime: Any, *, max_spikes: int | None = None) -> SimulationResult:
    state = prepare_runtime_tensors(runtime)
    for step in range(state.num_steps):
        if max_spikes is not None and len(state.spike_times) >= int(max_spikes):
            break
        _step_once(state, runtime, step=step)

    spikes = Spikes(
        times=np.asarray(state.spike_times, dtype=float),
        ids=np.asarray(state.spike_ids, dtype=int),
        types=np.asarray(state.spike_types, dtype=int),
    )
    return SimulationResult(
        spikes=spikes,
        t_ms=np.arange(state.num_steps, dtype=np.float32) * float(state.dt),
        voltage_trace=np.asarray(state.voltage_trace, dtype=np.float32),
        syn_current_trace=np.asarray(state.syn_current_trace, dtype=np.float32),
        ext_current_trace=np.asarray(state.ext_current_trace, dtype=np.float32),
        total_current_trace=np.asarray(state.total_current_trace, dtype=np.float32),
    )
