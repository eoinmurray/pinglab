"""Exp142 event-queued continuous COBA-LIF inference with a COBA readout.

Internal units: ms, mV, uS, nF. All population voltages use adaptive DOP853;
receptors and refractory boundaries are analytic. The recording clock never
constrains integration steps.

The simulation stage and numerical tests supply explicit recipe physics and
receive recordings and diagnostics. There is no training or run I/O."""

from __future__ import annotations

import heapq
from time import perf_counter

import numpy as np
import scipy
from scipy.integrate import solve_ivp

NS_PER_MS = 1_000_000
WEIGHT_KEYS = (
    "input_to_E.weight",
    "E_to_I.weight",
    "I_to_E.weight",
    "readout_projection.weight",
)


def _threshold_slope_boundary(ge, gi, leak, active, horizon, p):
    """First positive-to-negative threshold-slope zero, relative to now.

    F(s) = A exp(-s/tauE) - B exp(-s/tauI) - C. With A,B,C >= 0
    and tauE <= tauI, F(s)*exp(s/tauE) is nonincreasing. Thus F(0)<=0
    precludes an upward threshold crossing, while F(0)>0 has at most one
    positive zero. This is NOT the excitatory-only critical-g formula.
    """
    te, ti = p["tau_ampa_ms"], p["tau_gaba_ms"]
    a = ge * (p["excitatory_reversal_mv"] - p["threshold_mv"])
    b = gi * (p["threshold_mv"] - p["inhibitory_reversal_mv"])
    c = leak * (p["threshold_mv"] - p["resting_mv"])
    mask = active & (a - b - c > 0)
    mask &= a * np.exp(-horizon / te) - b * np.exp(-horizon / ti) - c < 0
    if not np.any(mask):
        return horizon
    a, b, c = a[mask], b[mask], c[mask]
    lo, hi = np.zeros(a.size), np.full(a.size, horizon)
    for _ in range(54):
        mid = (lo + hi) / 2
        positive = a * np.exp(-mid / te) - b * np.exp(-mid / ti) - c > 0
        lo = np.where(positive, mid, lo)
        hi = np.where(positive, hi, mid)
    return float(np.min(hi))


def _ordered_input_events(*, events, duration, nin):
    """Validate and order integer event coordinates without rescaling weights."""
    arrays = [np.asarray(events[key]) for key in ("time_ns", "batches", "channels")]
    if (
        any(x.ndim != 1 or not np.issubdtype(x.dtype, np.integer) for x in arrays)
        or len({x.size for x in arrays}) != 1
    ):
        raise ValueError(
            "event coordinates must be equal-length one-dimensional integer arrays"
        )
    time_ns, batches, channels = arrays
    if (
        np.any(time_ns < 0)
        or np.any(time_ns >= duration * NS_PER_MS)
        or np.any(batches != 0)
        or np.any(channels < 0)
        or np.any(channels >= nin)
    ):
        raise ValueError("events outside half-open time, batch or channel bounds")
    order = np.lexsort((channels, time_ns))
    input_times = time_ns[order].astype(np.float64) / NS_PER_MS
    channels = channels[order]
    return input_times, channels


def _promoted_weight_arrays(*, weights):
    """Validate float32 pathway shapes and promote their exact stored values."""
    if set(weights) != set(WEIGHT_KEYS):
        raise ValueError(f"weights must have exactly keys {WEIGHT_KEYS}")
    w = [np.asarray(weights[key]) for key in WEIGHT_KEYS]
    if any(
        x.ndim != 2
        or x.dtype != np.float32
        or not np.isfinite(x).all()
        or np.any(x < 0)
        for x in w
    ):
        raise ValueError("weights must be finite nonnegative float32 matrices")
    nin, ne = w[0].shape
    ni, nout = w[1].shape[1], w[3].shape[1]
    if (
        nin < 1
        or ne < 1
        or nout < 1
        or w[1].shape != (ne, ni)
        or w[2].shape != (ni, ne)
        or w[3].shape != (ne, nout)
    ):
        raise ValueError("incompatible weight shapes")
    win, wei, wie, wout = [x.astype(np.float64) for x in w]
    return nin, ne, ni, nout, win, wei, wie, wout


def _physical_parameters(*, defaults, parameters):
    """Resolve physical overrides and validate COBA threshold and reversal ordering."""
    p = dict(defaults or {})
    if not p:
        raise ValueError("explicit physical defaults are required")
    if parameters is not None:
        if set(parameters) - set(p):
            raise ValueError("unknown physics parameter")
        p.update(parameters)
    p = {key: float(value) for key, value in p.items()}
    if not all(np.isfinite(value) for value in p.values()):
        raise ValueError("physics must be finite")
    positive = (
        "capacitance_e_nf",
        "capacitance_i_nf",
        "tau_ampa_ms",
        "tau_gaba_ms",
        "refractory_e_ms",
        "refractory_i_ms",
        "recurrent_delay_ms",
    )
    if any(p[key] <= 0 for key in positive):
        raise ValueError(
            "capacitances, time constants, thresholds and delays must be positive"
        )
    if (
        min(p["leak_e_us"], p["leak_i_us"]) < 0
        or p["tau_ampa_ms"] > p["tau_gaba_ms"]
        or not p["inhibitory_reversal_mv"]
        <= p["resting_mv"]
        < p["threshold_mv"]
        < p["excitatory_reversal_mv"]
        or p["reset_mv"] >= p["threshold_mv"]
    ):
        raise ValueError("invalid COBA threshold-slope/reversal parameter ordering")
    return p


def _segment_callbacks(
    *, origin, initial_ge, initial_gi, active, leak, p, cap, tau_e, tau_i, threshold
):
    """Bind the fixed initial conditions for one continuous integration segment."""

    def derivative(time, state):
        s = time - origin
        e, i = initial_ge * np.exp(-s / tau_e), initial_gi * np.exp(-s / tau_i)
        result = (
            leak * (p["resting_mv"] - state)
            + e * (p["excitatory_reversal_mv"] - state)
            + i * (p["inhibitory_reversal_mv"] - state)
        ) / cap
        result[~active] = 0
        return result

    # One global earliest root, rather than one Python callback per cell.
    def crossing(_time, state):
        return float(np.max(state[active] - threshold)) if np.any(active) else -1.0

    crossing.terminal = True
    crossing.direction = 1
    return derivative, crossing


def _constant_voltage(voltage):
    """Return the resting segment interpolator with its voltage array bound."""

    def dense(times):
        return np.broadcast_to(voltage[:, None], (len(voltage), np.size(times)))

    return dense


def _schedule_recurrent_arrivals(
    *, now, p, duration, hit, ne, ni, wei, wie, heap, sequence
):
    """Queue delayed recurrent conductance jumps in their original order."""
    arrival = now + p["recurrent_delay_ms"]
    if arrival <= now:
        raise RuntimeError("recurrent delay cannot advance floating-point time")
    if arrival < duration:
        e, i = hit[hit < ne], hit[(hit >= ne) & (hit < ne + ni)] - ne
        for kind, kick in (
            ("exc", wei[e].sum(axis=0)),
            ("inh", wie[i].sum(axis=0)),
        ):
            if np.any(kick):
                heapq.heappush(heap, (arrival, sequence, kind, kick))
                sequence += 1
    return sequence


def simulate_continuous(
    events,
    weights,
    *,
    duration_ms=200.0,
    sample_dt_ms=0.1,
    parameters=None,
    defaults=None,
    rtol=1e-8,
    atol=1e-9,
    max_step_ms=0.05,
):
    """Return ``(recording, diagnostics)`` for one arbitrary-size E/I network.

    ``defaults`` must explicitly supply the physical parameter dictionary.
    The output population joins the adaptive voltage system, has E-cell
    physics, receives zero-delay full-weight AMPA jumps at hidden spikes,
    and emits physical threshold roots with hard resets. Global IDs are
    E, I, then output. Scores are output counts; ties are not resolved here.

    Weights must be finite nonnegative float32 matrices under WEIGHT_KEYS,
    shaped (nin,ne), (ne,ni), (ni,ne), (ne,nout); ni=0 is supported. Promote
    the actual float32 values directly, without decimal reconstruction/rescaling.
    Event integer arrays ``time_ns``, ``batches`` (all zero), ``channels`` are
    unordered coordinates in the half-open interval [0,duration); duplicates
    are additive. Parameters partially override defaults (unknown keys
    fail). Initial voltages are resting_mv. Receptor weights are uS.

    E/I spikes have unrounded float64 ns times and global IDs E first, I+ne.
    Excitation E->I and inhibition I->E arrive after recurrent_delay_ms (0.1
    by default), independent of sampling and max_step. Coincident arrivals are
    batched before evolving. Refractory voltage is exactly reset_mv on
    [spike,release), while conductance continues to decay and accept arrivals.

    g_time_ns: k*sample_dt < duration; v_time_ns: (k+1)*sample_dt <= duration.
    g_exc_ns/g_inh_ns are nS (samples,ne+ni+nout), right-continuous at jumps.
    voltage_mv is dense-segment sampled, post-reset at roots, never interpolated
    across a reset. Readout voltage is post-reset; readout_scores and class_scores count
    output spikes at each voltage sample and over the full duration, respectively.
    """
    started = perf_counter()
    duration, dt = float(duration_ms), float(sample_dt_ms)
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration_ms must be finite and nonnegative")
    if any(not np.isfinite(x) or x <= 0 for x in (dt, rtol, atol, max_step_ms)):
        raise ValueError(
            "sample_dt_ms, tolerances and max_step_ms must be positive finite"
        )
    p = _physical_parameters(defaults=defaults, parameters=parameters)
    nin, ne, ni, nout, win, wei, wie, wout = _promoted_weight_arrays(weights=weights)
    input_times, channels = _ordered_input_events(
        events=events, duration=duration, nin=nin
    )
    # Build labels first to avoid dropping a nominally exact endpoint in ms.
    count = int(np.ceil(duration / dt))
    labels = np.arange(count + 1, dtype=np.float64) * (dt * NS_PER_MS)
    glabel = labels[labels < duration * NS_PER_MS]
    vlabel = labels[1:][labels[1:] <= duration * NS_PER_MS]
    if np.all(labels == np.rint(labels)) and (not labels.size or labels[-1] < 2**63):
        glabel, vlabel = glabel.astype(np.int64), vlabel.astype(np.int64)
    gt, vt = glabel / NS_PER_MS, vlabel / NS_PER_MS
    n = ne + ni + (nout)
    voltage = np.empty((vt.size, n))
    gexc, ginh = np.empty((gt.size, n)), np.empty((gt.size, n))
    ge, gi = np.zeros(n), np.zeros(n)
    v = np.full(n, p["resting_mv"])
    release = np.full(n, -np.inf)
    cap = np.r_[
        np.full(ne, p["capacitance_e_nf"]),
        np.full(ni, p["capacitance_i_nf"]),
        np.full(nout, p["capacitance_e_nf"]),
    ]
    leak = np.r_[
        np.full(ne, p["leak_e_us"]),
        np.full(ni, p["leak_i_us"]),
        np.full(nout, p["leak_e_us"]),
    ]
    ref = np.r_[
        np.full(ne, p["refractory_e_ms"]),
        np.full(ni, p["refractory_i_ms"]),
        np.full(nout, p["refractory_e_ms"]),
    ]
    tau_e, tau_i = p["tau_ampa_ms"], p["tau_gaba_ms"]
    threshold, reset = p["threshold_mv"], p["reset_mv"]
    stats = {
        "solve_calls": 0,
        "nfev": 0,
        "accepted_steps": 0,
        "threshold_slope_splits": 0,
        "arrival_batches": 0,
        "recurrent_arrival_batches": 0,
        "max_root_residual_mv": 0.0,
    }
    heap, sequence = [], 0
    spike_times, spike_ids = [], []
    now, inp, gs, vs = 0.0, 0, 0, 0
    while now < duration:
        # Every boundary starts with all simultaneous receptor jumps.
        had_arrival = False
        if inp < input_times.size and input_times[inp] == now:
            stop = int(np.searchsorted(input_times, now, side="right"))
            ge[:ne] += win[channels[inp:stop]].sum(axis=0)
            inp, had_arrival = stop, True
        while heap and heap[0][0] == now:
            _, _, kind, kick = heapq.heappop(heap)
            if kind == "exc":
                ge[ne : ne + ni] += kick
            else:
                gi[:ne] += kick
            stats["recurrent_arrival_batches"] += 1
            had_arrival = True
        stats["arrival_batches"] += int(had_arrival)
        active = release <= now
        v[~active] = reset
        end = min(
            duration,
            float(input_times[inp]) if inp < input_times.size else duration,
            heap[0][0] if heap else duration,
            float(np.min(release[~active])) if np.any(~active) else duration,
        )
        critical = _threshold_slope_boundary(ge, gi, leak, active, end - now, p)
        # Ignore only sub-ulp-scale repeat splits at an already reached F=0.
        progress_floor = 64 * np.spacing(max(1.0, abs(now), abs(end)))
        if critical > progress_floor and now + critical < end - progress_floor:
            end = now + critical
            stats["threshold_slope_splits"] += 1
        origin, initial_ge, initial_gi = now, ge.copy(), gi.copy()

        derivative, crossing = _segment_callbacks(
            origin=origin,
            initial_ge=initial_ge,
            initial_gi=initial_gi,
            active=active,
            leak=leak,
            p=p,
            cap=cap,
            tau_e=tau_e,
            tau_i=tau_i,
            threshold=threshold,
        )
        root = False
        if not np.any(ge) and not np.any(gi) and np.all(v[active] == p["resting_mv"]):
            dense = _constant_voltage(v)
        else:
            solution = solve_ivp(
                derivative,
                (now, end),
                v,
                method="DOP853",
                dense_output=True,
                events=crossing,
                rtol=rtol,
                atol=atol,
                max_step=max_step_ms,
            )
            if not solution.success:
                raise RuntimeError(f"continuous integration failed: {solution.message}")
            stats["solve_calls"] += 1
            stats["nfev"] += solution.nfev
            stats["accepted_steps"] += solution.t.size - 1
            end = float(solution.t[-1])
            dense = solution.sol
            root = bool(solution.t_events[0].size)
        if end <= now:
            raise RuntimeError(f"continuous solver made no progress at {now!r} ms")
        stop = int(np.searchsorted(vt, end, side="left"))
        if stop > vs:
            voltage[vs:stop] = dense(vt[vs:stop]).T
        vs = stop
        stop = int(np.searchsorted(gt, end, side="left"))
        elapsed = gt[gs:stop] - now
        gexc[gs:stop] = np.exp(-elapsed / tau_e)[:, None] * ge * 1000
        ginh[gs:stop] = np.exp(-elapsed / tau_i)[:, None] * gi * 1000
        gs = stop
        v = np.asarray(dense(np.asarray([end])))[:, 0].copy()
        ge *= np.exp(-(end - now) / tau_e)
        gi *= np.exp(-(end - now) / tau_i)
        now = end
        if root:
            # Only roundoff-equal voltages are simultaneous, not an rtol-wide
            # band that could spuriously synchronize merely nearby roots.
            time_resolution = 8 * np.spacing(max(1.0, abs(now)))
            tolerance = max(
                64 * np.finfo(float).eps * max(1.0, abs(threshold)),
                time_resolution
                * float(np.max(np.abs(derivative(now, v)[active]), initial=0)),
            )
            hit = np.flatnonzero(active & (np.abs(v - threshold) <= tolerance))
            if not hit.size:
                raise RuntimeError("terminal event has no threshold root")
            stats["max_root_residual_mv"] = max(
                stats["max_root_residual_mv"], float(np.max(np.abs(v[hit] - threshold)))
            )
            spike_times.extend([now] * hit.size)
            spike_ids.extend(hit.tolist())
            v[hit], release[hit] = reset, now + ref[hit]
            # Hidden roots trigger immediate AMPA jumps at the output cells;
            # their voltages remain continuous until a physical threshold root.
            e = hit[hit < ne]
            ge[ne + ni :] += wout[e].sum(axis=0)
            sequence = _schedule_recurrent_arrivals(
                now=now,
                p=p,
                duration=duration,
                hit=hit,
                ne=ne,
                ni=ni,
                wei=wei,
                wie=wie,
                heap=heap,
                sequence=sequence,
            )
    voltage[vs:] = v
    spike_ms = np.asarray(spike_times, dtype=np.float64)
    neurons = np.asarray(spike_ids, dtype=np.int64)
    recording = {
        "g_time_ns": glabel,
        "v_time_ns": vlabel,
        "g_exc_ns": gexc,
        "g_inh_ns": ginh,
        "voltage_mv": voltage,
        "spike_time_ns": spike_ms * NS_PER_MS,
        "spike_neurons": neurons,
    }
    selected = neurons >= ne + ni
    output_times = spike_ms[selected] * NS_PER_MS
    output_neurons = neurons[selected] - ne - ni
    scores = np.stack(
        [
            np.searchsorted(output_times[output_neurons == k], vlabel, side="right")
            for k in range(nout)
        ],
        axis=1,
    )
    recording.update(
        output_spike_time_ns=output_times,
        output_spike_neurons=output_neurons,
        readout_voltage=voltage[:, ne + ni :],
        readout_scores=scores,
        class_scores=np.bincount(output_neurons, minlength=nout),
    )
    diagnostics = {
        "solver": "scipy.integrate.solve_ivp",
        "method": "DOP853",
        "scipy_version": scipy.__version__,
        "solver_dtype": "float64",
        "source_weights_dtype": "float32",
        "rtol": float(rtol),
        "atol": float(atol),
        "max_step_ms": float(max_step_ms),
        "duration_ms": duration,
        "sample_dt_ms": dt,
        "parameters": p,
        "n_input": nin,
        "n_e": ne,
        "n_i": ni,
        "n_output": nout,
        "input_event_count": int(input_times.size),
        "spike_count": int(neurons.size),
        "e_spike_count": int(np.count_nonzero(neurons < ne)),
        "i_spike_count": int(np.count_nonzero((neurons >= ne) & (neurons < ne + ni))),
        "output_spike_count": int(recording["output_spike_neurons"].size),
        "root_detection": "global earliest upward max(V-threshold); mixed-receptor threshold-slope boundaries",
        "readout": (
            "COBA-LIF with AMPA jumps at hidden roots; physical refractory clamp; spike-count logits"
        ),
        "readout_pre_reset": "not defined for COBA readout",
        "recurrent_delay_ms": p["recurrent_delay_ms"],
        "training": False,
        **stats,
        "wall_seconds": perf_counter() - started,
    }
    return recording, diagnostics
