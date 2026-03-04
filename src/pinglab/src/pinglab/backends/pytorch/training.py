"""PyTorch-backend training helpers: device selection and batched SNN forward pass."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def get_device() -> str:
    """Return the best available accelerator device string (e.g. 'cuda', 'mps', 'cpu')."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch backend requires torch to be installed") from exc

    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    return "cpu"


def run_batch(
    runtime: object,
    images: "torch.Tensor",
    *,
    T_steps: int,
    n_total: int,
    n_input: int,
    out_start: int,
    out_stop: int,
    input_scale: "float | torch.Tensor",
    sim_state: object = None,
    burn_in_steps: int = 0,
    readout: str = "spike_count",
    readout_alpha: float = 0.01,
    encoding: str = "tonic",
    return_spikes: bool = False,
    bptt_steps: int | None = None,
    cm_backward_scale: float = 1.0,
    detach_spikes: bool = True,
) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
    """Run a batch of images through the SNN; return logits [B, C].

    If ``return_spikes`` is True, returns ``(logits, spikes)`` where
    spikes is ``[B, T, N_E]`` (the full E-neuron spike tensor).

    Encodes each image to [T, N] via the chosen encoding, stacks into
    [B, T, N] for a single batched simulation call.

    Args:
        runtime: Compiled network runtime (``compile_graph_to_runtime`` output).
        images: Image batch of shape [B, ...] (any shape that flattens to n_input).
        T_steps: Number of simulation timesteps per sample.
        n_total: Total number of E neurons in the network.
        n_input: Number of input neurons (first n_input neurons receive the image).
        out_start: Start index of the output population within the E neuron block.
        out_stop: End index (exclusive) of the output population.
        input_scale: Multiplier applied to pixel values before injection.
        sim_state: Pre-built :class:`SimulationState` (optional).  When provided
            its ``batch_size`` is used for padding; it is passed directly to
            :func:`simulate_network`.
        burn_in_steps: Number of initial timesteps to exclude from readout decoding.
            The full simulation (including burn-in) still runs so membrane state
            can settle; only the readout window is shortened.
        readout: Decoding strategy for producing logits from the simulation.
            ``"spike_count"`` (default): sum output spike counts over readout window.
            ``"voltage"``: mean membrane voltage of output neurons over readout window.
            ``"hybrid"``: spike counts + alpha * mean voltage.  Provides smooth
            gradient signal while encouraging spiking output.
        readout_alpha: Mixing weight for the voltage component in hybrid readout.
        bptt_steps: If set, truncated BPTT — detach simulation state every this many
            steps to limit gradient depth.
        encoding: Input encoding strategy.
            ``"tonic"`` (default): constant current injection (pixel * scale each step).
            ``"poisson"``: stochastic Poisson spike trains (Bernoulli per step).

    Returns:
        Logits tensor of shape [B, C] where C = out_stop - out_start.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch backend requires torch to be installed") from exc

    from functools import partial
    from pinglab.io.training import encode_rate_to_tonic, encode_poisson
    from pinglab.backends.pytorch.simulate_network import simulate_network
    from pinglab.backends.pytorch.surrogate import surrogate_lif_step

    # Bind cm_backward_scale into the surrogate spike function so the
    # simulate_network call site doesn't need to know about it.
    if cm_backward_scale != 1.0:
        spike_fn = partial(surrogate_lif_step, cm_backward_scale=cm_backward_scale)
    else:
        spike_fn = surrogate_lif_step

    if encoding == "poisson":
        encode_fn = encode_poisson
    else:
        encode_fn = encode_rate_to_tonic

    B_actual = images.shape[0]
    # Encode with unit amplitude, then apply scale.
    # This allows input_scale to be a learnable tensor (gradient flows through the multiply).
    ext_batch = torch.stack(
        [
            encode_fn(img, T_steps=T_steps, n_total=n_total, n_input=n_input, scale=1.0)
            for img in images
        ],
        dim=0,
    )  # [B_actual, T, N]
    if isinstance(input_scale, torch.Tensor):
        ext_batch = ext_batch.to(device=input_scale.device) * input_scale
    else:
        ext_batch = ext_batch * input_scale

    # If the pre-built state has a larger batch dim (e.g. last mini-batch),
    # pad with zeros so the state tensor shapes match.
    B_state = sim_state.batch_size if sim_state is not None else B_actual
    if B_actual < B_state:
        pad = torch.zeros(
            B_state - B_actual, ext_batch.shape[1], ext_batch.shape[2],
            dtype=ext_batch.dtype, device=ext_batch.device,
        )
        ext_batch = torch.cat([ext_batch, pad], dim=0)  # [B_state, T, N]

    need_voltage = readout in ("voltage", "hybrid")
    sim_returns = simulate_network(
        runtime,
        external_input=ext_batch,
        spike_fn=spike_fn,
        return_spike_tensor=True,
        return_voltage_tensor=need_voltage,
        state=sim_state,
        bptt_steps=bptt_steps,
        detach_spikes=detach_spikes,
    )

    if need_voltage:
        _result, spikes, voltages = sim_returns
    else:
        _result, spikes = sim_returns

    out_spikes = spikes[:B_actual, burn_in_steps:, out_start:out_stop]

    if readout == "voltage":
        logits = voltages[:B_actual, burn_in_steps:, out_start:out_stop].mean(dim=1)
    elif readout == "hybrid":
        spike_counts = out_spikes.sum(dim=1)
        mean_voltage = voltages[:B_actual, burn_in_steps:, out_start:out_stop].mean(dim=1)
        logits = spike_counts + readout_alpha * mean_voltage
    else:
        logits = out_spikes.sum(dim=1)

    if return_spikes:
        return logits, spikes[:B_actual]
    return logits
