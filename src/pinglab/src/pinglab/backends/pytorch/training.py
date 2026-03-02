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
    input_scale: float,
    sim_state: object = None,
    burn_in_steps: int = 0,
) -> "torch.Tensor":
    """Run a batch of images through the SNN; return logits [B, C].

    Encodes each image to [T, N] via rate coding, stacks into [B, T, N] for a
    single batched simulation call.  The spike tensor returned is [B, T, N_E];
    summing over T gives per-sample output spike counts used as logits.

    If the last mini-batch is smaller than the pre-built *sim_state* batch size,
    the input is zero-padded to match and the extra rows are discarded afterwards.

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
        burn_in_steps: Number of initial timesteps to exclude from spike-count
            decoding.  The full simulation (including burn-in) still runs so
            membrane state can settle; only the readout window is shortened.

    Returns:
        Logits tensor of shape [B, C] where C = out_stop - out_start.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError("PyTorch backend requires torch to be installed") from exc

    from pinglab.io.training import encode_rate
    from pinglab.backends.pytorch.simulate_network import simulate_network
    from pinglab.backends.pytorch.surrogate import surrogate_lif_step

    B_actual = images.shape[0]
    ext_batch = torch.stack(
        [
            encode_rate(img, T_steps=T_steps, n_total=n_total, n_input=n_input, scale=input_scale)
            for img in images
        ],
        dim=0,
    )  # [B_actual, T, N]

    # If the pre-built state has a larger batch dim (e.g. last mini-batch),
    # pad with zeros so the state tensor shapes match.
    B_state = sim_state.batch_size if sim_state is not None else B_actual
    if B_actual < B_state:
        pad = torch.zeros(
            B_state - B_actual, ext_batch.shape[1], ext_batch.shape[2],
            dtype=ext_batch.dtype,
        )
        ext_batch = torch.cat([ext_batch, pad], dim=0)  # [B_state, T, N]

    _, spikes = simulate_network(
        runtime,
        external_input=ext_batch,
        spike_fn=surrogate_lif_step,
        return_spike_tensor=True,
        state=sim_state,
    )
    # spikes: [B_state, T, N_E] — sum over readout window, slice to actual batch size
    logits = spikes[:B_actual, burn_in_steps:, out_start:out_stop].sum(dim=1)  # [B_actual, C]
    return logits
