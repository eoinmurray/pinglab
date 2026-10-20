"""Record the fixed-batch teaching trace through the native spiking graph.

The loop retains losses, actual gradients, weights and two illustrative responses.
Its native end snapshot records the final teaching parameters; there is no
validation checkpoint in this deliberately separate fixed-batch protocol.
"""

import numpy as np
import torch
from snnlab.sim import GraphExecutor, plan_graph
from .network_artifacts import model_execution


def training_trace(bundle, bundle_path, values, artifacts, cfg):
    """Record a separate fixed-batch teaching loop through the native SNNLab graph."""
    spikes, labels, _, _ = values
    model = GraphExecutor(
        plan_graph(bundle.graph),
        seed=cfg["seed"],
        trainable_parameters=bundle.training["resolved_parameters"]["trainable"],
        surrogate=bundle.training["surrogate"],
        compile_mode=None,
    )
    parameter = next(iter(model.parameter_map().values()))
    optimizer = torch.optim.AdamW([parameter], lr=cfg["learning_rate"], weight_decay=0)
    inputs, targets = torch.from_numpy(spikes), torch.from_numpy(labels)
    indices = [int(np.flatnonzero(labels == cls)[0]) for cls in (0, 1)]
    weights = [parameter.detach().numpy().copy()]
    losses, gradients, outputs, voltages = [], [], [], []
    with model_execution(
        model,
        bundle,
        artifacts,
        kind="train",
        seed=cfg["seed"],
        settings=cfg,
        snapshot_stages=("end",),
    ):
        for _ in range(cfg["trace_updates"]):
            optimizer.zero_grad(set_to_none=True)
            result = model({"inputs": inputs}, diagnostics=True)
            loss = torch.nn.functional.cross_entropy(result.outputs["scores"], targets)
            losses.append(float(loss.detach()))
            outputs.append(
                result.diagnostics["output_spikes"][:, indices].detach().numpy()
            )
            voltages.append(
                result.diagnostics["output_voltage"][:, indices].detach().numpy()
            )
            loss.backward()
            gradients.append(parameter.grad.detach().numpy().copy())
            torch.nn.utils.clip_grad_norm_([parameter], cfg["gradient_clip"])
            optimizer.step()
            model.enforce_constraints()
            weights.append(parameter.detach().numpy().copy())
    return dict(
        losses=np.asarray(losses),
        weights=np.asarray(weights),
        gradients=np.asarray(gradients),
        outputs=np.asarray(outputs),
        voltages=np.asarray(voltages),
        indices=np.asarray(indices),
    )
