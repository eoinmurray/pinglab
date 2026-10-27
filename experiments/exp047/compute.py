"""Simulate each unique pool-size control; never aggregate, draw or publish.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import torch
from experiments.exp047 import recipe
from experiments.exp047.custom import evidence, inputs
from experiments.exp047.custom.network_artifacts import model_execution
from pingstore.library import PingstoreError, write_json_atomic
from snnlab.lang import duration_steps
from snnlab.sim import GraphExecutor, plan_graph, resolve_device


def initial_draws(training, *, ei_strength=None):
    """Ordered CPU weight draws, including unused zero recurrence.

    Return the generator cursor needed for CPU snapshots and the fresh loop
    needed for transfer probes. No model is rebuilt to recover these draws.
    """
    generator = torch.Generator().manual_seed(training["seed"])
    n_in, n_e, n_i, n_out = (
        training[k] for k in ("n_in", "n_hidden", "n_inh", "n_out")
    )

    def normal(shape, mean, sd, zero=0, scale=True):
        value = (
            torch.randn(*shape, generator=generator).mul_(sd).add_(mean).clamp_(min=0)
        )
        if zero:
            value *= (torch.rand(*shape, generator=generator) > zero).float()
            value /= 1 - zero
        return value / shape[0] if scale else value

    mean, sd = training["w_in"]
    external = normal((n_in, n_e), mean, sd, training["w_in_initial_zero_fraction"])
    if training.get("readout_w_init_mean") is not None:
        readout = normal(
            (n_e, n_out),
            training["readout_w_init_mean"],
            training["readout_w_init_std"],
            scale=False,
        )
    else:
        readout = normal((n_e, n_out), 5.1, 3.8) * training.get(
            "readout_w_out_scale", 225.0
        )
    normal((n_e, n_e), 0, 0)
    strength = training["ei_strength"] if ei_strength is None else ei_strength
    ei = normal((n_e, n_i), strength, strength * 0.1)
    ie_strength = strength * training["ei_ratio"]
    ie = normal((n_i, n_e), ie_strength, ie_strength * 0.1)
    normal((n_i, n_i), 0, 0)
    return generator, {
        "input_to_E.weight": external,
        "readout_projection.weight": readout,
        "E_to_I.weight": ei,
        "I_to_E.weight": ie,
    }


def simulate_probe(cfg, item, artifacts):
    settings = recipe.network_settings(cfg, item)
    bundle = recipe.author_network(cfg, item)
    model = GraphExecutor(plan_graph(bundle.graph), seed=item["seed"])
    _, values = initial_draws(settings)
    with torch.no_grad():
        for name, parameter in model.parameter_map().items():
            parameter.copy_(values[name])
    device = resolve_device(os.environ.get("PINGLAB_DEVICE", "auto"))
    model.to(device).eval()
    steps = duration_steps(cfg["t_ms"], cfg["dt_ms"])
    generator = torch.Generator().manual_seed(item["seed"] + 1)
    drive = (
        (
            torch.rand(steps, cfg["n_batch"], cfg["n_in"], generator=generator)
            < cfg["input_rate_hz"] * cfg["dt_ms"] / 1000
        )
        .float()
        .to(device)
    )
    with (
        model_execution(
            model,
            bundle,
            artifacts,
            seed=item["seed"],
            settings={"recipe": cfg, "job": item},
        ),
        torch.inference_mode(),
    ):
        result = model({"drive": drive}, diagnostics=False)
    rates = {
        key: float(result.outputs[f"spk_{label}_count"].sum())
        / (cfg["n_batch"] * size * steps * cfg["dt_ms"] / 1000)
        for label, key, size in (
            ("e", "hid", cfg["n_e"]),
            ("i", "inh", item["n_i"]),
        )
    }
    config = {
        "dt": cfg["dt_ms"],
        "t_ms": cfg["t_ms"],
        "n_in": cfg["n_in"],
        "n_hidden": cfg["n_e"],
        "n_inh": item["n_i"],
        "ei_strength": cfg["g_ei_total"],
        "ei_ratio": item["g_ie_total"] / cfg["g_ei_total"],
        "input_rate_hz": cfg["input_rate_hz"],
        "n_batch": cfg["n_batch"],
        "load_weights": None,
        **recipe.duration_configuration(cfg["t_ms"], cfg["dt_ms"]),
        **recipe.refractory_execution_configuration(cfg["dt_ms"]),
    }
    metrics = {
        "mode": "probe",
        "model": "ping",
        "config": config,
        "rates_hz": rates,
        "rate_e_hz": rates["hid"],
        "rate_i_hz": rates["inh"],
    }
    return metrics, {
        "job": item,
        "graph_digest": bundle.manifest["graph_digest"],
        "device": device,
    }


def compute(*, run_id=None):
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    with inputs.execution(
        REPO, "compute", sources={}, run_id=run_id, configuration=cfg
    ) as run:
        environment = {"PINGLAB_SMOKE": "1" if cfg["profile"] == "smoke" else "0"}
        run.record["execution"]["environment"] = environment
        for item in recipe.jobs(cfg):
            metrics, metadata = simulate_probe(
                cfg, item, run.export / f"{item['id']}--simulation"
            )
            evidence.metric(metrics, cfg, item)
            write_json_atomic(
                run.export / f"probe--{item['id']}--metrics.json", metrics
            )
            run.record["execution"].setdefault("trials", []).append(metadata)
        evidence.rows(run.export, cfg)
        write_json_atomic(
            run.export / "evidence.json",
            {"schema": "exp047.compute/v1", "recipe": cfg, "jobs": recipe.jobs(cfg)},
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="unused v4 identity reserved before dispatch")
    args = parser.parse_args()
    try:
        compute(run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"exp047 compute: {exc}\n")


if __name__ == "__main__":
    main()
