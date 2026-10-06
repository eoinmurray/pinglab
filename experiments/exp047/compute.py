"""Simulate each unique pool-size control; never aggregate, draw or publish."""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import torch
from experiments.exp047 import evidence, inputs, recipe
from experiments.helpers.checkpoint_graph import initial_draws
from pingstore.contracts import PingstoreError, write_json_atomic
from snnlab.sim.execution import GraphExecutor, plan_graph, resolve_device
from snnlab.sim.timing import duration_steps


def simulate_probe(cfg, item):
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
    with torch.inference_mode():
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
    return metrics, {'job':item,'graph_digest':bundle.manifest['graph_digest'],'device':device}


def compute(*, run_id=None):
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    with inputs.execution(
        REPO, "compute", sources={}, run_id=run_id, configuration=cfg
    ) as run:
        environment = {"PINGLAB_SMOKE": "1" if cfg["profile"] == "smoke" else "0"}
        run.record["execution"]["environment"] = environment
        for item in recipe.jobs(cfg):
            metrics, metadata = simulate_probe(cfg, item)
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
