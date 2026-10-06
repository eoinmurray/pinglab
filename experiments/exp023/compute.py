"""Execute two raster trials and fourteen f–I trials through native graphs."""

from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
import torch
from experiments.exp023 import recipe
from pingstore.contracts import PingstoreError
from pingstore.stages import stage_run
from snnlab.sim.execution import simulate
from snnlab.sim.timing import duration_steps


def recording(result, cfg, point, *, traces):
    """Adapt named graph tensors to the retained measurement interface."""
    steps = duration_steps(point["t_ms"], point["dt_ms"])
    data = {"dt": point["dt_ms"], "T": steps, "n_e": cfg["n_e"], "n_i": cfg["n_i"]}
    if not traces:
        for population in ("e", "i"):
            values = result.outputs[f"spk_{population}_count"].detach().cpu().numpy()
            if (
                values.shape != (1, cfg[f"n_{population}"])
                or not np.isfinite(values).all()
                or not np.equal(values, np.floor(values)).all()
                or (values < 0).any()
                or (values > steps).any()
            ):
                raise PingstoreError("invalid graph spike-count reduction")
            data[f"spk_{population}_count"] = np.int64(values.sum())
        return data
    signals = {
        name: tensor.detach().cpu().numpy()[:, 0]
        for name, tensor in result.diagnostics.items()
    }
    for population in ("e", "i"):
        spikes = signals[f"spk_{population}"]
        if (
            spikes.shape != (steps, cfg[f"n_{population}"])
            or not np.isin(spikes, (0, 1)).all()
        ):
            raise PingstoreError("invalid graph population spikes")
        data[f"spk_{population}"] = spikes.astype(bool)
        index = int(spikes.sum(axis=0).argmax())
        data[f"{population}_trace_index"] = index
        for signal in ("v", "ge", "gi") if population == "e" else ("v", "ge"):
            values = signals[f"{signal}_{population}"]
            if (
                values.shape != spikes.shape
                or not np.isfinite(values).all()
                or (signal.startswith("g") and (values < 0).any())
            ):
                raise PingstoreError("invalid graph voltage/conductance recording")
            data[f"{signal}_{population}_selected"] = values[:, index]
    data["has_gi_e"] = np.bool_(signals["gi_e"].any())
    return data


def compute(*, run_id: str | None = None) -> str:
    smoke = os.environ.get("PINGLAB_SMOKE") == "1"
    cfg = recipe.configuration(smoke=smoke)
    with stage_run(
        REPO, recipe.SLUG, "compute", run_id=run_id, configuration=cfg
    ) as run:
        run.record["execution"].update(
            environment={"PINGLAB_SMOKE": "1" if smoke else "0"},
            executor="snnlab.sim.GraphExecutor",
            snnlab_version=version("snnlab"),
            trials=[],
        )
        networks = {}
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for relative, cell, point, traces in recipe.trials(smoke=smoke):
                key = f"{'scope' if traces else 'fi'}-{cell}-network"
                if key not in networks:
                    bundle = recipe.author_network(cfg, point, traces=traces)
                    directory = run.export / key
                    bundle.write(directory)
                    (directory / "reports" / "summary.md").rename(
                        directory / "summary.md"
                    )
                    (directory / "reports").rmdir()
                    networks[key] = bundle
                bundle = networks[key]
                started = perf_counter()
                with torch.inference_mode():
                    result = simulate(
                        recipe.execution_request(bundle, point, traces=traces)
                    )
                seconds = perf_counter() - started
                destination = run.export / relative
                destination.mkdir(parents=True, exist_ok=False)
                filename = "recording.npz" if traces else "spikes.npz"
                np.savez_compressed(
                    destination / filename,
                    **recording(result, cfg, point, traces=traces),
                )
                weights = {
                    name: value.detach().cpu().numpy()
                    for name, value in result.parameters.items()
                }
                weights_path = run.export / key / "weights.npz"
                if weights_path.exists():
                    with np.load(weights_path, allow_pickle=False) as retained:
                        if set(retained.files) != set(weights) or any(
                            not np.array_equal(retained[name], value)
                            for name, value in weights.items()
                        ):
                            raise PingstoreError(
                                "weights changed between matched-drive trials"
                            )
                else:
                    np.savez_compressed(weights_path, **weights)
                run.record["execution"]["trials"].append(
                    {
                        "unit": relative,
                        "network_unit": key,
                        "graph_digest": bundle.manifest["graph_digest"],
                        "point": point,
                        "duration_seconds": seconds,
                        "metrics": result.metrics,
                    }
                )
                print(f"{relative}: {seconds:.2f} s", flush=True)
        finally:
            torch.set_num_threads(previous_threads)
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="unused v4 identity reserved before dispatch")
    args = parser.parse_args()
    compute(run_id=args.run_id)


if __name__ == "__main__":
    main()
