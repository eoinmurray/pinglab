"""Generate independent finite-window samples; never analyse, draw or publish."""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "tools")]

import numpy as np
from experiments.exp081 import recipe
from pingstore.contracts import write_json_atomic
from pingstore.stages import stage_run


def torch_device() -> Any:
    import torch

    requested = os.environ.get("EXP081_DEVICE", "auto")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def simulate_features(
    input_rates_hz: np.ndarray,
    probes_uS: np.ndarray,
    draws: int,
    seed: int,
    *,
    config: dict | None = None,
) -> np.ndarray:
    """Generate fresh finite-window features for aligned drive/probe conditions."""
    import torch

    cfg = config if config is not None else recipe.configuration()
    rates, probes = np.broadcast_arrays(
        np.asarray(input_rates_hz, dtype=np.float32),
        np.asarray(probes_uS, dtype=np.float32),
    )
    device = torch_device()
    probability = torch.as_tensor(
        rates.reshape(-1, 1) * cfg["dt_ms"] / 1000.0, device=device
    )
    probe = torch.as_tensor(probes.reshape(-1, 1), device=device)
    shape = (rates.size, draws)
    conductance = torch.zeros(shape, device=device)
    voltage = torch.full(shape, cfg["membrane"]["E_L_mV"], device=device)
    feature_sum = torch.zeros(shape, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    decay = math.exp(-cfg["dt_ms"] / cfg["membrane"]["tau_ampa_ms"])
    for _ in range(int(round(cfg["presentation_ms"] / cfg["dt_ms"]))):
        events = torch.rand(shape, device=device, generator=generator) < probability
        conductance = conductance * decay + probe * events
        total_g = cfg["membrane"]["g_L_uS"] + conductance
        equilibrium = (
            cfg["membrane"]["g_L_uS"] * cfg["membrane"]["E_L_mV"]
            + conductance * cfg["membrane"]["E_e_mV"]
        ) / total_g
        voltage = equilibrium + (voltage - equilibrium) * torch.exp(
            -cfg["dt_ms"] * total_g / cfg["membrane"]["C_m_nF"]
        )
        feature_sum += voltage - cfg["membrane"]["E_L_mV"]
    output = (
        (feature_sum / int(round(cfg["presentation_ms"] / cfg["dt_ms"]))).cpu().numpy()
    )
    return output.reshape(*rates.shape, draws)


def compute(*, run_id: str | None = None) -> str:
    import torch

    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    with stage_run(
        REPO, recipe.SLUG, "compute", run_id=run_id, configuration=cfg
    ) as run:
        environment = {
            "PINGLAB_SMOKE": "1" if cfg["profile"] == "smoke" else "0",
            "EXP081_DEVICE": str(torch_device()),
        }
        run.record["execution"]["environment"] = environment
        rates, probes = np.meshgrid(cfg["input_rate_grid_hz"], cfg["probes_uS"])
        features = simulate_features(
            rates, probes, cfg["moment_draws"], cfg["moment_seed"], config=cfg
        )
        samples = simulate_features(
            np.asarray(cfg["distribution_rates_hz"]),
            np.full(len(cfg["distribution_rates_hz"]), cfg["nominal_probe_uS"]),
            cfg["distribution_draws"],
            cfg["distribution_seed"],
            config=cfg,
        )
        np.savez_compressed(
            run.export / "feature_samples.npz",
            samples_mV=features,
            input_rates_hz=rates,
            probes_uS=probes,
        )
        np.savez_compressed(
            run.export / "distribution_samples.npz",
            samples_mV=samples,
            input_rates_hz=np.asarray(cfg["distribution_rates_hz"]),
        )
        write_json_atomic(
            run.scratch / "environment.json",
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": str(torch_device()),
            },
        )
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="unused v4 identity reserved before dispatch")
    args = parser.parse_args()
    compute(run_id=args.run_id)


if __name__ == "__main__":
    main()
