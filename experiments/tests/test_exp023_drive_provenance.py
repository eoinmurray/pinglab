from __future__ import annotations

from pathlib import Path

import numpy as np
from experiments import exp023


def _arg(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_raster_drive_provenance_matches_exact_execution_arguments() -> None:
    provenance = exp023.drive_provenance()["raster_operating_points"]
    for cell in exp023.CELLS:
        executed = exp023.raster_args(cell, Path("/tmp/ignored"))
        recorded = provenance[cell]
        assert recorded["input"] == _arg(executed, "--input") == "synthetic-spikes"
        assert recorded["input_rate_hz"] == float(_arg(executed, "--input-rate"))
        assert recorded["ei_strength"] == float(_arg(executed, "--ei-strength"))
        assert recorded["t_ms"] == float(_arg(executed, "--t-ms"))
        assert recorded["dt_ms"] == float(_arg(executed, "--dt"))


def test_fi_provenance_is_separate_and_matches_executed_grid(
    monkeypatch, tmp_path: Path,
) -> None:
    executed: list[list[str]] = []

    def fake_run_cli(args: list[str]) -> None:
        executed.append(args)
        out_dir = Path(_arg(args, "--out-dir"))
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_dir / "snapshot.npz",
            spk_e=np.zeros((2, 2), dtype=bool),
            spk_i=np.zeros((2, 1), dtype=bool),
            dt=np.asarray(exp023.DT_MS),
        )

    monkeypatch.setattr(exp023, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(exp023, "run_cli", fake_run_cli)
    exp023.fi_sweep()

    recorded = exp023.drive_provenance()["fi_sweep"]
    assert len(executed) == len(exp023.FI_EI) * len(exp023.FI_RATES_HZ)
    assert recorded["input_rates_hz"] == exp023.FI_RATES_HZ
    assert {
        float(_arg(args, "--input-rate")) for args in executed
    } == set(recorded["input_rates_hz"])
    assert all(_arg(args, "--input") == recorded["input"] for args in executed)
    assert {
        cell: float(exp023.FI_EI[cell]) for cell in exp023.FI_EI
    } == recorded["ei_strength_by_cell"]
