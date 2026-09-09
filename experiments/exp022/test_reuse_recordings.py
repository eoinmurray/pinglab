"""Scientific diagnostic acceptance for the replacement model bank."""

from pathlib import Path

import numpy as np
import pytest
from experiments.exp022 import recipe, reuse
from experiments.helpers.operating_point import duration_steps
from pingstore.contracts import PingstoreError


def _recording(dt: float) -> dict:
    steps = duration_steps(recipe.T_MS, dt)
    return {
        "dt": np.float32(dt), "n_e": 1024, "n_i": 256, "label": 0,
        "spk_e": np.zeros((steps, 1024), dtype=np.uint8),
        "spk_i": np.zeros((steps, 256), dtype=np.uint8),
    }


def _name(dt: float) -> str:
    return next(cell["name"] for cell in recipe.CANONICAL_CELLS
                if cell["family"] == "dt" and cell["dt_ms"] == dt and cell["seed"] == 42)


@pytest.mark.parametrize("dt", recipe.DT_SWEEP_MS)
def test_exact_grid_recordings_allow_float32_metadata(tmp_path: Path, dt: float):
    path = tmp_path / "recording.npz"
    np.savez_compressed(path, **_recording(dt))
    reuse._validate_recording(path, _name(dt))


@pytest.mark.parametrize("defect", ["dt", "n_e", "n_i", "label", "duration", "neurons", "spikes", "missing"])
def test_wrong_probe_is_rejected(tmp_path: Path, defect: str):
    payload = _recording(0.6)
    if defect in {"dt", "n_e", "n_i", "label"}:
        payload[defect] = 5
    elif defect == "duration":
        payload["spk_e"] = payload["spk_e"][:-1]
    elif defect == "neurons":
        payload["spk_i"] = payload["spk_i"][:, :-1]
    elif defect == "spikes":
        payload["spk_e"][0, 0] = 2
    else:
        del payload["spk_i"]
    path = tmp_path / "recording.npz"
    np.savez_compressed(path, **payload)
    with pytest.raises(PingstoreError, match="invalid diagnostic"):
        reuse._validate_recording(path, _name(0.6))


def test_corrupt_probe_is_rejected(tmp_path: Path):
    path = tmp_path / "recording.npz"
    path.write_bytes(b"incomplete recording")
    with pytest.raises(PingstoreError, match="invalid diagnostic"):
        reuse._validate_recording(path, _name(0.6))
