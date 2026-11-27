"""Compare retained Fig. 2 outputs; no simulation or model selection."""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "libraries"))
from pingstore.library import Store  # noqa: E402


def main():
    started = time.perf_counter()
    identity = "exp175-r20261118160856869264858065-compute"
    source = Store(ROOT / ".pingstore").get(identity)
    with np.load(source.file("reconstructions.npz")) as z:
        output, target = (z[k].astype(float) for k in ("reconstructed", "targets"))
        cue, phase = z["cue"], z["phase"]
        attended = np.where(cue[:, None] == 0, z["images_A"], z["images_B"])
        distractor = np.where(cue[:, None] == 0, z["images_B"], z["images_A"])
        np.testing.assert_array_equal(target, attended)
    assert output.shape == target.shape == distractor.shape == (8, 784)
    assert all(np.isfinite(a).all() for a in (output, target, distractor))

    def mse(a, b):
        return np.mean((a - b) ** 2, axis=1)

    def cosine(a, b):
        return np.sum(a * b, axis=1) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        )

    target_mse, distractor_mse = mse(output, target), mse(output, distractor)
    target_cos, distractor_cos = cosine(output, target), cosine(output, distractor)
    blank_mse = mse(np.zeros_like(target), target)
    rows = [
        dict(
            row=i + 1,
            target_group=i // 4 + 1,
            sender="AB"[int(cue[i])],
            window_ms=[[50, 90], [150, 190]][int(phase[i])],
            target_mse=float(target_mse[i]),
            distractor_mse=float(distractor_mse[i]),
            target_cosine=float(target_cos[i]),
            distractor_cosine=float(distractor_cos[i]),
            blank_target_mse=float(blank_mse[i]),
            target_closer_mse=bool(target_mse[i] < distractor_mse[i]),
        )
        for i in range(8)
    ]
    result = dict(
        source_run=identity,
        artifact="reconstructions.npz",
        rows=rows,
        output_clipped=False,
        similarity_scope="Eight retained Fig. 2 examples; two target identities, four cue conditions each",
        mean_target_mse=float(target_mse.mean()),
        mean_distractor_mse=float(distractor_mse.mean()),
        mean_blank_target_mse=float(blank_mse.mean()),
        mean_target_cosine=float(target_cos.mean()),
        mean_distractor_cosine=float(distractor_cos.mean()),
        target_closer_mse_count=int(np.sum(target_mse < distractor_mse)),
        target_closer_cosine_count=int(np.sum(target_cos > distractor_cos)),
        relative_mse_reduction_vs_blank=float(1 - target_mse.mean() / blank_mse.mean()),
        limitation="Descriptive analysis of two validation target identities; rows are dependent. No population significance test or independent test-set claim.",
        elapsed_seconds=time.perf_counter() - started,
    )
    destination = Path(__file__).parent / "data" / "similarity.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
