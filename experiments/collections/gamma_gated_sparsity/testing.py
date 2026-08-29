from pathlib import Path

from experiments.collections.gamma_gated_sparsity import slurm


def slurm_resources(tmp_path: Path) -> dict:
    """Return a complete synthetic scheduler resource configuration."""
    return {
        "account": "SL2-test",
        "partition": "ampere",
        "mnist_cache": str(tmp_path / "mnist"),
        "uv": "/usr/bin/uv",
        "exp022": {
            tier: {
                "time": "01:00:00",
                "cpus": 4,
                "memory_gb": 16,
                "gpus": 1,
                "concurrency": 2,
            }
            for tier in slurm.TIERS
        },
        "jobs": {
            kind: {"time": "00:30:00", "cpus": 2, "memory_gb": 8, "gpus": 0}
            for kind in ("aggregate", "downstream", "heavy_downstream", "finalize")
        },
    }
