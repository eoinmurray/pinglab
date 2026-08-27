"""Compute-only simulator adapter. All large process-boundary arrays are scratch."""

import fcntl
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from experiments.helpers.checkpoints import cache_tag, resolve_checkpoint
from experiments.helpers.run_cli import run_cli
from pingstore.contracts import write_json_atomic

from .recipe import CHECKPOINT_ROLE, EVAL_SEED
from .transforms import _build_override


class Simulator:
    def __init__(self, scratch, provenance, configuration, *, baseline_root=None):
        self.scratch = scratch
        self.baseline_root = baseline_root or scratch
        self.provenance = provenance
        self.configuration = configuration
        self.cache = {}
        self.command_count = 0

    def run(self, args):
        self.provenance.mkdir(parents=True, exist_ok=True)
        self.command_count += 1
        write_json_atomic(
            self.provenance / f"{self.command_count:04d}.json", {"arguments": args}
        )
        run_cli(args)

    def checkpoint_path(self, train_dir: Path) -> Path:
        return resolve_checkpoint(train_dir, CHECKPOINT_ROLE)["path"]

    def _baseline_complete(self, rasters_path: Path, metrics_path: Path) -> bool:
        """True iff a finished baseline (raster + metrics) is already on disk and
        loadable. Lets all-but-the-first sharer of a train_dir reuse it — see
        _run_baseline."""
        if not (rasters_path.exists() and metrics_path.exists()):
            return False
        try:
            json.loads(metrics_path.read_text())
            with np.load(rasters_path):
                pass
        except Exception:  # noqa: BLE001 — a torn/legacy file counts as incomplete
            return False
        return True

    def _run_baseline(self, train_dir: Path, tau_gaba=None):
        """Reuse the reservation-scoped baseline; serialize writers across compute shards."""
        checkpoint = resolve_checkpoint(train_dir, CHECKPOINT_ROLE)
        key = f"{train_dir}|{tau_gaba}|{cache_tag(checkpoint)}"
        if key not in self.cache:
            out_dir = (
                self.baseline_root / "baseline" / train_dir.name / cache_tag(checkpoint)
            ).resolve()
            rasters_path = out_dir / "rasters.npz"
            metrics_path = out_dir / "metrics.json"
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "cache.lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                marker = out_dir / "recipe.json"
                if (
                    marker.exists()
                    and json.loads(marker.read_text()) != self.configuration
                ):
                    raise ValueError(
                        "baseline scratch belongs to a different compute recipe"
                    )
                if not self._baseline_complete(rasters_path, metrics_path):
                    out_dir.mkdir(parents=True, exist_ok=True)
                    # A unique temp dir per WRITER: two pods are separate containers and
                    # can share a PID, so a pid-named dir on the shared volume would
                    # collide. mkdtemp guarantees uniqueness across pods.
                    tmp = Path(
                        tempfile.mkdtemp(
                            prefix=f".{train_dir.name}.tmp.", dir=out_dir.parent
                        )
                    )
                    try:
                        cmd = [
                            "sim",
                            "--infer",
                            "--load-config",
                            str((train_dir / "config.json").resolve()),
                            "--load-weights",
                            str(self.checkpoint_path(train_dir)),
                            "--outputs",
                            "rasters",
                            "--out-dir",
                            str(tmp),
                        ]
                        if tau_gaba is not None:
                            cmd += ["--tau-gaba", str(tau_gaba)]
                        cmd += [
                            "--max-samples",
                            str(self.configuration["evaluation_samples"]),
                        ]
                        self.run(cmd)
                        # Publish atomically; metrics last so _baseline_complete only
                        # passes once both files are live.
                        os.replace(tmp / "rasters.npz", rasters_path)
                        os.replace(tmp / "metrics.json", metrics_path)
                    finally:
                        shutil.rmtree(tmp, ignore_errors=True)
                write_json_atomic(marker, self.configuration)
            m = json.loads(metrics_path.read_text())
            with np.load(
                metrics_path.parent / "rasters.npz", allow_pickle=False
            ) as data:
                R = dict(data)
            self.cache[key] = (m, R)
        return self.cache[key]

    def _build_override_file(self, R, condition, gen, dt_ms, out_path):
        """Build a sparse I-override NPZ from baseline rasters R by applying the pure
        _build_override transform per trial (per-trial independent). The transform stays
        in the notebook; the CLI only injects the result."""
        import torch

        T, n_i, n_tr = int(R["T"]), int(R["n_i"]), int(R["n_trials"])
        tr = R["i_trial"]
        order = np.argsort(tr, kind="stable")
        tr, tt, tc = tr[order], R["i_t"][order], R["i_cell"][order]
        bounds = np.searchsorted(tr, np.arange(n_tr + 1))
        out_tr, out_t, out_c = [], [], []
        for b in range(n_tr):
            lo, hi = bounds[b], bounds[b + 1]
            s_i = np.zeros((T, 1, n_i), dtype=np.float32)
            s_i[tt[lo:hi], 0, tc[lo:hi]] = 1.0
            ov = _build_override(torch.from_numpy(s_i), condition, gen, dt_ms=dt_ms)
            ov = ov.detach().cpu().numpy()[:, 0, :]  # (T, n_i)
            ti, ci = ov.nonzero()
            out_t.append(ti.astype("int32"))
            out_c.append(ci.astype("int32"))
            out_tr.append(np.full(ti.size, b, dtype="int32"))
        cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, "int32")  # noqa: E731
        np.savez(
            out_path,
            n_trials=np.int32(n_tr),
            T=np.int32(T),
            n_i=np.int32(n_i),
            i_trial=cat(out_tr),
            i_t=cat(out_t),
            i_cell=cat(out_c),
        )

    def _run_with_override(
        self, train_dir: Path, override_path: Path, tau_gaba=None
    ) -> dict:
        """Pass B via `sim --infer --i-override-file`; return metrics."""
        checkpoint = resolve_checkpoint(train_dir, CHECKPOINT_ROLE)
        out_dir = (
            self.scratch
            / "ovrun"
            / f"{train_dir.name}__{override_path.stem}"
            / cache_tag(checkpoint)
        ).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "sim",
            "--infer",
            "--load-config",
            str((train_dir / "config.json").resolve()),
            "--load-weights",
            str(self.checkpoint_path(train_dir)),
            "--i-override-file",
            str(override_path),
            "--out-dir",
            str(out_dir),
        ]
        if tau_gaba is not None:
            cmd += ["--tau-gaba", str(tau_gaba)]
        cmd += ["--max-samples", str(self.configuration["evaluation_samples"])]
        self.run(cmd)
        return json.loads((out_dir / "metrics.json").read_text())

    def _snapshot(
        self, train_dir: Path, sample_idx: int, name: str, i_override=None, reuse=False
    ):
        """Single-trial snapshot via `sim --infer --sample-index N` (optional
        --i-override-file); return the loaded snapshot.npz dict.

        reuse=True: read the already-collected snapshot.npz from the same out_dir the
        compute path writes to, WITHOUT running the sim. Returns None on a cache miss
        so callers can fall through to compute."""
        checkpoint = resolve_checkpoint(train_dir, CHECKPOINT_ROLE)
        out_dir = (
            self.scratch
            / "condraster"
            / f"{train_dir.name}_{name}"
            / cache_tag(checkpoint)
        ).resolve()
        if reuse:
            try:
                return self.read_snapshot(out_dir / "snapshot.npz")
            except (OSError, ValueError):
                return None
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "sim",
            "--infer",
            "--load-config",
            str((train_dir / "config.json").resolve()),
            "--load-weights",
            str(self.checkpoint_path(train_dir)),
            "--sample-index",
            str(sample_idx),
            "--out-dir",
            str(out_dir),
        ]
        if i_override is not None:
            cmd += ["--i-override-file", str(i_override)]
        else:
            cmd += ["--outputs", "rasters"]  # baseline pass exposes the I-stream
        self.run(cmd)
        return self.read_snapshot(out_dir / "snapshot.npz")

    @staticmethod
    def read_snapshot(path):
        with np.load(path, allow_pickle=False) as data:
            return {key: np.array(data[key]) for key in ("spk_e", "spk_i", "label")}

    def evaluate(self, train_dir, job):
        import torch

        cfg = json.loads((train_dir / "config.json").read_text())
        baseline, rasters = self._run_baseline(train_dir)
        if job["condition"] == "baseline":
            return baseline
        gen = torch.Generator().manual_seed(EVAL_SEED + 17 + job["seed_offset"])
        with tempfile.TemporaryDirectory(
            prefix=".override-", dir=self.scratch
        ) as directory:
            path = Path(directory) / (job["id"] + ".npz")
            self._build_override_file(
                rasters, job["condition"], gen, float(cfg["dt"]), path
            )
            return self._run_with_override(train_dir, path)

    def recording(self, train_dir, condition, offset):
        import torch

        cfg = json.loads((train_dir / "config.json").read_text())
        sample = self.configuration["raster"]["sample_index"]
        base = self._snapshot(train_dir, sample, f"base_s{sample}", reuse=True)
        if base is None:
            base = self._snapshot(train_dir, sample, f"base_s{sample}")
        spikes = base["spk_i"]
        if spikes.ndim == 3:
            spikes = spikes[:, 0, :]
        gen = torch.Generator().manual_seed(EVAL_SEED + 17 + offset)
        ov = _build_override(
            torch.from_numpy(spikes[:, None, :].astype(np.float32)),
            condition,
            gen,
            dt_ms=float(cfg["dt"]),
        )
        ov = ov.detach().cpu().numpy()[:, 0, :]
        ti, ci = ov.nonzero()
        with tempfile.TemporaryDirectory(
            prefix=".override-", dir=self.scratch
        ) as directory:
            path = Path(directory) / "recording.npz"
            np.savez(
                path,
                n_trials=np.int32(1),
                T=np.int32(ov.shape[0]),
                n_i=np.int32(ov.shape[1]),
                i_trial=np.zeros(ti.size, "int32"),
                i_t=ti.astype("int32"),
                i_cell=ci.astype("int32"),
            )
            return self._snapshot(train_dir, sample, condition, i_override=path)
