"""Exp022 analysis: read completed computation, retain numbers and plot-ready arrays."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "tools")]

from experiments.exp022 import recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import source_run, stage_run

from helpers.checkpoints import checkpoint_provenance, epoch_metrics
from helpers.fmt import format_duration


def _gamma_psd(spk_i, dt):
    """I-population power spectrum + gamma peak from a single-trial raster.

    1 ms population rate → 3 ms Gaussian smooth (suppresses the harmonics that
    sharp inhibitory bursts inject) → Hann-windowed FFT. Returns (freqs, psd,
    f_gamma); f_gamma is None when the I population is silent (COBA — no E/I
    loop) or no prominent γ is resolved. The 3 ms kernel is chosen so the visible
    peak is the FUNDAMENTAL, not the 2× harmonic — verified against the
    multi-trial τ_GABA scaling, which matches nb041 (τ=6 ms → ≈ 45 Hz)."""
    import numpy as np

    T, ni = spk_i.shape
    spm = max(1, round(1.0 / dt))          # timesteps per 1 ms bin (dt-aware:
    nb = T // spm                          # the Δt sweep cells vary dt)
    b = spk_i[: nb * spm].reshape(nb, spm, ni).sum(axis=(1, 2)).astype(float)  # pop/ms
    if b.sum() < 50:                       # essentially silent (e.g. COBA I pop)
        return None, None, None
    k = np.exp(-0.5 * ((np.arange(31) - 15) / 3.0) ** 2)
    k /= k.sum()
    x = np.convolve(b - b.mean(), k, "same") * np.hanning(nb)
    fr = np.fft.rfftfreq(nb, 1 / 1000.0)   # 1 kHz after 1 ms binning
    P = np.abs(np.fft.rfft(x)) ** 2
    band = (fr >= 20) & (fr <= 110)
    fpk = float(fr[band][np.argmax(P[band])])
    prom = P[band].max() / (np.median(P[band]) + 1e-9)
    return fr, P, (fpk if prom > 2.5 else None)



def measure_snapshot(source: Path, destination: Path) -> None:
    """Measure the retained raster once; presentation only draws these arrays."""
    with np.load(source, allow_pickle=False) as record:
        se, si, dt = record["spk_e"], record["spk_i"], float(record["dt"])
        steps, ne = se.shape
        ni = si.shape[1]
        duration = steps * dt
        et, ec = np.nonzero(se)
        it, ic = np.nonzero(si)
        e_times, i_times = et * dt, it * dt
        e_hz = se.sum() / ne / (duration / 1000)
        i_hz = si.sum() / max(ni, 1) / (duration / 1000)
        frequencies, power, peak = _gamma_psd(si, dt)
        bins = np.arange(0, duration + 1, 1.0)
        e_rate = np.histogram(e_times, bins=bins)[0] / ne * 1000
        i_rate = np.histogram(i_times, bins=bins)[0] / max(ni, 1) * 1000
        if frequencies is None:
            plot_frequencies = normalized_power = np.array([])
        else:
            band = (frequencies >= 20) & (frequencies <= 110)
            norm = power[band].max() or 1.0
            selected = (frequencies >= 8) & (frequencies <= 120)
            plot_frequencies = frequencies[selected]
            normalized_power = np.clip(power[selected] / norm, 0, 1.3)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination, e_times=e_times, e_cells=ec, i_times=i_times, i_cells=ic,
            ne=ne, ni=ni, duration_ms=duration, e_hz=e_hz, i_hz=i_hz,
            bin_centres=(bins[:-1] + bins[1:]) / 2, e_rate=e_rate, i_rate=i_rate,
            frequencies=plot_frequencies, normalized_power=normalized_power,
            gamma_hz=peak if peak is not None else np.nan,
        )


def analyse(identity: str, *, run_id: str | None = None) -> str:
    root = REPO / ".pingstore"
    compute = source_run(root, identity, stage="compute", experiment=recipe.SLUG)
    inputs = {"compute": compute}
    bank = compute
    if "bank" in compute.record["inputs"]:
        reference = compute.record["inputs"]["bank"]
        bank = source_run(root, reference["run_id"], stage="compute",
                          experiment=recipe.SLUG, reference=reference)
        inputs["bank"] = bank
    expected = {cell["name"] for cell in recipe.CANONICAL_CELLS}
    actual = {path.name for path in bank.export.iterdir() if path.is_dir()}
    if actual != expected:
        raise PingstoreError("analysis requires the complete explicit 102-cell bank")
    started = time.monotonic()
    with stage_run(REPO, recipe.SLUG, "analyse", inputs=inputs, run_id=run_id,
                   configuration={"measurement": "retained epoch curves and fixed-probe diagnostics"}) as run:
        rows, plots, configurations = [], [], {}
        snapshots = []
        missing_snapshots = []
        for cell in recipe.CANONICAL_CELLS:
            directory = bank.export / cell["name"]
            metrics = load_json(directory / "metrics.json")
            config = load_json(directory / "config.json")
            history = epoch_metrics(directory)
            if not history:
                raise PingstoreError(f"no retained epoch history: {cell['name']}")
            e_rate, i_rate = recipe.final_rates(directory)
            rows.append({
                "name": cell["name"], "model": cell["model"], "family": cell["family"],
                "training_run_id": cell["training_run_id"], "tag": cell["tag"],
                "seed": cell["seed"], "acc": metrics["best_acc"],
                "best_epoch": metrics["best_epoch"], "rate_e": e_rate, "rate_i": i_rate,
            })
            epochs, accuracy = recipe.training_curve(directory)
            plots.append({
                "name": cell["name"], "family": cell["family"], "tag": cell["tag"],
                "model": cell["model"], "epochs": epochs, "accuracy_pct": accuracy,
            })
            configurations[cell["name"]] = config
            if cell["seed"] == 42:
                snapshot = compute.file("snapshots", cell["name"], "recording.npz")
                if snapshot.is_file():
                    measure_snapshot(snapshot, run.export / "rasters" / f"{cell['name']}.npz")
                    snapshots.append(cell["name"])
                else:
                    missing_snapshots.append(cell["name"])
        directories = [bank.export / cell["name"] for cell in recipe.CANONICAL_CELLS]
        checkpoint_records = {
            role: checkpoint_provenance(directories, role)
            for role in ("best_validation", "final_epoch")
        }
        elapsed = time.monotonic() - started
        common = {}
        for field in ("dataset", "epochs", "t_ms", "dt", "batch_size"):
            values = list(dict.fromkeys(config.get(field) for config in configurations.values()))
            common["dt_ms" if field == "dt" else field] = values[0] if len(values) == 1 else values
        common["max_samples_canonical"] = sorted({
            configurations[cell["name"]]["max_samples"] for cell in recipe.CANONICAL_CELLS
            if cell["family"] == "canonical"
        })
        common["max_samples_sweeps"] = sorted({
            configurations[cell["name"]]["max_samples"] for cell in recipe.CANONICAL_CELLS
            if cell["family"] != "canonical"
        })
        summary = {
            "notebook_run_id": run.run_id, "compute_run_id": identity,
            "bank_run_id": bank.record["run_id"],
            "git_sha": run.record["provenance"]["git_commit"],
            "duration_s": round(elapsed, 1), "duration": format_duration(elapsed),
            "standard": common,
            "training_root": {"run_id": bank.record["run_id"],
                              "path": bank.export.relative_to(bank.directory).as_posix()},
            "result_checkpoint_provenance": checkpoint_records[recipe.RESULT_CHECKPOINT_ROLE],
            "checkpoint_provenance": checkpoint_records,
            "families": recipe.FAMILY_ORDER, "training_run_ids": recipe.TRAINING_RUN_IDS,
            "family_status": {
                family: {"cells": sum(row["family"] == family for row in rows),
                         "trained": sum(row["family"] == family for row in rows)}
                for family in recipe.FAMILY_ORDER
            },
            "n_cells": len(rows), "cells": rows,
            "rasters": snapshots, "missing_raw_rasters": missing_snapshots,
            "measurement_notes": {
                "accuracy": "metrics.best_acc; selected epoch defined by retained checkpoint policy",
                "rates": "last retained epoch test_rate_e/test_rate_i, falling back to rate_e/rate_i",
                "curves": "retained epoch acc field; legacy field name does not change dataset provenance",
            },
        }
        write_json_atomic(run.export / "results.json", recipe._json_safe(summary))
        write_json_atomic(run.export / "curves.json", {"cells": plots})
        write_json_atomic(run.export / "cell-configurations.json", configurations)
        (run.directory / "README.md").write_text(
            "# Exp022 analysis\n\n"
            f"Measurements from compute run `{identity}`, bank `{bank.record['run_id']}`.\n\n"
            "Numerical results and both checkpoint-role inventories are in "
            "`export/results.json`; plot-ready curves are in `export/curves.json`. "
            "Available retained probe measurements are under `export/rasters/`.\n\n"
            f"Missing raw seed-42 probes: {len(missing_snapshots)}. No training or simulation "
            "was performed. Missing historical recordings are not reconstructed.\n"
        )
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp022 compute run ID")
    parser.add_argument("--run-id", help="already reserved analyse identity")
    args = parser.parse_args()
    analyse(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
