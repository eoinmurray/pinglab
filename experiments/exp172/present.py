"""Render four spatial-relay figures from explicitly pinned analysis; no simulation."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from snnlab.sim import NetworkSnapshot

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
from experiments.exp172.analyse import (
    fit_profile,
    materialise_compute,
    spatial_timecourse,
)
from experiments.exp172.recipe import gaussian
from pingstore.library import source_run, stage_run

sys.path.insert(0, str(REPO))
from experiments import theme


def phase_evidence(spikes, cfg):
    """Population rates, detected peaks and continuous peak-count phases."""
    dt = cfg["dt_ms"]
    t = np.arange(len(spikes["A_E"])) * dt
    phases = {}
    rates = {}
    detected = {}
    for c in "ABC":
        rate = gaussian_filter1d(spikes[c + "_E"].mean(1) * 1000 / dt, 1 / dt)
        peaks, _ = find_peaks(
            rate,
            distance=round(15 / dt),
            prominence=max(0.05, 0.1 * np.ptp(rate[t > 500])),
        )
        rates[c] = rate
        detected[c] = peaks
        phases[c] = (
            np.interp(
                t,
                t[peaks],
                2 * np.pi * np.arange(len(peaks)),
                left=np.nan,
                right=np.nan,
            )
            if len(peaks) >= 2
            else np.full(t.shape, np.nan)
        )
    return t, phases, rates, detected


def relative_phases(spikes, cfg):
    """Continuous sender-minus-C phase; no modulo wrapping or phase reset."""
    t, phases, _, _ = phase_evidence(spikes, cfg)
    return t, {c: phases[c] - phases["C"] for c in "AB"}


def amplitude_panel(ax, course, cfg):
    rows = course["points"]
    for c, color in zip("AB", [theme.INK_BLACK, theme.DEEP_RED], strict=False):
        ax.plot(
            [r["time_ms"] for r in rows],
            [r[c + "_amplitude_hz"] for r in rows],
            color=color,
            lw=1.2,
            label=f"{c}-like amplitude",
        )
    ax.axvline(cfg["switch_ms"], color=theme.MUTED, ls="--", lw=1)
    for start, end in cfg["windows_ms"]:
        ax.axvspan(start, end, color=theme.GREY_LIGHT, alpha=0.25)
    ax.set(
        title=f"C spatial components · {course['window_ms']:g} ms windows, {course['hop_ms']:g} ms step",
        xlabel="Time (ms)",
        ylabel="Fitted Gaussian amplitude (Hz)",
        xlim=(0, cfg["duration_ms"]),
        ylim=(0, None),
    )
    ax.legend(loc="upper right", fontsize=theme.SIZE_ANNOTATION)


def grade_trial(record, ers):
    phase_pass = True
    for e in record["epochs"]:
        c = e["attended"]
        other = "B" if c == "A" else "A"
        plv = e["phase"]["plv"]
        phase_pass &= (
            plv.get(c, 0) >= 0.9
            and plv.get(c, 0) - plv.get(other, 1) >= 0.4
            and len(e["phase"].get("rhythms", {})) == 3
            and all(v["valid"] for v in e["phase"]["rhythms"].values())
        )
    copies = [e.get("fidelity", {}).get("copy_r_squared") for e in record["epochs"]]
    score = (
        min(copies) + 2 * (min(ers) - 0.5)
        if all(v is not None for v in copies)
        else min(ers)
    )
    great = bool(
        phase_pass
        and all(v is not None and v >= 0.85 for v in copies)
        and min(ers) >= 0.9
    )
    return great, score, bool(phase_pass), -record["cue_hz"], record


def select_trial(data, cfg, args):
    representative_seed = args.seed if args.seed is not None else cfg["seeds"][0]
    candidates = []
    for path in sorted(data.glob("seed*/metrics.json")):
        record = json.loads(path.read_text())
        if (
            record["seed"] != representative_seed
            or record["swap"] != args.swap
            or record["cue_hz"] <= 0
        ):
            continue
        ers = [e["er"] for e in record["epochs"]]
        if any(e is None for e in ers):
            continue
        candidates.append(grade_trial(record, ers))
    if args.cue is None:
        if not candidates:
            raise ValueError(
                "No measurable cued trials for the requested seed / assignment"
            )
        best = max(candidates, key=lambda v: v[:4])
        metrics = best[-1]
        args.cue = metrics["cue_hz"]
    else:
        metrics = json.loads(
            (
                data
                / f"seed{representative_seed}-swap{int(args.swap)}-cue{args.cue:g}"
                / "metrics.json"
            ).read_text()
        )
    selection_metadata = dict(
        seed=representative_seed,
        swap=args.swap,
        cue_hz=args.cue,
        trial_id=metrics["id"],
        spatial_window_ms=100.0,
        spatial_hop_ms=5.0,
        criterion="Prefer copy R2 >= .85 and ER >= .9 with phase criteria in both windows; otherwise maximise worst-window copy R2 + 2*(worst-window ER - .5). No fitted rate gain/offset.",
        selection="best saved cue"
        if any(v[-1]["id"] == metrics["id"] for v in candidates)
        and max(candidates, key=lambda v: v[:4])[-1]["id"] == metrics["id"]
        else "explicit override",
        post_selection=True,
        averaging="none; one seed and one spatial assignment",
    )
    selection_metadata["phase_criteria_pass"] = next(
        (v[2] for v in candidates if v[-1]["id"] == metrics["id"]), None
    )
    selection_metadata["great_criteria_pass"] = next(
        (v[0] for v in candidates if v[-1]["id"] == metrics["id"]), None
    )
    (data / "presentation.json").write_text(
        json.dumps(selection_metadata, indent=2) + "\n"
    )
    return metrics, selection_metadata, representative_seed


def plot_populations(
    spikes,
    profiles,
    cfg,
    swap,
    colors,
    raster_fig,
    raster_grid,
    raster_axes,
    profile_fig,
    profile_grid,
    profile_axes,
):
    e_axes = []
    i_axes = []
    for row, c in enumerate("ABC"):
        raster = raster_fig.add_subplot(raster_grid[0, row])
        raster_axes.append(raster)
        ne = spikes[c + "_E"].shape[1]
        ni = spikes[c + "_I"].shape[1]
        for p, offset, color in [("E", 0, theme.INK_BLACK), ("I", ne, theme.DEEP_RED)]:
            t, n = np.nonzero(spikes[c + "_" + p])
            raster.scatter(
                t * cfg["dt_ms"],
                n + offset + 1,
                marker=".",
                s=2.5,
                linewidths=0,
                color=color,
                rasterized=True,
            )
        raster.axhline(ne + 0.5, color=theme.GREY_LIGHT, lw=0.5)
        raster.axvline(cfg["switch_ms"], color=theme.MUTED, ls="--", lw=1)
        for start, end in cfg["windows_ms"]:
            raster.axvline(start, color=theme.GREY_MID, ls=":", lw=0.65)
            raster.axvline(end, color=theme.GREY_MID, ls=":", lw=0.65)
        raster.set(
            title=f"{c} · E/I raster",
            xlim=(0, cfg["duration_ms"]),
            ylim=(0.5, ne + ni + 0.5),
            yticks=[1, ne, ne + ni],
            ylabel="Neuron index",
            xlabel="Time (ms)",
        )
        raster.text(
            0.98,
            0.96,
            "I",
            color=theme.DEEP_RED,
            ha="right",
            va="top",
            transform=raster.transAxes,
            fontweight="bold",
        )
        raster.text(
            0.98,
            0.72,
            "E",
            color=theme.INK_BLACK,
            ha="right",
            va="top",
            transform=raster.transAxes,
            fontweight="bold",
        )
        for col, p in [(1, "E"), (2, "I")]:
            ax = profile_fig.add_subplot(profile_grid[row, col - 1])
            profile_axes.append(ax)
            (e_axes if p == "E" else i_axes).append(ax)
            for epoch in range(2):
                y = profiles[f"epoch{epoch}_{c}_{p}"]
                x = np.arange(len(y)) + 1
                ax.plot(x, y, color=colors[epoch], lw=0.5, alpha=0.22)
                ax.plot(
                    x,
                    gaussian_filter1d(y, 2, mode="wrap"),
                    color=colors[epoch],
                    lw=1.3,
                    label=f"Attend {'AB'[epoch]}",
                )
                if p == "E":
                    sigma = np.sqrt(
                        cfg["spatial_sigma"] ** 2
                        + cfg["sensory_kernel_sigma"] ** 2
                        + (cfg["cross_kernel_sigma"] ** 2 if c == "C" else 0)
                    )
                    fit, pred, templates = fit_profile(
                        y,
                        (cfg["centres"][::-1] if swap else cfg["centres"]),
                        sigma,
                        cfg["minimum_total_amplitude_hz"],
                    )
                    ax.plot(x, pred, color=colors[epoch], ls="--", lw=1.7)
                    if c == "C":
                        for j in range(2):
                            ax.plot(
                                x,
                                fit["background_hz"]
                                + fit["amplitudes_hz"][j] * templates[:, j],
                                color=colors[epoch],
                                ls=":" if j == 0 else "-.",
                                lw=0.9,
                            )
            ax.set(
                title=f"{c}_{p} · spatial rate",
                xlabel="Neuron index",
                ylabel="Rate (Hz)",
                xlim=(1, len(y)),
                ylim=(0, None),
            )
            if row == 0:
                ax.legend(loc="upper right", fontsize=theme.SIZE_ANNOTATION)
    for group in [e_axes, i_axes]:
        upper = max(a.get_ylim()[1] for a in group)
        for a in group:
            a.set_ylim(0, upper)


def render(data, output, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cue", type=float, help="Override automatic best-cue selection (Hz/channel)"
    )
    parser.add_argument(
        "--seed", type=int, help="One input seed; default is the first saved seed"
    )
    parser.add_argument(
        "--swap", action="store_true", help="Use swapped spatial assignments throughout"
    )
    args = parser.parse_args(argv)
    perf_counter()
    cfg = json.loads((data / "configuration.json").read_text())
    metrics, selection_metadata, representative_seed = select_trial(data, cfg, args)
    directory = data / metrics["id"]
    points = json.loads((data / "sweep_points.json").read_text())
    phase_lookup = {}
    for path in sorted(data.glob("*/metrics.json")):
        record = json.loads(path.read_text())
        if record["active"] != "AB":
            continue
        for epoch in record["epochs"]:
            phase_lookup[
                (record["seed"], record["swap"], record["cue_hz"], epoch["attended"])
            ] = epoch["phase"]["plv"]
    for point in points:
        plv = phase_lookup[
            (point["seed"], point["swap"], point["cue_hz"], point["attended"])
        ]
        point["plv_CA"] = plv.get("A")
        point["plv_CB"] = plv.get("B")
        point["differential_plv"] = (
            float(plv["A"] - plv["B"])
            if plv.get("A") is not None and plv.get("B") is not None
            else None
        )
        if point["differential_plv"] is not None:
            assert -1 <= point["differential_plv"] <= 1
    (data / "phase_sweep_points.json").write_text(
        json.dumps(points, indent=2, allow_nan=False) + "\n"
    )
    points = [
        p for p in points if p["seed"] == representative_seed and p["swap"] == args.swap
    ]
    (data / "presentation_points.json").write_text(
        json.dumps(points, indent=2, allow_nan=False) + "\n"
    )
    with np.load(directory / "spikes.npz") as f:
        spikes = dict(f)
    with np.load(directory / "profiles.npz") as f:
        profiles = dict(f)
    theme.apply()
    colors = [theme.INK_BLACK, theme.DEEP_RED]
    course = spatial_timecourse(spikes, cfg, args.swap)
    (data / "spatial-timecourse.json").write_text(
        json.dumps(course, indent=2, allow_nan=False) + "\n"
    )
    raster_fig = plt.figure(figsize=(17, 5), layout="constrained")
    raster_grid = raster_fig.add_gridspec(1, 3)
    profile_fig = plt.figure(figsize=(11, 10), layout="constrained")
    profile_grid = profile_fig.add_gridspec(3, 2)
    signal_fig = plt.figure(figsize=(17, 5), layout="constrained")
    signal_grid = signal_fig.add_gridspec(1, 3)
    weight_fig = plt.figure(figsize=(11, 5), layout="constrained")
    weight_grid = weight_fig.add_gridspec(1, 2)
    raster_axes = []
    profile_axes = []
    signal_axes = []
    plot_populations(
        spikes,
        profiles,
        cfg,
        args.swap,
        colors,
        raster_fig,
        raster_grid,
        raster_axes,
        profile_fig,
        profile_grid,
        profile_axes,
    )
    reference = signal_fig.add_subplot(signal_grid[0, 0])
    signal_axes.append(reference)
    x = (
        np.arange(cfg["network"]["populations"]["E"]["size"])
        / cfg["network"]["populations"]["E"]["size"]
    )
    for j, c in enumerate("AB"):
        reference.plot(
            x * len(x) + 1,
            gaussian(
                x,
                (cfg["centres"][::-1] if args.swap else cfg["centres"])[j],
                cfg["spatial_sigma"],
            ),
            color=colors[j],
            lw=1.8,
            label=f"{c} input reference",
        )
        broadened = np.sqrt(
            cfg["spatial_sigma"] ** 2
            + cfg["sensory_kernel_sigma"] ** 2
            + cfg["cross_kernel_sigma"] ** 2
        )
        reference.plot(
            x * len(x) + 1,
            gaussian(
                x, (cfg["centres"][::-1] if args.swap else cfg["centres"])[j], broadened
            ),
            color=colors[j],
            ls="--",
            lw=1,
            label=f"{c} C-fit template",
        )
    reference.set(
        title="Gaussian references / fit templates",
        xlabel="Equivalent E neuron index",
        ylabel="Normalised shape (unit peak)",
        xlim=(1, len(x)),
        ylim=(0, 1.3),
    )
    reference.legend(loc="upper center", ncol=2, fontsize=theme.SIZE_ANNOTATION)
    relative_ax = signal_fig.add_subplot(signal_grid[0, 1])
    signal_axes.append(relative_ax)
    t, relative = relative_phases(spikes, cfg)
    for c, color in zip("AB", colors, strict=False):
        relative_ax.plot(
            t, relative[c] / (2 * np.pi), color=color, lw=1, label=f"{c} − C"
        )
    relative_ax.axhline(0, color=theme.GREY_MID, ls=":", lw=0.7)
    relative_ax.axvline(cfg["switch_ms"], color=theme.MUTED, ls="--", lw=1)
    for start, end in cfg["windows_ms"]:
        relative_ax.axvspan(start, end, color=theme.GREY_LIGHT, alpha=0.25)
    relative_ax.set(
        title="Unwrapped E-population phase difference",
        xlabel="Time (ms)",
        ylabel="Sender phase − C phase (cycles)",
        xlim=(0, cfg["duration_ms"]),
    )
    relative_ax.legend(loc="upper right", fontsize=theme.SIZE_ANNOTATION)
    amplitude_ax = signal_fig.add_subplot(signal_grid[0, 2])
    signal_axes.append(amplitude_ax)
    amplitude_panel(amplitude_ax, course, cfg)
    parameters = NetworkSnapshot.read(
        directory / "start/snapshot.json"
    ).load_parameters()
    matrices = [parameters[c + "_E_to_C_E.weight"].detach().cpu().numpy() for c in "AB"]
    vmax = max(float(matrix.max()) for matrix in matrices)
    matrix_axes = []
    for col, (c, matrix) in enumerate(zip("AB", matrices, strict=False)):
        ax = weight_fig.add_subplot(weight_grid[0, col])
        matrix_axes.append(ax)
        heat = ax.imshow(
            matrix,
            origin="lower",
            cmap="Greys",
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
            extent=(0.5, matrix.shape[1] + 0.5, 0.5, matrix.shape[0] + 0.5),
            rasterized=True,
        )
        ax.set(
            title=f"{c}_E → C_E · saved weights",
            xlabel="Target C_E neuron index",
            ylabel=f"Source {c}_E neuron index",
            xticks=[1, 100, 200, 300, 400],
            yticks=[1, 100, 200, 300, 400],
        )
    weight_fig.colorbar(
        heat, ax=matrix_axes, label="Synaptic weight (µS)", shrink=0.85, pad=0.02
    )
    for name, figure, panels in [
        ("rasters", raster_fig, raster_axes),
        ("profiles", profile_fig, profile_axes),
        ("selection", signal_fig, signal_axes),
        ("weights", weight_fig, matrix_axes),
    ]:
        theme.label_panels(panels)
        figure.savefig(output / f"{name}.png", dpi=240)
        figure.savefig(output / f"{name}.svg")
        plt.close(figure)
    for name in [
        "presentation.json",
        "presentation_points.json",
        "phase_sweep_points.json",
        "spatial-timecourse.json",
    ]:
        shutil.copy2(data / name, output / name)
    shutil.copy2(data / "circuit.svg", output / "circuit.svg")
    return selection_metadata


def present(source, run_id=None, cue=None, seed=None):
    started = perf_counter()
    analysed = source_run(
        REPO / ".pingstore", source, experiment="exp172", stage="analyse"
    )
    computed = source_run(
        REPO / ".pingstore",
        analysed.record["inputs"]["compute"]["run_id"],
        experiment="exp172",
        stage="compute",
        reference=analysed.record["inputs"]["compute"],
    )
    with stage_run(
        REPO,
        "exp172",
        "present",
        inputs={"analyse": analysed, "compute": computed},
        run_id=run_id,
        configuration=dict(
            cue=cue, seed=seed, spatial_window_ms=100.0, spatial_hop_ms=5.0
        ),
    ) as run:
        work = run.scratch / "evidence"
        materialise_compute(computed, work)
        shutil.copy2(analysed.file("sweep_points.json"), work / "sweep_points.json")
        args = []
        if cue is not None:
            args += ["--cue", str(cue)]
        if seed is not None:
            args += ["--seed", str(seed)]
        selected = render(work, run.export, args)
        trials = json.loads((work / "summary.json").read_text())["trials"]
        trial = next(r for r in trials if r["id"] == selected["trial_id"])
        cfg = computed.record["execution"]["configuration"]
        parameters = NetworkSnapshot.read(
            work / trial["id"] / "start/snapshot.json"
        ).load_parameters()
        matrix_stats = {
            k: dict(
                shape=list(v.shape),
                nonzero_fraction=float((v != 0).float().mean()),
                weight_min_us=float(v.min()),
                weight_max_us=float(v.max()),
                incoming_mean_us=float(v.sum(0).mean()),
            )
            for k, v in parameters.items()
            if k.endswith(".weight")
        }
        stability_selection = (
            json.loads(computed.file("stability-selection.json").read_text())
            if computed.file("stability-selection.json").exists()
            else None
        )
        (run.export / "numbers.json").write_text(
            json.dumps(
                dict(
                    configuration=cfg,
                    selection=selected,
                    stability_selection=stability_selection,
                    metrics=trial,
                    analysis=json.loads(analysed.file("results.json").read_text()),
                    matrix_stats=matrix_stats,
                    elapsed_seconds=perf_counter() - started,
                ),
                indent=2,
            )
            + "\n"
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    print(f"Presentation {perf_counter() - started:.2f} s; no simulation")
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--cue", type=float)
    parser.add_argument("--seed", type=int)
    present(**vars(parser.parse_args()))
