"""Present the lifetime replay and learning history from one analysis input.

Decisions, energy, activity, fixed-cue errors and post-update weights come from
validated recordings. Presentation adds no simulation. A shared encounter axis
aligns diagnostic prediction error with pathway weight norms and their ratio.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from snnlab.sim import NetworkSnapshot


def recorded_events(computed, row):
    import json

    return json.loads(computed.file(Path(row["directory"]) / "events.json").read_text())


def network_payload(snapshot, arrays, cfg):
    """Derive displayed topology and fixed magnitudes from native parameters."""
    parameters = snapshot.load_parameters()
    e_count, i_count = snapshot.populations["E"].size, snapshot.populations["I"].size
    offsets = dict(
        E=0, I=e_count, sensory=e_count + i_count, internal=e_count + i_count + 3
    )
    edges, scales = [], {}
    for name, projection in snapshot.projections.items():
        weights = np.asarray(parameters[name + ".weight"])
        plastic = ("EI", "IE").index(name) if name in {"EI", "IE"} else -1
        scales[name] = (
            float(arrays["weights_pre"][:, plastic].max())
            if plastic >= 0
            else float(weights.max())
        )
        for pre, post in np.ndindex(weights.shape):
            if plastic < 0 and weights[pre, post] == 0:
                continue
            edges.append(
                dict(
                    source=offsets[projection.source_id] + pre,
                    target=offsets[projection.target_id] + post,
                    pathway=name,
                    plastic_path=plastic,
                    parameter_index=pre * weights.shape[1] + post,
                    weight=float(weights[pre, post]),
                )
            )
    return dict(
        e_cells=e_count,
        i_cells=i_count,
        edges=edges,
        flow_scales=scales,
        dt_ms=cfg["dt_ms"],
        duration_ms=cfg["presentation_ms"],
        initial_rate=cfg["initial_rate"],
        rate_unit="normalized rate (0–1)",
    )


def attach_recording(event, arrays, index, cfg, *, internal=None):
    rates = np.concatenate(
        (arrays["replay_E_rate"][index], arrays["replay_I_rate"][index]), axis=1
    )
    event["rates"] = np.vstack(
        (np.full((1, rates.shape[1]), cfg["initial_rate"]), rates)
    ).tolist()
    event["weights"] = arrays["weights_pre"][index].tolist()
    event["weights_after"] = arrays["weights_post"][index].tolist()
    drives = np.zeros((len(rates) + 1, 6))
    if internal is None:
        drives[1:, :3] = [event["cue"], 1 - event["cue"], 1]
    else:
        drives[1:, 3:] = internal
    event["drives"] = drives.tolist()
    return event


def lifecycle_events(computed, row, cfg):
    directory = Path(row["directory"])
    food = recorded_events(computed, row)
    with np.load(computed.file(directory / "data.npz")) as arrays:
        for index, event in enumerate(food):
            event["conception"] = False
            attach_recording(event, arrays, index, cfg)
    prenatal = []
    if cfg["genome"]["prenatal_windows"]:
        prenatal = json.loads(
            computed.file(directory / "development-events.json").read_text()
        )
        with np.load(computed.file(directory / "development.npz")) as arrays:
            for index, event in enumerate(prenatal):
                event.update(
                    trial=event["before_trial"] - 1,
                    phase_age=event["window"],
                    age=0,
                    conception=event["window"] == 1,
                    birth=False,
                    death=False,
                    energy_before=None,
                    energy_after=None,
                    cue=None,
                    delta=None,
                    accepted=False,
                    explored=False,
                    updated=True,
                    validation_mse=None,
                )
                attach_recording(
                    event, arrays, index, cfg, internal=arrays["internal_drives"][index]
                )
    by_trial = {}
    for event in prenatal:
        by_trial.setdefault(event["before_trial"], []).append(event)
    timeline = []
    for event in food:
        timeline.extend(by_trial.get(event["trial"], []))
        timeline.append(event)
    for index, event in enumerate(timeline, 1):
        event["step"] = index
    return timeline


def replay_payload(computed, summary):
    cfg = summary["configuration"]
    conditions = []
    network = None
    for row in summary["conditions"]:
        events = lifecycle_events(computed, row, cfg)
        directory = Path(row["directory"])
        snapshot = NetworkSnapshot.read(
            computed.file(directory.parent / "snapshot-initial/start/snapshot.json")
        )
        network = network_payload(
            snapshot, {"weights_pre": np.asarray([e["weights"] for e in events])}, cfg
        )
        conditions.append(dict(seed=row["seed"], sign=row["sign"], events=events))
    return dict(
        schema="exp154.lifecycle/v7",
        network=network,
        genome=cfg["genome"],
        food_encounters=cfg["lifetime_presentations"],
        initialization=cfg["initialization"],
        mapping=dict(sign=1, formula="2x - 1"),
        conditions=conditions,
        initial_energy=cfg["initial_energy"],
        metabolic_cost=cfg["metabolic_cost"],
        exploration_probability=cfg["exploration_probability"],
        scope=summary["scope"],
        source_compute=computed.reference,
    )


def load_analysis(identity, protocol):
    analysed = source_run(
        REPO / ".pingstore", identity, stage="analyse", experiment="exp154"
    )
    reference = analysed.record["inputs"]["compute"]
    computed = source_run(
        REPO / ".pingstore",
        reference["run_id"],
        stage="compute",
        experiment="exp154",
        reference=reference,
    )
    result = load_json(analysed.file("results.json"))
    if (
        result["configuration"] != computed.record["execution"]["configuration"]
        or result["configuration"]["protocol"] != protocol
    ):
        raise ValueError("analysis protocol/configuration differs from compute")
    return analysed, computed, result


def learning_history(events, pre, post):
    """Align fixed-cue error and weight change after the same encounter."""
    trials = np.asarray([event.get("step", event["trial"]) for event in events])
    mse = np.asarray(
        [
            np.nan if event.get("phase") == "prenatal" else event["validation_mse"]
            for event in events
        ]
    )
    prenatal = np.asarray([event.get("phase") == "prenatal" for event in events])
    if (
        not events
        or not np.array_equal(trials, np.arange(1, len(events) + 1))
        or not np.isfinite(mse[~prenatal]).all()
        or np.any(mse < 0)
        or pre.shape != post.shape
        or post.ndim != 3
        or post.shape[:2] != (len(events), 2)
        or not np.isfinite(pre).all()
        or not np.isfinite(post).all()
    ):
        raise ValueError("invalid aligned learning history")
    # Treat each pathway as a flattened vector: its L2 norm is the matrix
    # Frobenius norm, rather than the matrix spectral norm.
    norms = np.linalg.norm(post, axis=-1)
    ratio = np.divide(
        norms[:, 0],
        norms[:, 1],
        out=np.full(len(events), np.nan),
        where=norms[:, 1] > 0,
    )
    return dict(
        trials=trials,
        rmse=np.sqrt(mse),
        weight_norms=norms,
        inherited_norms=np.linalg.norm(pre[0], axis=-1),
        norm_ratio=ratio,
        births=np.asarray(
            [
                event.get("step", event["trial"]) - 0.5
                for event in events
                if event["birth"]
            ]
        ),
        phases=[event.get("phase", "adult") for event in events],
    )


def learning_figure(history, diagnostic_cues):
    """Align prediction error, overall pathway strengths and their norm ratio."""
    theme.apply()
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9, 6.4),
        sharex=True,
        height_ratios=(1.2, 1, 1),
        layout="constrained",
    )
    trials, rmse = history["trials"], history["rmse"]
    axes[0].plot(trials, rmse, color=theme.INK_BLACK)
    axes[0].set(
        ylabel="Prediction RMSE\n(energy units)",
        title=f"Prediction error ({diagnostic_cues} fixed cues)",
        ylim=(0, float(np.nanmax(rmse)) * 1.12),
    )
    for index, label, color in (
        (0, "E→I", theme.INK_BLACK),
        (1, "I→E", theme.DEEP_RED),
    ):
        axes[1].plot(
            trials, history["weight_norms"][:, index], color=color, label=label
        )
    axes[1].set(
        ylabel="Weight L2 norm",
        title="Synaptic strengths",
        ylim=(0, float(history["weight_norms"].max()) * 1.1),
    )
    axes[1].legend(loc="center right", ncol=2)
    axes[2].plot(trials, history["norm_ratio"], color=theme.INK_BLACK)
    axes[2].set(
        xlabel="Lifecycle step",
        ylabel="EI / IE",
        title="Weight-norm ratio",
    )
    axes[2].set_ylim(bottom=0)
    for ax in axes:
        start = 0
        phases = history["phases"]
        for end in range(1, len(phases) + 1):
            if end == len(phases) or phases[end] != phases[start]:
                color = {
                    "prenatal": theme.GREY_MID,
                    "juvenile": theme.AMBER,
                    "adult": theme.PAPER,
                }[phases[start]]
                ax.axvspan(
                    start + 0.5,
                    end + 0.5,
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                    zorder=0,
                )
                start = end
        for birth in history["births"]:
            ax.axvline(
                birth,
                color=theme.GREY_MID,
                linestyle="--",
                linewidth=1,
                zorder=4,
                clip_on=False,
            )
        ax.set_xlim(0.5, len(trials) + 0.5)
    theme.label_panels(axes)
    return fig


def present(*, lifetime, run_id=None):
    analysed, computed, result = load_analysis(lifetime, "lifetime")
    cfg = result["configuration"]
    if (
        cfg.get("schema") != "exp154/v6"
        or cfg.get("initialization") != "paired-e/v1"
        or cfg["mapping_signs"] != [1]
        or len(cfg["seeds"]) != 1
    ):
        raise ValueError("requires the single mapping and neutral initialization")
    if cfg["profile"] != "pilot":
        raise ValueError("report presentation requires the full lifetime pilot")
    if not result["integrity_pass"]:
        raise ValueError("lifetime bookkeeping integrity failed")
    with stage_run(
        REPO,
        "exp154",
        "present",
        inputs=dict(analyse=analysed),
        configuration=cfg,
        run_id=run_id,
    ) as run:
        payload = replay_payload(computed, result)
        write_json_atomic(run.export / "lifecycle.json", payload)
        events = payload["conditions"][0]["events"]
        history = learning_history(
            events,
            np.asarray([e["weights"] for e in events]),
            np.asarray([e["weights_after"] for e in events]),
        )
        figure = learning_figure(history, cfg["validation_cues"])
        for extension in ("svg", "png"):
            figure.savefig(run.export / f"learning.{extension}", dpi=240)
        plt.close(figure)
        write_json_atomic(
            run.export / "numbers.json",
            dict(
                lifetime=result,
                learning=dict(
                    first_post_encounter_rmse=float(
                        history["rmse"][np.isfinite(history["rmse"])][0]
                    ),
                    final_rmse=float(history["rmse"][-1]),
                    minimum_rmse=float(np.nanmin(history["rmse"])),
                    diagnostic_cues=cfg["validation_cues"],
                    births=len(history["births"]),
                    inherited_weight_norms=history["inherited_norms"].tolist(),
                    final_weight_norms=history["weight_norms"][-1].tolist(),
                    final_norm_ratio=(
                        float(history["norm_ratio"][-1])
                        if np.isfinite(history["norm_ratio"][-1])
                        else None
                    ),
                ),
            ),
        )
        for source in (analysed, computed):
            source.check_unchanged()
    print(f"Presentation: {run.run_id}", flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifetime", required=True)
    parser.add_argument("--run-id")
    present(**vars(parser.parse_args()))
