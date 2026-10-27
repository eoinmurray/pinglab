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
from experiments.exp154.recipe import individual_configuration
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
            event["conception"] = bool(
                event["birth"] and not cfg["genome"]["prenatal_windows"]
            )
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
        food_encounters=max(
            event["trial"] for row in conditions for event in row["events"]
        ),
        initialization=cfg["initialization"],
        mapping=dict(sign=1, formula="2x - 1"),
        conditions=conditions,
        initial_energy=cfg["initial_energy"],
        metabolic_cost=cfg["metabolic_cost"],
        exploration_probability=cfg["exploration_probability"],
        scope=summary["scope"],
        source_compute=computed.reference,
        **(
            dict(reproduction=cfg["evolution"]["reproduction"])
            if cfg.get("evolution", {}).get("reproduction")
            else {}
        ),
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
        conceptions=np.asarray(
            [
                event.get("step", event["trial"]) - 0.5
                for event in events
                if event.get("conception", event["birth"])
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
        axes[1].axhline(
            history["inherited_norms"][index],
            color=color,
            linestyle="--",
            linewidth=1,
            label=f"{label} inherited reset",
        )
    axes[1].set(
        ylabel="Weight L2 norm",
        title="Synaptic strengths",
        ylim=(0, float(history["weight_norms"].max()) * 1.1),
    )
    axes[1].legend(loc="center right", ncol=2)
    axes[2].plot(trials, history["norm_ratio"], color=theme.INK_BLACK)
    inherited_ei, inherited_ie = history["inherited_norms"]
    if inherited_ie > 0:
        axes[2].axhline(
            inherited_ei / inherited_ie,
            color=theme.INK_BLACK,
            linestyle="--",
            linewidth=1,
            label="Inherited reset",
        )
        axes[2].legend(loc="upper right")
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
        for conception in history["conceptions"]:
            ax.axvline(
                conception,
                color=theme.GREY_MID,
                linestyle="--",
                linewidth=1,
                zorder=4,
                clip_on=False,
            )
        ax.set_xlim(0.5, len(trials) + 0.5)
    theme.label_panels(axes)
    return fig


def evolution_outcome_figure(result, *, score_name, life_name, label):
    """Compare selection progress with post-selection independent lives."""
    theme.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    generations = result["generations"]
    x = [row["generation"] for row in generations]
    for legend_label, reducer, color in (
        ("Best", np.max, theme.INK_BLACK),
        ("Population mean", np.mean, theme.DEEP_RED),
    ):
        axes[0].plot(
            x,
            [
                reducer([score[score_name] for score in row["ranked"]])
                for row in generations
            ],
            marker="o",
            label=legend_label,
            color=color,
        )
    axes[0].set(
        xlabel="Generation",
        ylabel=label,
        title="Selection lives",
        xticks=x,
    )
    founder_score = next(
        row[score_name]
        for row in generations[0]["ranked"]
        if row["candidate"] == result["founder"]
    )
    axes[0].axhline(
        founder_score,
        color=theme.GREY_MID,
        linestyle="--",
        linewidth=1,
        label="Founder",
    )
    axes[0].legend()
    for index, (role, color) in enumerate(
        (("founder", theme.DEEP_RED), ("winner", theme.INK_BLACK))
    ):
        lives = [
            row
            for row in result["conditions"]
            if row["cohort"] == "fresh" and row["role"] == role
        ]
        values = [row[life_name] for row in lives]
        axes[1].scatter(
            index + np.linspace(-0.08, 0.08, len(values)), values, color=color
        )
        axes[1].hlines(
            np.mean(values), index - 0.2, index + 0.2, color=color, linewidth=2
        )
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Founder", "Winner"],
        title="Fresh lives (no selection)",
        ylabel=label,
    )
    largest_mean = max(
        row[score_name] for generation in generations for row in generation["ranked"]
    )
    largest_fresh = max(
        row[life_name] for row in result["conditions"] if row["cohort"] == "fresh"
    )
    for ax in axes:
        upper = max(1, largest_mean, largest_fresh)
        ax.set_ylim(-0.04 * upper, upper * 1.12)
    fig.suptitle(
        f"Evolution {result['configuration']['profile']} — "
        f"{result['configuration']['lifetime_presentations']}-encounter horizon"
    )
    theme.label_panels(axes)
    return fig


def evolution_survival_figure(result):
    return evolution_outcome_figure(
        result,
        score_name="mean_survival",
        life_name="longest_observed_lifetime",
        label="Mean food encounters survived",
    )


def evolution_reproduction_figure(result):
    return evolution_outcome_figure(
        result,
        score_name="mean_offspring",
        life_name="offspring",
        label="Offspring per life",
    )


def plot_inherited_strength(ax, x, genomes, founder, pathway):
    """Label recorded normal centres or historical uniform endpoints accurately."""
    if "recurrent_normal" in founder:
        means = [g["recurrent_normal"][pathway]["mean"] for g in genomes]
        ax.plot(x, means, color=theme.INK_BLACK, marker="o")
        ax.axhline(
            founder["recurrent_normal"][pathway]["mean"],
            color=theme.DEEP_RED,
            linestyle="--",
            linewidth=1,
        )
        ax.text(
            0.04,
            0.95,
            f"Fixed σ = {founder['recurrent_normal'][pathway]['std']:.4g}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
    else:
        for endpoint, color in ((0, theme.INK_BLACK), (1, theme.DEEP_RED)):
            ax.plot(
                x,
                [g["recurrent_ranges"][pathway][endpoint] for g in genomes],
                color=color,
                marker="o",
                label="Lower" if endpoint == 0 else "Upper",
            )
            ax.axhline(
                founder["recurrent_ranges"][pathway][endpoint],
                color=color,
                linestyle="--",
                linewidth=1,
            )
        ax.legend()


def evolution_parameter_figure(result):
    """Show the best-ranked inherited settings, never acquired synaptic weights."""
    theme.apply()
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.8), layout="constrained")
    individuals = {row["candidate"]: row["genome"] for row in result["individuals"]}
    genomes = [
        individuals[row["ranked"][0]["candidate"]] for row in result["generations"]
    ]
    founder = individuals[result["founder"]]
    x = np.arange(1, len(genomes) + 1)
    traits = [
        (
            "Initial E→I mean"
            if "recurrent_normal" in founder
            else "Initial E→I range",
            "EI",
        ),
        (
            "Initial I→E mean"
            if "recurrent_normal" in founder
            else "Initial I→E range",
            "IE",
        ),
        ("Prenatal learning rate", "prenatal_learning_rate"),
        ("Juvenile learning rate", "juvenile_learning_rate"),
        ("Adult learning rate", "adult_learning_rate"),
        ("Prenatal activity target", "homeostatic_target_rate"),
        ("Prenatal windows", "prenatal_windows"),
        ("Juvenile encounters", "juvenile_encounters"),
    ]
    for ax, (label, key) in zip(axes.flat, traits, strict=True):
        if key in {"EI", "IE"}:
            plot_inherited_strength(ax, x, genomes, founder, key)
        else:
            ax.plot(x, [g[key] for g in genomes], color=theme.INK_BLACK, marker="o")
            ax.axhline(founder[key], color=theme.DEEP_RED, linestyle="--", linewidth=1)
        ax.set(title=label, xlabel="Generation", xticks=x)
    fig.suptitle(
        f"Evolution {result['configuration']['profile']} — best-ranked genome each generation"
    )
    return fig


def evolution_overview_figure(result):
    """Align population-mean survival and the eight adaptive inherited scalars."""
    theme.apply()
    individuals = {row["candidate"]: row["genome"] for row in result["individuals"]}
    founder = individuals[result["founder"]]
    if "recurrent_normal" not in founder:
        raise ValueError("evolution overview requires normally sampled initial weights")
    generations = result["generations"]
    populations = [[individuals[key] for key in row["members"]] for row in generations]
    x = [row["generation"] for row in generations]
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, layout="constrained")
    survival = [
        np.mean([score["mean_survival"] for score in row["ranked"]])
        for row in generations
    ]
    founding_survival = next(
        row["mean_survival"]
        for row in generations[0]["ranked"]
        if row["candidate"] == result["founder"]
    )
    axes[0, 0].plot(
        x, survival, color=theme.INK_BLACK, marker="o", label="Population mean"
    )
    axes[0, 0].axhline(
        founding_survival,
        color=theme.DEEP_RED,
        linestyle="--",
        linewidth=1,
        label="Founder",
    )
    axes[0, 0].set(
        title="Food encounters survived",
        ylabel="Encounters",
        ylim=(0, max(survival + [founding_survival]) * 1.15),
    )
    axes[0, 0].legend(loc="best")
    traits = [
        ("Initial E→I mean", "EI", "Underlying normal mean"),
        ("Initial I→E mean", "IE", "Underlying normal mean"),
        ("Prenatal learning rate", "prenatal_learning_rate", "Learning rate"),
        ("Juvenile learning rate", "juvenile_learning_rate", "Learning rate"),
        ("Adult learning rate", "adult_learning_rate", "Learning rate"),
        ("Prenatal activity target", "homeostatic_target_rate", "Normalized activity"),
        ("Prenatal duration", "prenatal_windows", "Windows"),
        ("Juvenile duration", "juvenile_encounters", "Food encounters"),
    ]
    for ax, (title, key, unit) in zip(list(axes.flat)[1:], traits, strict=True):
        if key in {"EI", "IE"}:
            values = [
                np.mean([g["recurrent_normal"][key]["mean"] for g in population])
                for population in populations
            ]
            initial = founder["recurrent_normal"][key]["mean"]
            title += f"\n(fixed σ = {founder['recurrent_normal'][key]['std']:.4g})"
        else:
            values = [
                np.mean([g[key] for g in population]) for population in populations
            ]
            initial = founder[key]
        ax.plot(x, values, color=theme.INK_BLACK, marker="o")
        ax.axhline(initial, color=theme.DEEP_RED, linestyle="--", linewidth=1)
        ax.set(title=title, ylabel=unit)
    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xlim(x[0] - 0.2, x[-1] + 0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Generation")
    settings = result["configuration"]["evolution"]
    fig.suptitle(
        f"Evolution over {len(generations)} generations — population averages\n"
        f"{'Up to ' if 'reproduction' in settings else ''}{settings['population']} genomes × {len(settings['search_seeds'])} selection lives; "
        "dashed lines show the founding genome"
    )
    theme.label_panels(list(axes.flat))
    return fig


def present_evolution(run, analysed, computed, result):
    """Keep a deterministically selected fresh-seed winner life as a diagnostic replay."""
    cfg = result["configuration"]
    winner = next(
        row for row in result["individuals"] if row["candidate"] == result["winner"]
    )
    row = min(
        (
            row
            for row in result["conditions"]
            if row["cohort"] == "fresh" and row["role"] == "winner"
        ),
        key=lambda row: row["seed"],
    )
    resolved = individual_configuration(cfg, winner["genome"], row["seed"])
    payload = replay_payload(
        computed, dict(configuration=resolved, conditions=[row], scope=result["scope"])
    )
    write_json_atomic(run.export / "evolution-lifecycle.json", payload)
    result["replay"] = dict(seed=row["seed"], directory=row["directory"])
    figures = [
        ("evolution-survival", evolution_survival_figure),
        ("evolution-parameters", evolution_parameter_figure),
    ]
    if "recurrent_normal" in cfg["genome"]:
        figures.append(("evolution-overview", evolution_overview_figure))
    if "reproduction" in cfg["evolution"]:
        figures.append(("evolution-reproduction", evolution_reproduction_figure))
    for stem, function in figures:
        figure = function(result)
        for extension in ("svg", "png"):
            figure.savefig(run.export / f"{stem}.{extension}", dpi=240)
        plt.close(figure)
    for source in (analysed, computed):
        source.check_unchanged()


def present(*, lifetime, evolution=None, run_id=None):
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
    inputs = dict(analyse=analysed)
    evolved = None
    if evolution is not None:
        evolution_analysis, evolution_compute, evolved = load_analysis(
            evolution, "evolution"
        )
        if not evolved["integrity_pass"]:
            raise ValueError("evolution integrity failed")
        inputs["evolution"] = evolution_analysis
    with stage_run(
        REPO,
        "exp154",
        "present",
        inputs=inputs,
        configuration=cfg,
        run_id=run_id,
    ) as run:
        (run.export / "circuit.svg").write_bytes(
            computed.file("circuit.svg").read_bytes()
        )
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
        if evolved is not None:
            present_evolution(run, evolution_analysis, evolution_compute, evolved)
        write_json_atomic(
            run.export / "numbers.json",
            dict(
                lifetime=result,
                evolution=evolved,
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
    parser.add_argument("--evolution")
    parser.add_argument("--run-id")
    present(**vars(parser.parse_args()))
