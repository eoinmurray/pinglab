"""Compute capacity and consumption-gated lifetimes as separate Pingstore runs.

SNNLab owns dynamics, gradients, recordings, constraints and native snapshots.
This module schedules exposures, gates feedback, maintains energy and restores
inherited initialization at birth. Run analyse and present separately.
"""

import argparse
import copy
import sys
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
import torch
from experiments.exp154.custom.snapshot import capture
from experiments.exp154.recipe import (
    author_network,
    configuration,
    developmental_phase,
    encode,
    encode_internal,
    individual_configuration,
    make_model,
    next_population,
    rank_candidates,
    reproductive_population,
    targets,
)
from pingstore.library import stage_run, write_json_atomic
from snnlab.sim import RecordingSpec, SignalRecording

FORWARD_OPTIONS = dict(
    diagnostics=False,
    retain_parameters=False,
    retain_final_state=False,
    retain_runtime_state=False,
)
PROBE_SIGNALS = ("E.rate", "I.rate", "sensory_E.current", "EE.current", "IE.current")


def predict(model, inputs):
    return model(inputs, **FORWARD_OPTIONS).outputs["prediction"].squeeze(-1)


def recorded_forward(model, inputs, cfg):
    """Keep differentiable native outputs alongside detached replay recordings."""
    result = model(
        inputs,
        recording=RecordingSpec(
            tuple(SignalRecording(name) for name in cfg["lifetime_record_signals"])
        ),
        **FORWARD_OPTIONS,
    )
    signals = {
        name.replace(".", "_"): value.detach().cpu().numpy()[:, 0, :].copy()
        for name, value in result.recorded_signals.items()
    }
    return result, signals


def recorded_prediction(model, cue, cfg):
    """Record native rates and projection currents from the decision forward pass."""
    result, signals = recorded_forward(model, encode([cue], cfg), cfg)
    return result.outputs["prediction"].squeeze(-1), signals


def evaluate(model, inputs, expected, grid_count):
    with torch.no_grad():
        prediction = predict(model, inputs).cpu().numpy()
    grid_mse = float(np.mean((prediction[:grid_count] - expected[:grid_count]) ** 2))
    validation_mse = float(
        np.mean((prediction[grid_count:] - expected[grid_count:]) ** 2)
    )
    return prediction, grid_mse, validation_mse


def train_controller(model, cfg, sign, cues, evaluation_inputs, expected, grid_count):
    parameters = model.parameter_map()
    optimizer = torch.optim.AdamW(
        [parameters[name] for name in ("EI.weight", "IE.weight")],
        lr=cfg["learning_rate"],
        weight_decay=0,
    )
    prediction, loss, validation_loss = evaluate(
        model, evaluation_inputs, expected, grid_count
    )
    initial_prediction = prediction[:grid_count].copy()
    history = [[0, loss, validation_loss]]
    best_loss = validation_loss
    best_state = copy.deepcopy(model.state_dict())
    best_presentation = 0
    for start in range(0, len(cues), cfg["batch_size"]):
        batch = cues[start : start + cfg["batch_size"]]
        target = torch.as_tensor(
            targets(batch, sign), dtype=getattr(torch, cfg["dtype"])
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = predict(model, encode(batch, cfg))
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        model.enforce_constraints()
        completed = start + len(batch)
        if completed % cfg["evaluation_every"] == 0 or completed == len(cues):
            prediction, loss, validation_loss = evaluate(
                model, evaluation_inputs, expected, grid_count
            )
            history.append([completed, loss, validation_loss])
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                best_presentation = completed
    return dict(
        initial_prediction=initial_prediction,
        final_prediction=prediction[:grid_count].copy(),
        history=np.asarray(history),
        best_state=best_state,
        best_presentation=best_presentation,
    )


def probe(model, cfg):
    recording = RecordingSpec(tuple(SignalRecording(name) for name in PROBE_SIGNALS))
    with torch.no_grad():
        result = model(
            encode(cfg["probe_cues"], cfg, duration_ms=cfg["probe_ms"]),
            recording=recording,
            **FORWARD_OPTIONS,
        )
    return {
        name.replace(".", "_"): value.cpu().numpy()
        for name, value in result.recorded_signals.items()
    }


def compute_condition(
    bundle, cfg, seed, sign, directory, initial_directory, grid, validation
):
    directory.mkdir(parents=True)
    model = make_model(bundle, cfg, seed)
    if not initial_directory.exists():
        capture(
            model,
            bundle,
            initial_directory,
            cfg,
            seed,
            stage="start",
            selection="initial frozen control",
        )
    cues = np.random.default_rng(seed + cfg["training_seed_offset"]).uniform(
        size=cfg["training_presentations"]
    )
    evaluation_cues = np.concatenate((grid, validation))
    expected = targets(evaluation_cues, sign)
    trained = train_controller(
        model, cfg, sign, cues, encode(evaluation_cues, cfg), expected, len(grid)
    )
    capture(
        model,
        bundle,
        directory / "snapshot-final" / "end",
        cfg,
        seed,
        stage="end",
        selection="final update; reported checkpoint",
    )
    probes = probe(model, cfg)
    final_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(trained.pop("best_state"))
    capture(
        model,
        bundle,
        directory / "snapshot-best-validation" / "end",
        cfg,
        seed,
        stage="end",
        selection=f"minimum separate-validation MSE at presentation {trained['best_presentation']}; evidence only",
    )
    model.load_state_dict(final_state)
    np.savez_compressed(
        directory / "data.npz",
        cue_grid=grid,
        expected_grid=targets(grid, sign),
        training_cues=cues,
        validation_cues=validation,
        initial_prediction=trained["initial_prediction"],
        final_prediction=trained["final_prediction"],
        history=trained["history"],
        probe_time_ms=(np.arange(next(iter(probes.values())).shape[0]) + 1)
        * cfg["dt_ms"],
        probe_cues=np.asarray(cfg["probe_cues"]),
        **probes,
    )
    row = dict(
        seed=seed,
        sign=sign,
        directory=directory.name,
        best_validation_presentation=trained["best_presentation"],
    )
    write_json_atomic(directory / "condition.json", row)
    print(
        f"seed={seed} mapping={sign:+d} final MSE={trained['history'][-1, 1]:.5f} "
        f"frozen={trained['history'][0, 1]:.5f}",
        flush=True,
    )
    return row


def optimizer_for(model, cfg, *, phase="juvenile"):
    return torch.optim.AdamW(
        [model.parameter_map()[name] for name in ("EI.weight", "IE.weight")],
        lr=cfg["genome"][phase + "_learning_rate"],
        weight_decay=0,
    )


def cross_weights(model):
    return np.stack(
        [
            model.parameter_map()[name].detach().cpu().numpy().ravel().copy()
            for name in ("EI.weight", "IE.weight")
        ]
    )


def consume(prediction, exploration_draw, cfg):
    """Choice uses prediction and exogenous exploration, never hidden outcome."""
    return (
        prediction > cfg["decision_threshold"]
        or exploration_draw < cfg["exploration_probability"]
    )


def energy_after(energy, accepted, delta, cfg):
    return energy - cfg["metabolic_cost"] + int(accepted) * delta


def reproduction_after(energy, age, phase, reproduced, policy):
    """Spend a litter's energy once, after a living adult's food encounter."""
    if policy is None:
        return energy, {}
    count = 0
    if (
        not reproduced
        and phase == "adult"
        and age >= policy["minimum_age_encounters"]
        and energy > 0
    ):
        count = min(
            policy["maximum_offspring"],
            max(
                0,
                int(
                    np.floor(
                        (energy - policy["reserve_energy"])
                        / policy["cost_per_offspring"]
                    )
                ),
            ),
        )
    cost = count * policy["cost_per_offspring"]
    return energy - cost, dict(
        offspring=count,
        reproduced=bool(count),
        reproduction_cost=cost,
        energy_before_reproduction=energy,
    )


def apply_loss(model, optimizer, loss, cfg):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    model.enforce_constraints()


def update_consumed(model, optimizer, prediction, delta, accepted, cfg):
    if accepted:
        apply_loss(model, optimizer, (prediction - delta).square().mean(), cfg)


def prenatal_development(model, cfg, *, life, before_trial, seed):
    """Develop a fresh inherited circuit using internal pulses and no food targets."""
    genome = cfg["genome"]
    rng_seed = seed + cfg["development_seed_offset"] + life
    rng = np.random.default_rng(rng_seed)
    optimizer = optimizer_for(model, cfg, phase="prenatal")
    steps = round(cfg["presentation_ms"] / cfg["dt_ms"])
    time = np.arange(steps)[:, None] / steps
    records = []
    for window in range(genome["prenatal_windows"]):
        drives = rng.uniform(0.3, 1.0, 3) * np.maximum(
            0, np.sin(2 * np.pi * time + rng.uniform(0, 2 * np.pi, 3))
        )
        before = cross_weights(model)
        result, signals = recorded_forward(model, encode_internal(drives, cfg), cfg)
        target = genome["homeostatic_target_rate"]
        loss = (
            sum(
                (result.outputs[name + "_activity"] - target).square().mean()
                for name in ("E", "I")
            )
            / 2
        )
        apply_loss(model, optimizer, loss, cfg)
        records.append(
            dict(
                event=dict(
                    life=life,
                    window=window + 1,
                    before_trial=before_trial,
                    seed=rng_seed,
                    phase="prenatal",
                    learning_rate=genome["prenatal_learning_rate"],
                    loss=float(loss.detach()),
                    mean_e=float(signals["E_rate"].mean()),
                    mean_i=float(signals["I_rate"].mean()),
                    prediction=float(result.outputs["prediction"].detach().item()),
                ),
                weights_pre=before,
                weights_post=cross_weights(model),
                internal_drives=drives,
                signals=signals,
            )
        )
    return records


def run_lifetime(
    model,
    cfg,
    sign,
    cues,
    exploration,
    *,
    recording_sink=None,
    development_sink=None,
    seed=None,
):
    """Record terminal outcomes before resetting at the next encounter's birth.

    Validation targets never drive an update or a decision. Their only purpose
    is retaining an evidence-only best-validation native snapshot.
    """
    inherited = copy.deepcopy(model.state_dict())
    optimizer = optimizer_for(model, cfg)
    energy, life, age = cfg["initial_energy"], 1, 0
    dead, reproduced = False, False
    reproduction = cfg.get("evolution", {}).get("reproduction")
    validation = np.random.default_rng(cfg["validation_seed"]).uniform(
        size=cfg["validation_cues"]
    )
    validation_input = encode(validation, cfg)
    validation_target = targets(validation, sign)
    best_state, best_mse, best_trial = inherited, float("inf"), 0
    events, pre, post = [], [], []
    seed = cfg["seeds"][0] if seed is None else seed
    for trial, (cue, draw) in enumerate(zip(cues, exploration, strict=True)):
        birth = trial == 0 or dead
        if birth:
            model.load_state_dict(inherited)
            energy, age, reproduced = cfg["initial_energy"], 0, False
            if trial:
                life += 1
            development = prenatal_development(
                model, cfg, life=life, before_trial=trial + 1, seed=seed
            )
            if development_sink is not None:
                development_sink.extend(development)
            optimizer = optimizer_for(model, cfg)
        age += 1
        phase = developmental_phase(age, cfg)
        learning_rate = cfg["genome"][phase + "_learning_rate"]
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        pre.append(cross_weights(model))
        prediction, activity = recorded_prediction(model, cue, cfg)
        if recording_sink is not None:
            recording_sink.append(activity)
        value = prediction.detach().item()
        accepted = consume(value, draw, cfg)
        # Outcome is revealed to the learner only inside this consumption gate.
        delta = float(targets([cue], sign)[0])
        update_consumed(model, optimizer, prediction, delta, accepted, cfg)
        after = energy_after(energy, accepted, delta, cfg)
        after, reproductive_event = reproduction_after(
            after, age, phase, reproduced, reproduction
        )
        reproduced |= reproductive_event.get("reproduced", False)
        dead = after <= 0
        post.append(cross_weights(model))
        with torch.no_grad():
            validation_mse = float(
                np.mean(
                    (predict(model, validation_input).cpu().numpy() - validation_target)
                    ** 2
                )
            )
        if validation_mse < best_mse:
            best_mse, best_trial = validation_mse, trial + 1
            best_state = copy.deepcopy(model.state_dict())
        events.append(
            dict(
                trial=trial + 1,
                cue=float(cue),
                delta=delta,
                prediction=value,
                accepted=bool(accepted),
                explored=bool(accepted and value <= cfg["decision_threshold"]),
                feedback=bool(accepted),
                updated=bool(accepted),
                energy_before=energy,
                energy_after=after,
                death=bool(dead),
                birth=bool(birth),
                life=life,
                age=age,
                phase=phase,
                phase_age=(
                    age
                    if phase == "juvenile"
                    else age - cfg["genome"]["juvenile_encounters"]
                ),
                learning_rate=learning_rate,
                validation_mse=validation_mse,
                **reproductive_event,
            )
        )
        energy = after
        if dead and cfg.get("stop_at_death", False):
            break
    return events, np.asarray(pre), np.asarray(post), best_state, best_trial


def lifetime_condition(bundle, cfg, seed, sign, directory, cues, exploration):
    directory.mkdir(parents=True)
    model = make_model(bundle, cfg, seed)
    grid = np.linspace(0, 1, cfg["evaluation_cues"])
    with torch.no_grad():
        initial_prediction = predict(model, encode(grid, cfg)).cpu().numpy()
    recordings, development = [], []
    events, pre, post, best, best_trial = run_lifetime(
        model,
        cfg,
        sign,
        cues,
        exploration,
        recording_sink=recordings,
        development_sink=development,
        seed=seed,
    )
    cues, exploration = cues[: len(events)], exploration[: len(events)]
    capture(
        model,
        bundle,
        directory / "snapshot-final" / "end",
        cfg,
        seed,
        stage="end",
        selection="last encounter before any next birth",
    )
    final = copy.deepcopy(model.state_dict())
    probes = probe(model, cfg)
    model.load_state_dict(best)
    capture(
        model,
        bundle,
        directory / "snapshot-best-validation" / "end",
        cfg,
        seed,
        stage="end",
        selection=f"minimum diagnostic validation MSE at encounter {best_trial}; never deployed",
    )
    model.load_state_dict(final)
    grid = np.linspace(0, 1, cfg["evaluation_cues"])
    with torch.no_grad():
        final_prediction = predict(model, encode(grid, cfg)).cpu().numpy()
    write_json_atomic(directory / "events.json", events)
    write_json_atomic(
        directory / "development-events.json", [row["event"] for row in development]
    )
    if development:
        np.savez_compressed(
            directory / "development.npz",
            weights_pre=np.stack([row["weights_pre"] for row in development]),
            weights_post=np.stack([row["weights_post"] for row in development]),
            internal_drives=np.stack([row["internal_drives"] for row in development]),
            **{
                f"replay_{name}": np.stack(
                    [row["signals"][name] for row in development]
                )
                for name in development[0]["signals"]
            },
        )
    np.savez_compressed(
        directory / "data.npz",
        **{
            f"replay_{name}": np.stack([r[name] for r in recordings])
            for name in recordings[0]
        },
        initial_prediction=initial_prediction,
        weights_pre=pre,
        weights_post=post,
        cues=cues,
        exploration=exploration,
        cue_grid=grid,
        expected_grid=targets(grid, sign),
        final_prediction=final_prediction,
        probe_time_ms=(np.arange(next(iter(probes.values())).shape[0]) + 1)
        * cfg["dt_ms"],
        **probes,
    )
    row = dict(
        seed=seed,
        sign=sign,
        best_validation_presentation=best_trial,
        survival=len(events),
        remaining_energy=max(0.0, events[-1]["energy_after"]),
    )
    if cfg.get("evolution", {}).get("reproduction"):
        row["offspring"] = sum(event["offspring"] for event in events)
        row["reproduction_age"] = next(
            (event["age"] for event in events if event["reproduced"]), None
        )
    print(
        f"seed={seed} mapping={sign:+d}: "
        f"{sum(e['death'] for e in events)} deaths, final energy={events[-1]['energy_after']:.2f}",
        flush=True,
    )
    return row


def compute_standard(run, cfg):
    """Run the original capacity/lifetime conditions without changing their policy."""
    rows = []
    for seed in cfg["seeds"]:
        bundle = author_network(cfg, seed)
        seed_dir = run.export / f"seed-{seed:03d}"
        seed_dir.mkdir()
        bundle.write(seed_dir / "network.bundle", visualise=False)
        if seed == cfg["seeds"][0]:
            bundle.visualise(run.export / "circuit.svg", view="circuit")
        initial = seed_dir / "snapshot-initial" / "start"
        model = make_model(bundle, cfg, seed)
        capture(
            model,
            bundle,
            initial,
            cfg,
            seed,
            stage="start",
            selection="inherited initialization",
        )
        rng = np.random.default_rng(seed + cfg["training_seed_offset"])
        cues = rng.uniform(size=cfg["lifetime_presentations"])
        exploration = rng.uniform(size=len(cues))
        for sign in cfg["mapping_signs"]:
            label = "single"
            if cfg["protocol"] == "capacity":
                directory = seed_dir / label
                row = compute_condition(
                    bundle,
                    cfg,
                    seed,
                    sign,
                    directory,
                    initial,
                    np.linspace(0, 1, cfg["evaluation_cues"]),
                    np.random.default_rng(cfg["validation_seed"]).uniform(
                        size=cfg["validation_cues"]
                    ),
                )
                row["directory"] = str(directory.relative_to(run.export))
                rows.append(row)
            else:
                directory = seed_dir / label
                row = lifetime_condition(
                    bundle, cfg, seed, sign, directory, cues, exploration
                )
                row["directory"] = str(directory.relative_to(run.export))
                rows.append(row)
    write_json_atomic(run.export / "conditions.json", rows)


def evolution_life(run, cfg, individual, seed, *, cohort, generation, role):
    """Build a fresh native circuit; a parent's model/state is never accepted."""
    resolved = individual_configuration(cfg, individual["genome"], seed)
    root = (
        run.export
        / cohort
        / f"generation-{generation:03d}"
        / role
        / individual["candidate"]
        / f"seed-{seed:03d}"
    )
    root.mkdir(parents=True)
    bundle = author_network(resolved, seed)
    bundle.write(root / "network.bundle", visualise=False)
    if not (run.export / "circuit.svg").exists():
        bundle.visualise(run.export / "circuit.svg", view="circuit")
    capture(
        make_model(bundle, resolved, seed),
        bundle,
        root / "snapshot-initial/start",
        resolved,
        seed,
        stage="start",
        selection="fresh inherited initialization; no parent state",
    )
    rng = np.random.default_rng(seed + cfg["training_seed_offset"])
    cues = rng.uniform(size=cfg["lifetime_presentations"])
    exploration = rng.uniform(size=len(cues))
    row = lifetime_condition(
        bundle, resolved, seed, 1, root / "single", cues, exploration
    )
    return dict(
        **row,
        candidate=individual["candidate"],
        cohort=cohort,
        generation=generation,
        role=role,
        directory=str((root / "single").relative_to(run.export)),
    )


def fitness_scores(rows, population):
    """Aggregate independent lives; diagnostic prediction errors are excluded."""
    scores = []
    for individual in population:
        lives = [row for row in rows if row["candidate"] == individual["candidate"]]
        scores.append(
            dict(
                candidate=individual["candidate"],
                mean_survival=float(np.mean([row["survival"] for row in lives])),
                mean_energy=float(np.mean([row["remaining_energy"] for row in lives])),
            )
        )
        if lives and "offspring" in lives[0]:
            scores[-1]["mean_offspring"] = float(
                np.mean([row["offspring"] for row in lives])
            )
    return rank_candidates(scores)


def compute_evolution(run, cfg):
    """Evaluate every generation before creating offspring; fresh seeds come last."""
    settings = cfg["evolution"]
    rng = np.random.default_rng(settings["seed"])
    founder = dict(
        candidate="g001-c000",
        parent=None,
        created_generation=1,
        genome=copy.deepcopy(cfg["genome"]),
    )
    parents, individuals, generations, rows = [founder], {}, [], []
    reproduction = "reproduction" in settings
    population = next_population(parents, settings, rng, 1)
    offspring = []
    termination = "generation_limit"
    for generation in range(1, settings["generations"] + 1):
        if generation > 1 and not reproduction:
            population = next_population(parents, settings, rng, generation)
        individuals.update({row["candidate"]: row for row in population})
        evaluated = [
            evolution_life(
                run,
                cfg,
                individual,
                seed,
                cohort="search",
                generation=generation,
                role="candidate",
            )
            for individual in population
            for seed in settings["search_seeds"]
        ]
        ranked = fitness_scores(evaluated, population)
        rows.extend(evaluated)
        generations.append(
            dict(
                generation=generation,
                members=[row["candidate"] for row in population],
                ranked=ranked,
            )
        )
        parents = [
            individuals[row["candidate"]] for row in ranked[: settings["elites"]]
        ]
        print(f"Generation {generation}: {ranked[0]}", flush=True)
        if reproduction:
            population, newborns = reproductive_population(
                population,
                ranked,
                evaluated,
                settings,
                rng,
                generation,
                admit=generation < settings["generations"],
            )
            offspring.extend(newborns)
            if generation < settings["generations"] and not any(
                child["admitted"] for child in newborns
            ):
                termination = "no_offspring"
                break
    winner = generations[-1]["ranked"][0]["candidate"]
    for role, individual in (("winner", individuals[winner]), ("founder", founder)):
        for seed in settings["fresh_seeds"]:
            rows.append(
                evolution_life(
                    run,
                    cfg,
                    individual,
                    seed,
                    cohort="fresh",
                    generation=generations[-1]["generation"],
                    role=role,
                )
            )
    write_json_atomic(run.export / "conditions.json", rows)
    run.record["execution"]["configuration"]["evaluated_genomes"] = {
        key: individual["genome"] for key, individual in individuals.items()
    }
    write_json_atomic(
        run.export / "evolution.json",
        dict(
            schema="exp154.evolution/v2" if reproduction else "exp154.evolution/v1",
            individuals=list(individuals.values()),
            generations=generations,
            founder=founder["candidate"],
            winner=winner,
            **(
                dict(offspring=offspring, termination=termination)
                if reproduction
                else {}
            ),
        ),
    )


def compute(*, profile="pilot", protocol="lifetime", run_id=None):
    cfg = configuration(profile, protocol)
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    with stage_run(REPO, "exp154", "compute", configuration=cfg, run_id=run_id) as run:
        run.record["execution"].update(
            backend="snnlab.GraphExecutor / torch autograd",
            snnlab_version=version("snnlab"),
        )
        if protocol == "evolution":
            compute_evolution(run, cfg)
        else:
            compute_standard(run, cfg)
        # Every execution retains its required native stage; initial and final
        # snapshots are deliberately separate from the validation selection.
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={
                str(p.parent.parent.relative_to(run.export)): [p.parent.name]
                for p in run.export.rglob("snapshot.json")
            }
        )
    print(f"Compute: {run.run_id}", flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pilot", "smoke"), default="pilot")
    parser.add_argument(
        "--protocol", choices=("capacity", "lifetime", "evolution"), default="lifetime"
    )
    parser.add_argument("--run-id")
    compute(**vars(parser.parse_args()))
