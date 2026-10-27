"""Validate native parameters and recorded capacity/lifetime measurements.

Analysis checks feedback gating, energy bookkeeping, genuine death and exact
birth resets. Rejected hidden outcomes are used only for offline diagnostics.
Every invocation consumes one explicit completed compute run.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
from experiments.exp154.recipe import (
    SLUG,
    author_network,
    individual_configuration,
    make_model,
    next_population,
    rank_candidates,
    reproductive_population,
)
from pingstore.library import source_run, stage_run, write_json_atomic
from snnlab.sim import NetworkSnapshot

GROUPS = {
    "Sensory": ("sensory_E.weight", "sensory_I.weight"),
    "E→E": ("EE.weight",),
    "E→I": ("EI.weight",),
    "I→E": ("IE.weight",),
    "I→I": ("II.weight",),
    "Readout": ("value_weight",),
}


def parameter_changes(initial, final):
    initial = {
        name: np.asarray(value)
        for name, value in NetworkSnapshot.read(initial).load_parameters().items()
    }
    final = {
        name: np.asarray(value)
        for name, value in NetworkSnapshot.read(final).load_parameters().items()
    }
    changes = {}
    for group, names in GROUPS.items():
        difference = np.concatenate(
            [(final[name] - initial[name]).ravel() for name in names]
        )
        changes[group] = float(np.sqrt(np.mean(difference**2)))
    frozen = [name for name in initial if name not in {"EI.weight", "IE.weight"}]
    frozen_unchanged = all(
        np.array_equal(initial[name], final[name]) for name in frozen
    )
    return changes, frozen_unchanged, final


def settling(data, cfg):
    tail = round(cfg["probe_tail_ms"] / cfg["dt_ms"])
    rates = [data["E_rate"], data["I_rate"]]
    finite_bounded = all(
        np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1
        for value in rates
    )
    step_rate = max(
        float(np.max(np.abs(np.diff(value[-tail:], axis=0)))) / cfg["dt_ms"]
        for value in rates
    )
    return dict(
        finite_bounded=bool(finite_bounded),
        max_tail_step_rate_per_ms=step_rate,
        settled=bool(step_rate <= cfg["stability_step_rate_per_ms"]),
    )


def condition_summary(source, row, cfg):
    directory = Path(row["directory"])
    with np.load(source.file(directory / "data.npz")) as data:
        error = float(np.mean((data["final_prediction"] - data["expected_grid"]) ** 2))
        frozen_error = float(
            np.mean((data["initial_prediction"] - data["expected_grid"]) ** 2)
        )
        stability = settling(data, cfg)
        finite_data = all(np.isfinite(data[name]).all() for name in data.files)
    changes, frozen_unchanged, parameters = parameter_changes(
        source.file(directory.parent / "snapshot-initial" / "start" / "snapshot.json"),
        source.file(directory / "snapshot-final" / "end" / "snapshot.json"),
    )
    weights_bounded = all(
        np.all((parameters[name] >= 0) & (parameters[name] <= cfg["weight_upper"]))
        for name in ("EI.weight", "IE.weight")
    )
    selective_update = frozen_unchanged and all(
        changes[name] > 0 for name in ("E→I", "I→E")
    )
    accurate = error < cfg["pass_mse"] and error < frozen_error
    passed = (
        accurate
        and selective_update
        and weights_bounded
        and stability["finite_bounded"]
        and stability["settled"]
        and finite_data
    )
    return {
        **row,
        "mse": error,
        "frozen_mse": frozen_error,
        "accuracy_pass": accurate,
        "parameter_rms_change": changes,
        "frozen_unchanged": frozen_unchanged,
        "selective_update": selective_update,
        "weights_bounded": bool(weights_bounded),
        "stability": stability,
        "finite_data": bool(finite_data),
        "passed": bool(passed),
    }


def activity_checks(data, initial, cfg, events, *, internal_drives=None):
    """Cross-check native recordings against saved weights and decision readouts.

    Recurrent current at a step uses the previous rate state. The first step
    uses the configured reset rate; sensory channels are held at x, 1-x, 1.
    """
    count, steps = len(events), round(cfg["presentation_ms"] / cfg["dt_ms"])
    rates = {}
    for name, size in (("E", cfg["e_cells"]), ("I", cfg["i_cells"])):
        value = data[f"replay_{name}_rate"]
        if value.shape != (count, steps, size) or not np.isfinite(value).all():
            raise ValueError("invalid native rate recording")
        rates[name] = value
    previous = {
        name: np.concatenate(
            (np.full((count, 1, value.shape[-1]), cfg["initial_rate"]), value[:, :-1]),
            axis=1,
        )
        for name, value in rates.items()
    }
    if internal_drives is None:
        cues = np.asarray([e["cue"] for e in events])
        sensory = np.broadcast_to(
            np.column_stack((cues, 1 - cues, np.ones(count)))[:, None],
            (count, steps, 3),
        )
        internal = np.zeros_like(sensory)
    else:
        internal = internal_drives
        sensory = np.zeros_like(internal)
    current_error = 0.0
    for name in (
        "sensory_E",
        "sensory_I",
        "internal_E",
        "internal_I",
        "EE",
        "EI",
        "IE",
        "II",
    ):
        if name.startswith("sensory_"):
            source = sensory
        elif name.startswith("internal_"):
            source = internal
        else:
            source = previous[name[0]]
        if name in ("EI", "IE"):
            index = 0 if name == "EI" else 1
            weights = data["weights_pre"][:, index].reshape(count, source.shape[-1], -1)
            expected = np.einsum("bti,bij->btj", source, weights)
        else:
            expected = np.einsum(
                "bti,ij->btj", source, np.asarray(initial[name + ".weight"])
            )
        recorded = data[f"replay_{name}_current"]
        if recorded.shape != expected.shape or not np.isfinite(recorded).all():
            raise ValueError("invalid native current recording")
        current_error = max(current_error, float(np.max(np.abs(expected - recorded))))
    tail = round(cfg["readout_tail_ms"] / cfg["dt_ms"])
    readout = np.einsum(
        "bi,ij->bj",
        rates["E"][:, -tail:].mean(axis=1),
        np.asarray(initial["value_weight"]),
    )[:, 0]
    prediction_error = float(
        np.max(np.abs(readout - [e["prediction"] for e in events]))
    )
    bounded = all(np.all((value >= 0) & (value <= 1)) for value in rates.values())
    return dict(
        passed=bool(
            bounded
            and current_error <= cfg["activity_validation_atol"]
            and prediction_error <= cfg["activity_validation_atol"]
        ),
        rates_bounded=bool(bounded),
        max_current_error_na=current_error,
        max_readout_error=prediction_error,
        frames_per_encounter=steps,
    )


def development_summary(source, directory, initial, cfg, food_events):
    """Check inheritance, prenatal isolation, homeostatic loss and native currents."""
    inherited = np.stack(
        [np.asarray(initial[n]).ravel() for n in ("EI.weight", "IE.weight")]
    )
    births = [event for event in food_events if event["birth"]]
    count = cfg["genome"]["prenatal_windows"]
    if count == 0:
        return dict(passed=True, windows=0), {
            event["life"]: inherited for event in births
        }
    events = json.loads(source.file(directory / "development-events.json").read_text())
    with np.load(source.file(directory / "development.npz")) as archive:
        data = {name: archive[name] for name in archive.files}
    pre, post = data["weights_pre"], data["weights_post"]
    if (
        len(events) != count * len(births)
        or pre.shape != post.shape
        or pre.shape != (len(events), *inherited.shape)
    ):
        raise ValueError("invalid prenatal coverage")
    checks = dict(
        inheritance=True, continuity=True, schedule=True, loss=True, inputs=True
    )
    developed = {}
    for birth in births:
        indices = [
            i for i, event in enumerate(events) if event["life"] == birth["life"]
        ]
        if len(indices) != count:
            raise ValueError("invalid prenatal lifetime coverage")
        previous = inherited
        for window, index in enumerate(indices, 1):
            event = events[index]
            checks["inheritance"] &= window != 1 or np.array_equal(
                pre[index], inherited
            )
            checks["continuity"] &= np.array_equal(pre[index], previous)
            checks["schedule"] &= (
                event["window"] == window
                and event["before_trial"] == birth["trial"]
                and event["phase"] == "prenatal"
                and event["seed"]
                == cfg["seeds"][0] + cfg["development_seed_offset"] + birth["life"]
                and event["learning_rate"] == cfg["genome"]["prenatal_learning_rate"]
            )
            target = cfg["genome"]["homeostatic_target_rate"]
            loss = (
                sum(
                    np.mean(
                        (data[f"replay_{name}_rate"][index].mean(axis=0) - target) ** 2
                    )
                    for name in ("E", "I")
                )
                / 2
            )
            checks["loss"] &= np.isclose(event["loss"], loss, rtol=0, atol=1e-12)
            previous = post[index]
        developed[birth["life"]] = previous
    drives = data["internal_drives"]
    checks["inputs"] &= bool(
        np.isfinite(drives).all() and np.all((drives >= 0) & (drives <= 1))
    )
    bounded = bool(
        np.isfinite(pre).all()
        and np.isfinite(post).all()
        and min(pre.min(), post.min()) >= 0
        and max(pre.max(), post.max()) <= cfg["weight_upper"]
    )
    neural = activity_checks(data, initial, cfg, events, internal_drives=drives)
    return dict(
        passed=bool(all(checks.values()) and bounded and neural["passed"]),
        checks={key: bool(value) for key, value in checks.items()},
        weights_bounded=bounded,
        windows=len(events),
        neural_recording=neural,
        initial_loss=float(events[0]["loss"]),
        final_loss=float(events[-1]["loss"]),
    ), developed


def reproduction_checks(events, cfg):
    """Independently reconstruct eligibility, litter size and its energy charge."""
    policy = cfg.get("evolution", {}).get("reproduction")
    if policy is None:
        return None
    passed, reproduced, offspring, cost = True, False, 0, 0.0
    reproduction_age = None
    for event in events:
        if event["birth"]:
            reproduced = False
        food_energy = (
            event["energy_before"]
            - cfg["metabolic_cost"]
            + event["accepted"] * event["delta"]
        )
        count = 0
        if (
            not reproduced
            and event["age"] > cfg["genome"]["juvenile_encounters"]
            and event["age"] >= policy["minimum_age_encounters"]
            and food_energy > 0
        ):
            affordable = np.floor(
                (food_energy - policy["reserve_energy"]) / policy["cost_per_offspring"]
            )
            count = int(np.clip(affordable, 0, policy["maximum_offspring"]))
        charge = count * policy["cost_per_offspring"]
        passed &= (
            type(event.get("offspring")) is int
            and event["offspring"] == count
            and event.get("reproduced") == bool(count)
            and event.get("reproduction_cost") == charge
            and np.isclose(
                event.get("energy_before_reproduction", np.nan),
                food_energy,
                rtol=0,
                atol=1e-12,
            )
            and np.isclose(
                event["energy_after"], food_energy - charge, rtol=0, atol=1e-12
            )
        )
        if count:
            reproduction_age = event["age"]
        reproduced |= count > 0
        offspring += count
        cost += charge
    return dict(
        passed=bool(passed),
        offspring=offspring,
        reproduction_age=reproduction_age,
        reproduction_energy_cost=cost,
    )


def lifetime_summary(source, row, cfg):
    directory = Path(row["directory"])
    events = json.loads(source.file(directory / "events.json").read_text())
    with np.load(source.file(directory / "data.npz")) as archive:
        data = {name: archive[name] for name in archive.files}
    changes, fixed, final = parameter_changes(
        source.file(directory.parent / "snapshot-initial/start/snapshot.json"),
        source.file(directory / "snapshot-final/end/snapshot.json"),
    )
    initial = NetworkSnapshot.read(
        source.file(directory.parent / "snapshot-initial/start/snapshot.json")
    ).load_parameters()
    development, developed = development_summary(
        source, directory, initial, cfg, events
    )
    pre, post = data["weights_pre"], data["weights_post"]
    if (
        not events
        or (
            len(events) > cfg["lifetime_presentations"]
            if cfg.get("stop_at_death")
            else len(events) != cfg["lifetime_presentations"]
        )
        or pre.shape != post.shape
        or pre.shape != (len(events), 2, cfg["e_cells"] * cfg["i_cells"])
    ):
        raise ValueError("invalid lifetime event/weight coverage")
    checks = dict(
        energy=True,
        feedback=True,
        choice=True,
        births=True,
        weight_continuity=True,
        rejection_freeze=True,
        phases=True,
    )
    reproductive = reproduction_checks(events, cfg)
    if reproductive is not None:
        checks["reproduction"] = (
            reproductive["passed"]
            and row.get("offspring") == reproductive["offspring"]
            and row.get("reproduction_age") == reproductive["reproduction_age"]
        )
    if cfg.get("stop_at_death"):
        checks["single_life"] = bool(
            not any(event["death"] for event in events[:-1])
            and (events[-1]["death"] or len(events) == cfg["lifetime_presentations"])
            and row["survival"] == len(events)
            and row["remaining_energy"] == max(0.0, events[-1]["energy_after"])
        )
    # Neutrality is an inherited initial condition. Prenatal plasticity can
    # break the paired readout symmetry before any food has been experienced.
    if (
        cfg.get("initialization") == "paired-e/v1"
        and not cfg["genome"]["prenatal_windows"]
    ):
        checks["neutral_undeveloped_birth"] = True
    life, age = 0, 0
    for index, event in enumerate(events):
        birth = index == 0 or events[index - 1]["death"]
        life += int(birth)
        age = 1 if birth else age + 1
        before = cfg["initial_energy"] if birth else events[index - 1]["energy_after"]
        expected = before - cfg["metabolic_cost"] + event["accepted"] * event["delta"]
        if reproductive is not None:
            expected -= event.get("reproduction_cost", 0)
        checks["energy"] &= bool(
            np.isclose(event["energy_before"], before, rtol=0, atol=1e-12)
            and np.isclose(event["energy_after"], expected, rtol=0, atol=1e-12)
            and event["death"] == (expected <= 0)
        )
        checks["feedback"] &= (
            event["feedback"] == event["accepted"]
            and event["updated"] == event["accepted"]
        )
        checks["choice"] &= event["accepted"] == (
            event["prediction"] > cfg.get("decision_threshold", 0)
            or data["exploration"][index] < cfg["exploration_probability"]
        )
        if birth and "neutral_undeveloped_birth" in checks:
            checks["neutral_undeveloped_birth"] &= (
                abs(event["prediction"]) <= cfg["decision_threshold"]
            )
        checks["births"] &= (
            event["birth"] == birth and event["life"] == life and event["age"] == age
        )
        phase = "juvenile" if age <= cfg["genome"]["juvenile_encounters"] else "adult"
        checks["phases"] &= (
            event["phase"] == phase
            and event["learning_rate"] == cfg["genome"][phase + "_learning_rate"]
            and event["phase_age"]
            == (
                age
                if phase == "juvenile"
                else age - cfg["genome"]["juvenile_encounters"]
            )
        )
        checks["weight_continuity"] &= np.array_equal(
            pre[index], developed[life] if birth else post[index - 1]
        )
        if not event["updated"]:
            checks["rejection_freeze"] &= np.array_equal(pre[index], post[index])
        if (
            event["trial"] != index + 1
            or event["cue"] != data["cues"][index]
            or not np.isclose(event["delta"], row["sign"] * (2 * event["cue"] - 1))
        ):
            raise ValueError("invalid recorded encounter identity")
    if cfg.get("initialization") == "paired-e/v1":
        checks["neutral_initial_grid"] = bool(
            np.max(np.abs(data["initial_prediction"])) <= cfg["decision_threshold"]
        )
    checks = {key: bool(value) for key, value in checks.items()}
    bounded = bool(
        np.isfinite(pre).all()
        and np.isfinite(post).all()
        and pre.min() >= 0
        and post.min() >= 0
        and pre.max() <= cfg["weight_upper"]
        and post.max() <= cfg["weight_upper"]
    )
    final_weights = np.stack([final[n].ravel() for n in ("EI.weight", "IE.weight")])
    native_matches = np.array_equal(post[-1], final_weights)
    neural = activity_checks(data, initial, cfg, events)
    ages = [e["age"] for e in events if e["death"]]
    return dict(
        **(
            row
            | (
                {key: value for key, value in reproductive.items() if key != "passed"}
                if reproductive is not None
                else {}
            )
        ),
        checks=checks,
        frozen_unchanged=fixed,
        weights_bounded=bounded,
        native_final_matches=bool(native_matches),
        parameter_rms_change=changes,
        stability=settling(data, cfg),
        neural_recording=neural,
        development=development,
        phase_counts={
            phase: sum(e["phase"] == phase for e in events)
            for phase in ("juvenile", "adult")
        },
        max_birth_prediction_abs=max(
            abs(e["prediction"]) for e in events if e["birth"]
        ),
        integrity_pass=bool(
            all(checks.values())
            and bounded
            and fixed
            and native_matches
            and neural["passed"]
            and development["passed"]
        ),
        deaths=sum(e["death"] for e in events),
        births=sum(e["birth"] for e in events),
        completed_lifetime_lengths=ages,
        surviving_age=0 if events[-1]["death"] else events[-1]["age"],
        final_energy=events[-1]["energy_after"],
        accepted=sum(e["accepted"] for e in events),
        harmful_consumed=sum(e["accepted"] and e["delta"] < 0 for e in events),
        beneficial_consumed=sum(e["accepted"] and e["delta"] > 0 for e in events),
        final_mse=float(
            np.mean((data["final_prediction"] - data["expected_grid"]) ** 2)
        ),
        net_food_energy=sum(e["delta"] for e in events if e["accepted"]),
        longest_observed_lifetime=max(e["age"] for e in events),
    )


def evolution_life_summary(source, row, cfg, individual):
    """Authenticate matched random inputs and fresh native initialization."""
    resolved = individual_configuration(cfg, individual["genome"], row["seed"])
    directory = Path(row["directory"])
    rng = np.random.default_rng(row["seed"] + cfg["training_seed_offset"])
    cues = rng.uniform(size=cfg["lifetime_presentations"])
    draws = rng.uniform(size=len(cues))
    with np.load(source.file(directory / "data.npz")) as data:
        count = len(data["cues"])
        matched = np.array_equal(data["cues"], cues[:count]) and np.array_equal(
            data["exploration"], draws[:count]
        )
    saved = NetworkSnapshot.read(
        source.file(directory.parent / "snapshot-initial/start/snapshot.json")
    ).load_parameters()
    expected = make_model(author_network(resolved, row["seed"]), resolved, row["seed"])
    inherited = all(
        np.array_equal(saved[name], value.detach().cpu().numpy())
        for name, value in expected.parameter_map().items()
    )
    summary = lifetime_summary(source, row, resolved)
    summary.update(matched_inputs=matched, fresh_inherited_parameters=inherited)
    summary["integrity_pass"] &= bool(matched and inherited and row["sign"] == 1)
    return summary


def evolution_scores(conditions, members):
    """Recompute fitness from validated terminal events, never diagnostic MSE."""
    scores = []
    for candidate in members:
        lives = [row for row in conditions if row["candidate"] == candidate]
        scores.append(
            dict(
                candidate=candidate,
                mean_survival=float(
                    np.mean([row["longest_observed_lifetime"] for row in lives])
                ),
                mean_energy=float(
                    np.mean([max(0.0, row["final_energy"]) for row in lives])
                ),
            )
        )
        if lives and "offspring" in lives[0]:
            scores[-1]["mean_offspring"] = float(
                np.mean([row["offspring"] for row in lives])
            )
    return rank_candidates(scores)


def evolution_inputs(source, rows, cfg):
    """Validate generation coverage and unique recorded evaluation identities."""
    recorded = json.loads(source.file("evolution.json").read_text())
    settings = cfg["evolution"]
    reproduction = "reproduction" in settings
    if (
        recorded["schema"]
        != ("exp154.evolution/v2" if reproduction else "exp154.evolution/v1")
        or not 1 <= len(recorded["generations"]) <= settings["generations"]
        or (
            not reproduction and len(recorded["generations"]) != settings["generations"]
        )
    ):
        raise ValueError("invalid evolution generation coverage")
    individuals = {row["candidate"]: row for row in recorded["individuals"]}
    if len(individuals) != len(recorded["individuals"]):
        raise ValueError("duplicate evolution candidate")
    if cfg["evaluated_genomes"] != {
        key: row["genome"] for key, row in individuals.items()
    }:
        raise ValueError("evolution genomes differ from recorded configuration")
    indexed = {
        (
            row["cohort"],
            row["generation"],
            row["role"],
            row["candidate"],
            row["seed"],
        ): row
        for row in rows
    }
    if len(indexed) != len(rows):
        raise ValueError("duplicate evolution evaluation")
    return recorded, individuals, indexed


def evolution_summary(source, rows, cfg):
    """Replay genealogy and selection with an independent fitness reconstruction."""
    recorded, individuals, indexed = evolution_inputs(source, rows, cfg)
    settings = cfg["evolution"]
    reproduction = "reproduction" in settings
    founder = dict(
        candidate="g001-c000", parent=None, created_generation=1, genome=cfg["genome"]
    )
    rng = np.random.default_rng(settings["seed"])
    parents, observed, generations, conditions = [founder], {}, [], []
    population = next_population(parents, settings, rng, 1)
    offspring, termination = [], "generation_limit"
    for number, generation in enumerate(recorded["generations"], 1):
        if number > 1 and not reproduction:
            population = next_population(parents, settings, rng, number)
        members = [row["candidate"] for row in population]
        if generation["generation"] != number or generation["members"] != members:
            raise ValueError("invalid evolution population/elite retention")
        if any(individuals.get(row["candidate"]) != row for row in population):
            raise ValueError("invalid inherited genome mutation or parentage")
        observed.update({row["candidate"]: row for row in population})
        evaluated = evolution_evaluations(
            source,
            indexed,
            cfg,
            population,
            cohort="search",
            generation=number,
            roles=["candidate"],
            seeds=settings["search_seeds"],
        )
        ranked = evolution_scores(evaluated, members)
        if ranked != generation["ranked"]:
            raise ValueError("recorded evolution ranking differs from fitness")
        conditions.extend(evaluated)
        generations.append(dict(generation=number, members=members, ranked=ranked))
        parents = [
            individuals[row["candidate"]] for row in ranked[: settings["elites"]]
        ]
        if reproduction:
            population, newborns = reproductive_population(
                population,
                ranked,
                evaluated,
                settings,
                rng,
                number,
                admit=number < settings["generations"],
            )
            offspring.extend(newborns)
            if number < settings["generations"] and not any(
                child["admitted"] for child in newborns
            ):
                termination = "no_offspring"
                if number != len(recorded["generations"]):
                    raise ValueError("evolution continued without offspring")
    if reproduction and (
        recorded.get("offspring") != offspring
        or recorded.get("termination") != termination
        or (
            termination == "generation_limit"
            and len(generations) != settings["generations"]
        )
    ):
        raise ValueError("invalid reproductive offspring or termination")
    winner = generations[-1]["ranked"][0]["candidate"]
    if (
        observed != individuals
        or recorded["winner"] != winner
        or recorded["founder"] != founder["candidate"]
    ):
        raise ValueError("invalid evolution winner or candidate coverage")
    fresh = []
    for role, individual in (("winner", individuals[winner]), ("founder", founder)):
        fresh.extend(
            evolution_evaluations(
                source,
                indexed,
                cfg,
                [individual],
                cohort="fresh",
                generation=generations[-1]["generation"],
                roles=[role],
                seeds=settings["fresh_seeds"],
            )
        )
    if indexed:
        raise ValueError("unexpected evolution evaluations")
    conditions.extend(fresh)
    return dict(
        **recorded,
        conditions=conditions,
        fresh=[
            dict(
                role=role,
                **evolution_scores(
                    [row for row in fresh if row["role"] == role], [candidate]
                )[0],
            )
            for role, candidate in (
                ("winner", winner),
                ("founder", founder["candidate"]),
            )
        ],
        integrity_pass=all(row["integrity_pass"] for row in conditions),
    )


def evolution_evaluations(
    source, indexed, cfg, population, *, cohort, generation, roles, seeds
):
    summaries = []
    for individual in population:
        for role in roles:
            for seed in seeds:
                key = (cohort, generation, role, individual["candidate"], seed)
                if key not in indexed:
                    raise ValueError("missing evolution evaluation")
                summaries.append(
                    evolution_life_summary(source, indexed.pop(key), cfg, individual)
                )
    return summaries


def analyse(*, source, run_id=None):
    computed = source_run(REPO / ".pingstore", source, stage="compute", experiment=SLUG)
    cfg = computed.record["execution"]["configuration"]
    rows = json.loads(computed.file("conditions.json").read_text())
    if cfg["protocol"] == "evolution":
        result = evolution_summary(computed, rows, cfg)
        result.update(
            configuration=cfg,
            scope=(
                "Evolution pilot on the positive mapping. Age- and energy-gated reproduction, actual offspring admission, reproductive-output selection and fresh-seed evaluation; no frozen/oracle controls or neural activity cost."
                if "reproduction" in cfg["evolution"]
                else "Evolution pilot on the positive mapping. Independent lives, survival selection and fresh-seed evaluation; no frozen/oracle controls or neural activity cost."
            ),
        )
    else:
        result = standard_summary(computed, rows, cfg)
    with stage_run(
        REPO,
        SLUG,
        "analyse",
        inputs=dict(compute=computed),
        configuration=cfg,
        run_id=run_id,
    ) as run:
        write_json_atomic(run.export / "results.json", result)
        computed.check_unchanged()
    print(f"Analysis: {run.run_id}", flush=True)
    return run.run_id


def standard_summary(computed, rows, cfg):
    expected = {(seed, sign) for seed in cfg["seeds"] for sign in cfg["mapping_signs"]}
    if (
        len(rows) != len(expected)
        or {(row["seed"], row["sign"]) for row in rows} != expected
    ):
        raise ValueError("incomplete condition coverage")
    if cfg["protocol"] == "capacity":
        conditions = [condition_summary(computed, row, cfg) for row in rows]
        counts = {
            str(sign): sum(r["passed"] for r in conditions if r["sign"] == sign)
            for sign in cfg["mapping_signs"]
        }
        passed = (
            all(n >= cfg["required_seeds_per_sign"] for n in counts.values())
            if cfg["profile"] == "pilot"
            else None
        )
        result = dict(
            conditions=conditions,
            pass_counts=counts,
            passed=passed,
            decision="SMOKE ONLY" if passed is None else "PASS" if passed else "FAIL",
        )
    else:
        conditions = [lifetime_summary(computed, row, cfg) for row in rows]
        result = dict(
            conditions=conditions,
            integrity_pass=all(r["integrity_pass"] for r in conditions),
        )
    result.update(
        configuration=cfg,
        scope="One fixed abstract genome; prenatal homeostasis, juvenile food learning and reduced adult plasticity. No evolution or selection. Validation snapshots never control behaviour.",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    analyse(**vars(parser.parse_args()))
