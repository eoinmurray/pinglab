"""Independent sensory signals and matched native emitted-spike controls."""

from time import perf_counter

import numpy as np
import torch
from experiments.exp136 import compute as fixed
from experiments.exp136 import recipe
from experiments.helpers.network_artifacts import snapshot_parameters
from pingstore.stages import source_run, stage_run
from scipy.ndimage import gaussian_filter1d
from snnlab.sim import (
    DenseArrayBinding,
    DenseSpikeReplay,
    ExecutionSpec,
    ReplaySpikes,
    SnapshotSpec,
    build,
    save_spike_replay,
    simulate,
)


def sensory_signal(seed, cfg):
    settings = cfg["signal"]
    steps = round(cfg["duration_ms"] / settings["bin_ms"])
    # Warm-up padding keeps the finite trial boundary from defining the signal.
    values = np.random.default_rng(seed).normal(size=steps + 200)
    values = gaussian_filter1d(values, settings["smoothing_ms"] / settings["bin_ms"])
    values = values[100 : 100 + steps]
    values = np.tanh(values / max(values.std(), 1e-8))
    values -= values.mean()
    values /= max(1.0, abs(values).max())
    return values.astype(np.float32)


def make_task_inputs(cfg, unit):
    """Cue-order pairs share signals and every background/cue random draw."""
    total = round(cfg["duration_ms"] / cfg["dt_ms"])
    width = round(cfg["segment_ms"] / cfg["dt_ms"])
    repeat = round(cfg["signal"]["bin_ms"] / cfg["dt_ms"])
    lanes, signals, batches = [], [], []
    for trial in unit["trials"]:
        local = dict(
            cfg,
            input_seeds={
                k: v + trial["noise_offset"] for k, v in cfg["input_seeds"].items()
            },
        )
        base = fixed.make_inputs(local)
        target = np.stack(
            [sensory_signal(trial["signal_seed"] + j, cfg) for j in (0, 1)], axis=-1
        )
        for j, sender in enumerate(("S1", "S2")):
            uniforms = np.random.default_rng(local["input_seeds"][sender]).random(
                (total, 1, cfg["input_channels"]), dtype=np.float32
            )
            rates = cfg["background_hz"][sender] + cfg["signal"][
                "amplitude_hz"
            ] * np.repeat(target[:, j], repeat)
            base[f"{sender}_sensory_drive"] = (
                uniforms < rates[:, None, None] * cfg["dt_ms"] / 1000
            ).astype(np.uint8)
        active = base["cue_S1_drive"][width : 2 * width].copy()
        for order in cfg["cue_orders"]:
            drives = {k: v for k, v in base.items() if not k.startswith("cue_")}
            for segment, sender in enumerate(order, 1):
                cue = np.zeros((total, 1, cfg["cue_input_channels"]), dtype=np.uint8)
                cue[segment * width : (segment + 1) * width] = active
                drives[f"cue_{sender}_drive"] = cue
            lanes.append(dict(**trial, cue_order=order))
            signals.append(target)
            batches.append(drives)
    return (
        {k: np.concatenate([b[k] for b in batches], axis=1) for k in batches[0]},
        np.stack(signals, axis=1),
        lanes,
    )


def jitter_spikes(spikes, cfg, seed):
    """Independent permutations preserve each cell's exact count in every block."""
    width = round(cfg["jitter"]["block_ms"] / cfg["dt_ms"])
    if len(spikes) % width:
        raise ValueError("jitter blocks must cover the complete trial")
    blocks = spikes.reshape(-1, width, *spikes.shape[1:])
    result = np.random.default_rng(seed).permuted(blocks, axis=1)
    if not np.array_equal(result.sum(axis=1), blocks.sum(axis=1)):
        raise ValueError("timing control changed blockwise spike counts")
    return result.reshape(spikes.shape)


def execute_attention(run, model, bundle, values, cfg):
    """Execute declared trial batches with native snapshots for every condition."""
    for number, unit in enumerate(cfg["units"]):
        drives, signals, lanes = make_task_inputs(cfg, unit)
        unit["lanes"] = lanes
        np.savez_compressed(
            run.export / f"{unit['unit']}--inputs.npz", **drives, signals=signals
        )
        bindings = tuple(
            DenseArrayBinding(k, torch.from_numpy(v).float()) for k, v in drives.items()
        )
        del drives
        original = None
        for condition in cfg["conditions"]:
            expected = fixed.condition_parameters(
                values, "zero" if condition == "zero" else "coupled"
            )
            with torch.no_grad():
                for key, parameter in model.parameter_map().items():
                    parameter.copy_(torch.from_numpy(expected[key]))
            checkpoint = run.scratch / "parameters.pt"
            torch.save(model.state_dict(), checkpoint)
            interventions = []
            if condition == "jittered":
                for j, sender in enumerate(("S1", "S2")):
                    scrambled = jitter_spikes(
                        original[f"{sender}_E_spikes"],
                        cfg,
                        cfg["jitter"]["seed"] + number * 10 + j,
                    )
                    replay = DenseSpikeReplay.from_tensor(
                        torch.from_numpy(scrambled), dt_ms=cfg["dt_ms"]
                    )
                    save_spike_replay(
                        run.export / f"{unit['unit']}--{sender}-replay.npz", replay
                    )
                    interventions.append(ReplaySpikes(f"{sender}_E", replay))
            label = f"{unit['unit']}-{condition}"
            started = perf_counter()
            result = simulate(
                ExecutionSpec(
                    kind="infer",
                    graph=bundle.graph,
                    seed=cfg["network_seed"],
                    device="cpu",
                    checkpoint=checkpoint,
                    input_bindings=bindings,
                    interventions=tuple(interventions),
                    snapshot=SnapshotSpec(
                        run.export / f"{label}-execution",
                        stage="both",
                        parameters=True,
                    ),
                )
            )
            spikes = {
                k: v.detach().cpu().numpy().astype(np.uint8)
                for k, v in result.outputs.items()
            }
            for stage in ("start", "end"):
                fixed.verify_snapshot(
                    run.export / f"{label}-execution" / stage, expected
                )
            if condition == "coupled":
                original = spikes
            elif condition == "zero":
                if any(
                    not np.array_equal(v, original[k])
                    for k, v in spikes.items()
                    if not k.startswith("R_")
                ):
                    raise ValueError("zero coupling changed sender/cue outputs")
                for population in ("E", "I"):
                    for lane in range(0, len(lanes), 2):
                        if not np.array_equal(
                            spikes[f"R_{population}_spikes"][:, lane],
                            spikes[f"R_{population}_spikes"][:, lane + 1],
                        ):
                            raise ValueError(
                                "cue order affected the disconnected receiver"
                            )
            else:
                for j, sender in enumerate(("S1", "S2")):
                    expected_spikes = jitter_spikes(
                        original[f"{sender}_E_spikes"],
                        cfg,
                        cfg["jitter"]["seed"] + number * 10 + j,
                    )
                    if not np.array_equal(
                        spikes[f"{sender}_E_spikes"], expected_spikes
                    ):
                        raise ValueError("native replay did not replace emitted spikes")
            np.savez_compressed(run.export / f"{label}--spikes.npz", **spikes)
            seconds = perf_counter() - started
            run.record["execution"].setdefault("simulations", {})[label] = dict(
                seconds=seconds,
                batch_size=len(lanes),
                interventions=result.metrics.get("inference_interventions"),
            )
            print(
                f"Signals {number + 1}/{len(cfg['units'])}, {condition}: {seconds:.1f}s",
                flush=True,
            )
            del result, spikes
        del bindings, original


def compute_attention(repo, identity, *, run_id=None):
    torch.set_num_threads(1)
    source = source_run(
        repo / ".pingstore", identity, stage="compute", experiment=recipe.SLUG
    )
    cfg = recipe.attention_configuration()
    values = snapshot_parameters(source, cfg["parameter_unit"], "start")
    bundle = recipe.author_network()
    model = build(
        ExecutionSpec(
            kind="build", graph=bundle.graph, seed=cfg["network_seed"], device="cpu"
        )
    ).model
    if set(values) != set(model.parameter_map()):
        raise ValueError("parameter source differs from the fixed circuit")
    with stage_run(
        repo,
        recipe.SLUG,
        "compute",
        inputs={"parameters": source},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        recipe.write_network_bundle(bundle, run.export / "network.bundle")
        execute_attention(run, model, bundle, values, cfg)
        run.record["execution"]["paired_parameters_and_inputs_verified"] = True
    return run.run_id
