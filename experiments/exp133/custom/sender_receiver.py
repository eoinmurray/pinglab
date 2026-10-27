"""Define and validate the reported reciprocal coupling-onset diagnostic.

Compute transfers the selected map's matrices and input realization; analysis
and presentation validate its binary recordings and saved native start parameters."""

import hashlib

import numpy as np
from experiments.exp133 import recipe
from pingstore.library import PingstoreError, load_json
from snnlab import lang as snn
from snnlab.sim import NetworkSnapshot

from . import evidence, inputs

SCHEMA = "exp133.sender-receiver/v1"
N_INPUT = 1024
DURATION_MS = 1500.0
ONSET_MS = 500.0
BASELINE_START_MS = 100.0
COUPLING_US = 0.09
DRIVE_HZ = {"S": 23.0, "R": 27.0}
INPUT_GAIN = 9.2


def input_scales():
    return {f"PING_{name}_input_to_E": INPUT_GAIN for name in ("S", "R")}


def author_network():
    net = snn.Network("exp133_sender_receiver", dt=recipe.DT_MS * snn.ms)
    circuits = {}
    for name in ("S", "R"):
        drive = net.input(
            f"drive_{name}",
            shape=("time", "batch", N_INPUT),
            signal_type="spikes",
            unit="spike",
        )
        circuits[name] = recipe.add_ping(net, f"PING_{name}", drive)
    for source, target in (("S", "R"), ("R", "S")):
        for pop in ("E", "I"):
            net.connect(
                circuits[source].E.spikes,
                getattr(circuits[target], pop).excitatory,
                name=f"PING_{source}_E_to_PING_{target}_{pop}",
                synapse=snn.AMPA(tau=2 * snn.ms),
                weight=snn.LowerClampedNormal(
                    COUPLING_US,
                    0.0,
                    initial_zero_fraction=0.9,
                    zeroing="exact_k",
                ),
                constraint=snn.NonNegative(),
                connection="feedback",
                delay=2 * snn.ms,
                modulation=snn.StepGain(at=ONSET_MS * snn.ms, before=0.0, after=1.0),
            )
    for name in ("S", "R"):
        for pop in ("E", "I"):
            net.expose(getattr(circuits[name], pop).spikes, name=f"{name}_{pop}")
    return snn.compile(net, target=None)


def configuration():
    return dict(
        schema=SCHEMA,
        duration_ms=DURATION_MS,
        onset_ms=ONSET_MS,
        dt_ms=recipe.DT_MS,
        network_seed=recipe.NETWORK_SEED,
        input_seeds=list(recipe.INPUT_SEEDS),
        input_rates_hz=DRIVE_HZ,
        input_projection_gain=INPUT_GAIN,
        input_channels=N_INPUT,
        coupling_us=COUPLING_US,
        direction="reciprocal S_E to R_E/R_I and R_E to S_E/S_I",
        parameter_transfer="A to S, B to R; saved 0.09 uS grid matrices; retain both directions",
        bundle_digest=author_network().manifest["graph_digest"],
        coupling_schedule="native StepGain: zero before 500 ms, one from 500 ms; continuous state",
        phase="unwrapped S-minus-R E-volley phase in cycles; full common valid window",
    )


def make_inputs():
    shape = (round(DURATION_MS / recipe.DT_MS), 1, N_INPUT)
    return {
        f"drive_{name}": (
            np.random.default_rng(seed).random(shape, dtype=np.float32)
            < DRIVE_HZ[name] * recipe.DT_MS / 1000
        ).astype(np.uint8)
        for name, seed in zip(("S", "R"), recipe.INPUT_SEEDS, strict=True)
    }


def recording(source):
    with np.load(
        source.export / "sender-receiver--spikes.npz", allow_pickle=False
    ) as raw:
        values = dict(raw)
    shapes = {
        f"{name}_{pop}": (
            round(DURATION_MS / recipe.DT_MS),
            1,
            count,
        )
        for name in ("S", "R")
        for pop, count in (("E", recipe.N_E), ("I", recipe.N_I))
    }
    if set(values) != set(shapes) or any(
        value.shape != shapes[key]
        or value.dtype != np.uint8
        or not ((value == 0) | (value == 1)).all()
        for key, value in values.items()
    ):
        raise PingstoreError("incomplete sender–receiver spike recording")
    return values


def transferred_parameters(source):
    """Map the selected reciprocal condition onto the renamed reciprocal network."""
    evidence.configuration(source)
    original = {
        name: tensor.numpy()
        for name, tensor in NetworkSnapshot.read(
            source.file(
                f"{recipe.label(COUPLING_US)}-execution", "start", "snapshot.json"
            )
        )
        .load_parameters()
        .items()
    }
    values = {}
    for target, origin in (("S", "A"), ("R", "B")):
        for suffix in ("input_to_E", "E_to_I", "I_to_E"):
            values[f"PING_{target}_{suffix}.weight"] = original[
                f"PING_{origin}_{suffix}.weight"
            ]
    for target_source, target_dest, origin_source, origin_dest in (
        ("S", "R", "A", "B"),
        ("R", "S", "B", "A"),
    ):
        for pop in ("E", "I"):
            values[f"PING_{target_source}_E_to_PING_{target_dest}_{pop}.weight"] = (
                original[
                    f"PING_{origin_source}_E_to_PING_{origin_dest}_{pop}_K_E{pop}.weight"
                ]
            )
    if any(
        values[f"PING_{n}_input_to_E.weight"].shape != (N_INPUT, recipe.N_E)
        for n in ("S", "R")
    ):
        raise PingstoreError("sender–receiver parameter source has wrong input count")
    return values


def compute_contract(source):
    cfg = configuration()
    pins = source.record["inputs"]
    if source.record["execution"].get("configuration") != cfg or set(pins) != {
        "parameter_reference"
    }:
        raise PingstoreError("sender–receiver compute recipe mismatch")
    pin = pins["parameter_reference"]
    reference = inputs.source(
        source.directory.parents[2], pin["run_id"], "compute", reference=pin
    )
    transferred = transferred_parameters(reference)
    manifest = load_json(source.file("sender-receiver.bundle", "manifest.json"))
    if manifest["graph_digest"] != cfg["bundle_digest"]:
        raise PingstoreError("sender–receiver bundle differs from recipe")
    recording(source)
    expected_inputs = make_inputs()
    with np.load(source.export / "inputs.npz", allow_pickle=False) as raw:
        if set(raw.files) != set(expected_inputs) or any(
            not np.array_equal(raw[key], value)
            for key, value in expected_inputs.items()
        ):
            raise PingstoreError("sender–receiver inputs differ from recipe")
    descriptors = {
        key: dict(
            shape=list(value.shape),
            dtype="float32",
            sha256=hashlib.sha256(value.astype(np.float32).tobytes()).hexdigest(),
        )
        for key, value in expected_inputs.items()
    }
    metadata = load_json(
        source.file("sender-receiver-execution", "start", "snapshot.json")
    )
    actual = {
        row["input_id"]: row["value"] for row in metadata["request"]["input_bindings"]
    }
    if (
        actual != descriptors
        or metadata["request"]["options"]
        .get("inference_overrides", {})
        .get("projection_scales")
        != input_scales()
    ):
        raise PingstoreError("native sender–receiver inputs or input gains mismatch")
    values = {
        name: tensor.numpy()
        for name, tensor in NetworkSnapshot.read(
            source.file("sender-receiver-execution", "start", "snapshot.json")
        )
        .load_parameters()
        .items()
    }
    if set(values) != set(transferred) or any(
        not np.allclose(values[key], value, rtol=1e-6, atol=1e-9)
        for key, value in transferred.items()
    ):
        raise PingstoreError(
            "sender–receiver parameters differ from selected grid matrices"
        )
    for origin, target in (("S", "R"), ("R", "S")):
        for pop in ("E", "I"):
            weights = values[f"PING_{origin}_E_to_PING_{target}_{pop}.weight"]
            if (
                not np.allclose(weights.sum(axis=0), COUPLING_US, rtol=0, atol=1e-7)
                or not ((weights != 0).sum(axis=0) == 8).all()
            ):
                raise PingstoreError(
                    "sender–receiver coupling weight or fan-in mismatch"
                )
    return cfg
