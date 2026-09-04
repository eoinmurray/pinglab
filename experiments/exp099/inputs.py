"""Resolve explicit v4 runs and verify their retained scientific configuration."""

from pathlib import Path

from pingstore.contracts import PingstoreError
from pingstore.stages import source_run
from tools.snnviz import load_snnsim_recording  # noqa: TID251

from . import recipe


def source(repo: Path, identity: str, stage: str, *, reference=None):
    return source_run(
        repo / ".pingstore",
        identity,
        stage=stage,
        experiment=recipe.SLUG,
        reference=reference,
    )


def configuration(run) -> dict:
    cfg = run.record["execution"].get("configuration")
    if not isinstance(cfg, dict) or cfg.get("schema") != "exp099.recipe/v1":
        raise PingstoreError("exp099 requires a retained scientific recipe")
    return cfg


def recording(compute):
    cfg = configuration(compute)
    record = load_snnsim_recording(compute.export / "simulation")
    if record.dt_ms != cfg["dt_ms"] or record.duration_ms != cfg["t_ms"]:
        raise PingstoreError("recording timebase disagrees with retained recipe")
    if record.metadata.get("config", {}).get("_simulation_recipe") != cfg.get(
        "simulation"
    ):
        raise PingstoreError(
            "recording input recipe disagrees with retained configuration"
        )
    for name, size in (
        ("spk_e", cfg["n_e"]),
        ("spk_i", cfg["n_i"]),
        ("v_e_1", cfg["n_e"]),
        ("v_i_1", cfg["n_i"]),
        ("ge_e_1", cfg["n_e"]),
        ("gi_e_1", cfg["n_e"]),
    ):
        (value,) = record.require(name)
        if value.shape != (record.steps, size):
            raise PingstoreError(f"recording shape disagrees with recipe: {name}")
    # This experiment always records authenticated weather inputs. Do not
    # reconstruct missing inputs or substitute silent zero backgrounds.
    record.require(
        "input_afferent_shared",
        "input_afferent_e_private",
        "input_afferent_i_private",
        "input_structured_spikes_e",
        "input_structured_spikes_i",
        "input_weather_scale",
        "input_afferent_scale",
        "input_afferent_shared_scale",
    )
    for population in ("e", "i"):
        for channel in ("excitatory", "inhibitory"):
            record.require(
                *(
                    f"input_{channel}_{population}_{kind}"
                    for kind in ("private", "shared", "executed")
                )
            )
    return record
