"""Read an explicit exp033 analysis; no numerical solving or remeasurement."""

import numpy as np
from experiments.exp033 import evidence as mf_evidence
from experiments.exp033 import inputs as mf_inputs
from experiments.exp033 import measurements as mf_measurements
from experiments.exp033 import recipe as mf_recipe
from pingstore.contracts import PingstoreError, load_json

from . import inputs


def source(repo, identity, frequencies, reference=None):
    analysis = inputs.source(
        repo, identity, "analyse", experiment="exp033", reference=reference
    )
    cfg = mf_inputs.configuration(analysis)
    if cfg != mf_recipe.configuration(version=2):
        raise PingstoreError("theory refresh requires the adopted exp033 recipe")
    refs = analysis.record["inputs"]
    if (
        set(refs) != {"compute", "frequencies"}
        or refs["frequencies"] != frequencies.reference
    ):
        raise PingstoreError("exp033 theory pins different frequency evidence")
    compute = inputs.source(
        repo,
        refs["compute"]["run_id"],
        "compute",
        experiment="exp033",
        reference=refs["compute"],
    )
    if mf_inputs.configuration(compute) != cfg or compute.record["inputs"]:
        raise PingstoreError("exp033 theory and compute recipes disagree")
    numbers = load_json(analysis.file("results.json"))
    keys = {
        "tau_E_ms",
        "tau_I_ms",
        "tau_AMPA_ms",
        "tau_GABA_ms",
        "W_tilde_EI",
        "W_tilde_IE",
        "dV_inh_mV",
        "dV_exc_mV",
        "sigma_V_mV",
        "cell_E",
        "cell_I",
    }
    if numbers.get("slug") != "exp033" or numbers.get("config") != {
        k: cfg[k] for k in keys
    }:
        raise PingstoreError("exp033 theory numbers disagree with their recipe")
    result = numbers["results"]
    coords = mf_evidence.read(analysis.export)
    grid = np.linspace(*cfg["drive_grid"])
    mf_measurements.validate_continuation(
        {"sweep": coords["sweep"], "hopf": result["hopf"]}, grid
    )
    if (
        len(coords["sweep"]) != len(grid)
        or not result["hopf"]
        or not result["criticality"]
    ):
        raise PingstoreError("exp033 theory reference is incomplete")
    freq = result["frequency_vs_tau_gaba"]
    medians = {
        str(k): v
        for k, v in mf_measurements.spiking_medians(
            load_json(frequencies.file("results.json"))
        ).items()
    }
    if (
        freq["spiking_exp041"] != medians
        or [r["tau_gaba_ms"] for r in freq["mean_field"]] != cfg["tau_grid_ms"]
    ):
        raise PingstoreError("exp033 theory frequency measurements disagree")
    return (
        analysis,
        cfg,
        {
            "config": numbers["config"],
            "sweep": coords["sweep"],
            "hopf": result["hopf"],
            "criticality": result["criticality"],
            "frequency_vs_tau_gaba": freq["mean_field"],
            "spiking_exp041": freq["spiking_exp041"],
        },
    )
