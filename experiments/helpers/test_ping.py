"""Historical COBA timing and portable bundle authentication."""

import pytest
from experiments.helpers.ping import compile_legacy_coba
from snnlab import lang
from snnlab.sim import plan_graph


def network(duration_ms):
    net = lang.Network("historical_coba", dt=0.1 * lang.ms)
    population = net.population(
        "cells", size=1, neuron=lang.COBA_LIF(refractory=duration_ms * lang.ms)
    )
    net.output("spikes", population.spikes)
    return net


@pytest.mark.parametrize("duration_ms,count,held", [(0, 0, 0), (0.1, 1, 0), (1.2, 12, 11)])
def test_historical_timing_survives_authenticated_roundtrip(tmp_path, duration_ms, count, held):
    bundle = compile_legacy_coba(network(duration_ms))
    bundle.write(tmp_path / "bundle")
    loaded = lang.load_bundle(tmp_path / "bundle")
    assert loaded.graph == bundle.graph
    assert loaded.graph["refractory_timing"] == "legacy_coba"
    neuron = loaded.graph["populations"][0]["neuron"]
    assert neuron["refractory"] == {"value": duration_ms, "unit": "ms"}
    assert "refractory_steps" not in neuron
    timing = plan_graph(loaded.graph).refractory["cells"]
    assert timing["refractory_steps"] == count
    assert timing["held_steps"] == held


def test_historical_authoring_still_rejects_inexact_durations():
    with pytest.raises(ValueError, match="whole number of steps"):
        compile_legacy_coba(network(0.15))
