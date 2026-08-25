from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1] / "exp098.py"


def test_scout_is_a_single_file_experiment() -> None:
    source = EXPERIMENT.read_text()
    assert 'CONDITIONS = ("baseline", "w-ei-zero", "input-ramp")' in source
    assert '"dt_ms": DT_MS' in source
    assert '"ramp_input_hz": [20.0, 160.0]' in source
    assert "exp098_support" not in source


def test_scout_uses_graph_native_execution() -> None:
    source = EXPERIMENT.read_text()
    assert "from tools import snnlang as snn" in source
    assert 'executor="graph"' in source
    assert ".canvas" not in source
