"""Tests for the snn tool — the demolab manifest contract + CLI wiring.

The heavy path (running the pinglab CLI) is covered by the CLI's own suite under
src/cli/tests; here we test the contract that the tool adds on top: write_output
validation and the argparse surface. Kept off the filesystem where possible.
"""
import json

import pytest

from tools.snn import tool


def test_write_output_rejects_metric_absent_from_output(tmp_path):
    # A manifest can never lie: a headline metric not in the metrics dict raises.
    with pytest.raises(ValueError, match="not present in output.json"):
        tool.write_output(tmp_path, {"rate_e_hz": 6.4}, {"headline_metrics": ["rate_i_hz"]})


def test_write_output_writes_the_file_set(tmp_path):
    tool.write_output(
        tmp_path, {"rate_e_hz": 6.4, "rate_i_hz": 37.3},
        {"headline_metrics": ["rate_e_hz", "rate_i_hz"]},
    )
    out = json.loads((tmp_path / "output.json").read_text())
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert out["rate_e_hz"] == 6.4 and out["rate_i_hz"] == 37.3
    assert man["headline_metrics"] == ["rate_e_hz", "rate_i_hz"]


def test_write_output_missing_figure_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        tool.write_output(
            tmp_path, {"rate_e_hz": 6.4},
            {"headline_metrics": ["rate_e_hz"], "headline_figure": "nope.png"},
        )


def test_provenance_has_expected_keys():
    p = tool._run_provenance()
    assert {"commit", "dirty", "generated_at"} <= p.keys()
    assert isinstance(p["dirty"], bool)


def test_sim_parser_defaults():
    args = tool.build_parser().parse_args(["sim", "--n-inh", "64", "--ei-ratio", "4.0"])
    assert args.command == "sim"
    assert args.n_inh == 64 and args.ei_ratio == 4.0
    assert args.n_hidden == 1024 and args.seed == 42  # defaults
    assert args.func is tool.cmd_sim
