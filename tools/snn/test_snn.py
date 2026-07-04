"""Tests for the snn tool — the demolab manifest contract + passthrough wiring.

The heavy path (actually running the pinglab CLI) is covered by the CLI's own
suite under src/cli/tests; here we test the contract the tool adds on top:
write_output validation, provenance, the subcommand-validating parser, and the
--out-dir passthrough resolution. Kept off the filesystem where possible.
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


def test_parser_accepts_known_subcommands():
    for cmd in tool.SUBCOMMANDS:
        args, rest = tool.build_parser().parse_known_args([cmd, "--t-ms", "300"])
        assert args.command == cmd
        assert rest == ["--t-ms", "300"]  # passthrough args captured verbatim


def test_parser_rejects_unknown_subcommand():
    with pytest.raises(SystemExit):
        tool.build_parser().parse_args(["frobnicate"])


def test_cli_out_dir_honours_caller_out_dir(tmp_path):
    # Space-separated and = forms both resolve; a sweep point keeps its own dir.
    assert tool._cli_out_dir(["--out-dir", "runs/a", "--t-ms", "300"], tmp_path).name == "a"
    assert tool._cli_out_dir(["--out-dir=runs/b"], tmp_path).name == "b"


def test_cli_out_dir_falls_back_to_default(tmp_path):
    default = tmp_path / "_cli"
    assert tool._cli_out_dir(["--t-ms", "300"], default) is default


def test_setup_run_dir_writes_config_and_run_sh(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "TEMP_DIR", tmp_path)
    run_dir, _ = tool.setup_run_dir("sim", ["--t-ms", "300", "--seed", "1"])
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["subcommand"] == "sim"
    assert cfg["cli_args"] == ["--t-ms", "300", "--seed", "1"]
    assert {"commit", "dirty", "generated_at"} <= cfg["_provenance"].keys()
    assert (run_dir / "run.sh").stat().st_mode & 0o111  # executable
