"""Cross-experiment compact-input transfer contract, without simulation."""

import pytest
import torch

from experiments.exp171 import recipe


@pytest.mark.parametrize("target", recipe.RATE_TARGETS_HZ)
def test_compact_source_export_preserves_weights_and_detects_corruption(
    tmp_path, monkeypatch, target
):
    import json
    import subprocess
    from types import SimpleNamespace

    from pingstore import core
    from pingstore.library import file_sha256, source_run, stage_run

    from experiments.exp022 import analyse as source_analysis
    from experiments.exp171 import analyse as analysis
    from experiments.exp171 import compute as training

    monkeypatch.setattr(source_analysis, "REPO", tmp_path)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    cfg = recipe.configuration(rate_target_hz=target)
    conf = dict(
        dt=0.1,
        t_ms=200.0,
        max_samples=7000,
        input_rate=25.0,
        fr_reg_upper_strength=cfg["rate_penalty"],
        fr_reg_upper_target_hz=target or 0.0,
        readout_mode="mem-mean",
        seed=42,
        n_in=784,
        n_hidden=1024,
        n_inh=256,
        n_out=10,
        tau_gaba_ms=6.0,
        tau_ampa_ms=2.0,
        v_grad_dampen=1000.0,
        epochs=50,
        dataset_split={"validation_samples": 700},
        validation_encoder_draws={"count": 3},
    )
    cells = [cfg[kind + "_cell"] for kind in ("coba", "ping")]
    with stage_run(tmp_path, "exp022", "compute") as bank:
        provenance = {}
        for cell in cells:
            directory = bank.export / cell
            directory.mkdir()
            (directory / "config.json").write_text(json.dumps(conf))
            (directory / "metrics.json").write_text(
                json.dumps(
                    dict(
                        config=conf,
                        training_run_id="TR-02",
                        epochs=[
                            dict(ep=50, acc=85.0, test_rate_e=12.0, test_rate_i=30.0)
                        ],
                    )
                )
            )
            weights = {key: torch.zeros(1, 1) for key in recipe.MAPPING.values()}
            weights["W_ff.0"] = torch.ones(1, 1)
            torch.save(weights, directory / "weights_final.pth")
            provenance[cell] = {
                "checkpoint": dict(
                    role="final_epoch",
                    epoch=50,
                    filename="weights_final.pth",
                    sha256=file_sha256(directory / "weights_final.pth"),
                )
            }
        (bank.export / "unused-recording.npz").write_bytes(b"unused")
        bank.details["bank_reuse"] = {"diagnostic_provenance": provenance}
    original = source_run(tmp_path / ".pingstore", bank.run_id)
    cfg.update(source_run=bank.run_id, source_digest="historical-scientific-provenance")
    _, expected = training.load_source(cfg)

    def computed(pin):
        return SimpleNamespace(
            details={
                "inputs": {"bank": pin},
                "execution": {"configuration": cfg},
            }
        )

    expected_controls = analysis.source_controls(computed(original.reference))
    identity = source_analysis.export_finetuning(bank.run_id, cells)
    compact = source_run(tmp_path / ".pingstore", identity)
    assert compact.details["inputs"]["bank"] == original.reference
    assert not (compact.export / "unused-recording.npz").exists()
    cfg.update(source_run=identity, source_digest="historical-scientific-provenance")
    _, actual = training.load_source(cfg)
    assert analysis.source_controls(computed(compact.reference)) == expected_controls
    assert analysis.source_bank(cfg, root=tmp_path)[1] == original.reference

    class ReachedNativeArtifacts(Exception):
        pass

    def stop_at_artifacts(*parts):
        assert parts == ("network.bundle", "training.json")
        raise ReachedNativeArtifacts

    evidence = computed(compact.reference)
    evidence.file = stop_at_artifacts
    with pytest.raises(ReachedNativeArtifacts):
        analysis.validate_native(evidence)
    for kind in expected:
        for name in expected[kind]:
            torch.testing.assert_close(actual[kind][name], expected[kind][name])
    (compact.export / cells[0] / "config.json").write_text("{}")
    with pytest.raises((core.PingstoreError, KeyError)):
        training.load_source(cfg)
