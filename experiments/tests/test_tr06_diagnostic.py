from __future__ import annotations

from experiments.exp022_support import tr06_diagnostic


def test_tr06_diagnostic_variants_change_only_the_readout_contract(tmp_path) -> None:
    commands = {
        variant: tr06_diagnostic.diagnostic_args(
            variant,
            output=tmp_path / variant,
            max_samples=700,
            epochs=10,
            seed=42,
            device="cuda",
        )
        for variant in tr06_diagnostic.VARIANTS
    }
    registered = commands["registered-spike-count"]
    fanin = commands["fanin-spike-count"]
    control = commands["mem-mean-control"]

    assert registered[registered.index("--readout") + 1] == "spike-count"
    assert "--readout-w-init-mean" in registered
    assert "--readout-w-init-std" in registered
    assert fanin[fanin.index("--readout") + 1] == "spike-count"
    assert "--readout-w-init-mean" not in fanin
    assert "--readout-w-init-std" not in fanin
    assert control[control.index("--readout") + 1] == "mem-mean"
    for command in commands.values():
        assert command[command.index("--max-samples") + 1] == "700"
        assert command[command.index("--epochs") + 1] == "10"
        assert command[command.index("--seed") + 1] == "42"
        assert command[command.index("--device") + 1] == "cuda"
        start = command.index("--input-rates") + 1
        assert tuple(map(float, command[start : start + 11])) == (
            0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 25.0,
        )
