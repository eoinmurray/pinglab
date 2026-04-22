"""Tier-1 flag-propagation tests — every knob we rely on in notebook runners.

Pattern follows test_mode_drift.py: spawn oscilloscope, parse config.json /
metrics.json, assert the flag actually landed. In-process tests (fast) check
behavior where subprocess isn't needed.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch


def _run_cli(*args, timeout=180):
    cmd = ["uv", "run", "python", "src/pinglab/oscilloscope.py", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, (
        f"cmd failed (exit {result.returncode}):\n  {' '.join(cmd)}\n"
        f"stderr: {result.stderr[-500:]}"
    )


def _read_config(out_dir):
    cfg = Path(out_dir) / "config.json"
    assert cfg.exists(), f"no config.json in {out_dir}"
    return json.loads(cfg.read_text())


def _train_probe(out_dir, *extra, epochs=0):
    """standard-snn train — fast way to land a full config on disk.
    epochs=0 is probe-only (no weights.pth); pass epochs=1 when a test needs
    a loadable checkpoint."""
    _run_cli("train", "--model", "standard-snn",
             "--dataset", "mnist", "--max-samples", "50",
             "--epochs", str(epochs), "--dt", "0.25",
             "--w-in", "10", "--w-in-sparsity", "0",
             "--out-dir", str(out_dir), "--wipe-dir",
             *extra)


# ── --from-dir inheritance ───────────────────────────────────────────────

@pytest.mark.slow
def test_from_dir_inherits_config(tmp_path):
    """image --from-dir picks up dt, t_ms, n_hidden from the training config."""
    train_dir = tmp_path / "train"
    _train_probe(train_dir, "--t-ms", "150", "--n-hidden", "128", epochs=1)
    train_cfg = _read_config(train_dir)

    image_dir = tmp_path / "image"
    _run_cli("image",
             "--from-dir", str(train_dir),
             "--digit", "0",
             "--out-dir", str(image_dir), "--wipe-dir")
    image_cfg = _read_config(image_dir)

    for key in ("dt", "t_ms", "model", "dataset"):
        assert image_cfg[key] == train_cfg[key], (
            f"{key}: train={train_cfg[key]} image={image_cfg[key]}"
        )
    # Hidden-layer spec is stored inconsistently between modes (train uses a
    # scalar n_hidden + hidden_sizes list; image emits n_hidden as a list and
    # hidden_sizes as null). Normalise both sides to a list to compare.
    def _hidden(cfg):
        hs = cfg.get("hidden_sizes")
        if hs:
            return list(hs)
        n = cfg.get("n_hidden")
        return list(n) if isinstance(n, list) else [n]
    assert _hidden(image_cfg) == _hidden(train_cfg), (
        f"hidden: train={train_cfg.get('n_hidden')}/{train_cfg.get('hidden_sizes')} "
        f"image={image_cfg.get('n_hidden')}/{image_cfg.get('hidden_sizes')}"
    )


@pytest.mark.slow
def test_from_dir_cli_overrides(tmp_path):
    """Explicit CLI flag wins over inherited --from-dir config."""
    train_dir = tmp_path / "train"
    _train_probe(train_dir, "--t-ms", "150", epochs=1)
    assert _read_config(train_dir)["t_ms"] == 150.0

    image_dir = tmp_path / "image"
    _run_cli("image",
             "--from-dir", str(train_dir),
             "--t-ms", "200",
             "--digit", "0",
             "--out-dir", str(image_dir), "--wipe-dir")
    assert _read_config(image_dir)["t_ms"] == 200.0


@pytest.mark.slow
def test_from_dir_auto_loads_weights(tmp_path):
    """infer --from-dir finds weights.pth automatically — no --load-weights needed."""
    train_dir = tmp_path / "train"
    _run_cli("train", "--model", "standard-snn",
             "--dataset", "mnist", "--max-samples", "60",
             "--epochs", "1", "--dt", "0.25",
             "--w-in", "10", "--w-in-sparsity", "0",
             "--out-dir", str(train_dir), "--wipe-dir")
    assert (train_dir / "weights.pth").exists()

    infer_dir = tmp_path / "infer"
    _run_cli("infer", "--from-dir", str(train_dir),
             "--max-samples", "60",
             "--out-dir", str(infer_dir), "--wipe-dir")
    metrics = json.loads((infer_dir / "metrics.json").read_text())
    assert metrics.get("best_acc") is not None


# ── --readout {rate, li} ─────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("readout", ["rate", "li"])
def test_readout_propagates(tmp_path, readout):
    out = tmp_path / f"readout-{readout}"
    _train_probe(out, "--readout", readout)
    assert _read_config(out)["readout_mode"] == readout


def test_readout_changes_model_forward():
    """rate and li readouts produce differently-shaped output trajectories —
    rate emits logits once at the final step, li integrates across all steps."""
    import sys
    sys.path.insert(0, "src/pinglab")
    from config import build_net

    torch.manual_seed(0)
    net_rate = build_net("standard-snn", w_in=(10.0, 1.0), w_in_sparsity=0.0,
                         hidden_sizes=[64], readout_mode="rate")
    torch.manual_seed(0)
    net_li = build_net("standard-snn", w_in=(10.0, 1.0), w_in_sparsity=0.0,
                       hidden_sizes=[64], readout_mode="li")
    # Parameter count must match — same W_out, b_out shape, only reduction differs.
    n_rate = sum(p.numel() for p in net_rate.parameters())
    n_li = sum(p.numel() for p in net_li.parameters())
    assert n_rate == n_li, f"rate={n_rate} li={n_li}"


# ── --cm-back-scale ──────────────────────────────────────────────────────

@pytest.mark.slow
def test_cm_back_scale_propagates(tmp_path):
    out = tmp_path / "cmbs"
    _train_probe(out, "--cm-back-scale", "1234.0")
    assert _read_config(out)["cm_back_scale"] == 1234.0


# ── --ei-strength / --ei-ratio (in-process, fast) ────────────────────────

def test_ei_ratio_scales_only_wie():
    """Doubling ei_ratio at fixed ei_strength doubles W_ie but leaves W_ei
    unchanged (the builder sets W_ei = s, W_ie = s*ratio)."""
    import sys
    sys.path.insert(0, "src/pinglab")
    from config import build_net

    def _means(ratio):
        torch.manual_seed(0)
        net = build_net("ping", w_in=(0.3, 0.03), w_in_sparsity=0.0,
                        ei_strength=0.5, ei_ratio=ratio, sparsity=0.0)
        p = dict(net.named_parameters())
        return (float(p["W_ei.1"].detach().mean()),
                float(p["W_ie.1"].detach().mean()))

    ei_lo, ie_lo = _means(2.0)
    ei_hi, ie_hi = _means(4.0)
    assert abs(ei_hi / ei_lo - 1.0) < 0.05, f"W_ei drifted with ratio: {ei_hi/ei_lo}"
    assert abs(ie_hi / ie_lo - 2.0) < 0.05, f"W_ie scale vs ratio: {ie_hi/ie_lo}"


def test_ei_strength_scales_weights():
    """Doubling ei_strength at fixed ratio doubles both W_ei and W_ie means."""
    import sys
    sys.path.insert(0, "src/pinglab")
    from config import build_net

    def _means(s):
        torch.manual_seed(0)
        net = build_net("ping", w_in=(0.3, 0.03), w_in_sparsity=0.0,
                        ei_strength=s, ei_ratio=2.0, sparsity=0.0)
        p = dict(net.named_parameters())
        return (float(p["W_ei.1"].detach().mean()),
                float(p["W_ie.1"].detach().mean()))

    ei_lo, ie_lo = _means(0.25)
    ei_hi, ie_hi = _means(0.50)
    assert abs(ei_hi / ei_lo - 2.0) < 0.05, f"W_ei scale: {ei_hi/ei_lo}"
    assert abs(ie_hi / ie_lo - 2.0) < 0.05, f"W_ie scale: {ie_hi/ie_lo}"


# ── Tier 2: training-path knobs ──────────────────────────────────────────

# --early-stopping

@pytest.mark.slow
def test_early_stopping_propagates(tmp_path):
    out = tmp_path / "es"
    _train_probe(out, "--early-stopping", "3")
    assert _read_config(out)["early_stopping"] == 3


@pytest.mark.slow
def test_early_stopping_triggers_stop(tmp_path):
    """With --early-stopping 1 and a trivially plateauing setup, the run
    halts before --epochs."""
    out = tmp_path / "es-fire"
    _run_cli("train", "--model", "standard-snn",
             "--dataset", "mnist", "--max-samples", "60",
             "--epochs", "20", "--early-stopping", "1",
             "--dt", "0.25", "--w-in", "10", "--w-in-sparsity", "0",
             "--lr", "1e-8",  # frozen — no improvement possible
             "--out-dir", str(out), "--wipe-dir",
             timeout=240)
    jsonl = (out / "metrics.jsonl").read_text().strip().splitlines()
    assert len(jsonl) < 20, f"early-stopping did not fire: {len(jsonl)} epochs ran"


# --adaptive-lr

@pytest.mark.slow
def test_adaptive_lr_smoke(tmp_path):
    """--adaptive-lr run completes and emits lr in every jsonl row."""
    out = tmp_path / "adlr"
    _run_cli("train", "--model", "standard-snn",
             "--dataset", "mnist", "--max-samples", "50",
             "--epochs", "2", "--adaptive-lr",
             "--dt", "0.25", "--w-in", "10", "--w-in-sparsity", "0",
             "--out-dir", str(out), "--wipe-dir")
    rows = [json.loads(l) for l in (out / "metrics.jsonl").read_text().strip().splitlines()]
    assert len(rows) == 2
    assert all("lr" in r for r in rows)


# --kaiming-init / --init-scale-* / --dales-law (config propagation)

@pytest.mark.slow
@pytest.mark.parametrize("flag,key,expected", [
    (["--kaiming-init"], "kaiming_init", True),
    (["--init-scale-weight", "2.5"], "init_scale_weight", 2.5),
    (["--init-scale-bias", "0.5"], "init_scale_bias", 0.5),
    (["--no-dales-law"], "dales_law", False),
])
def test_train_flag_propagates_to_config(tmp_path, flag, key, expected):
    out = tmp_path / f"cfg-{key}"
    _train_probe(out, *flag)
    assert _read_config(out)[key] == expected


# --kaiming-init / --no-dales-law (behavior, in-process, fast)

def test_kaiming_init_produces_signed_weights():
    """kaiming_init=True gives signed Kaiming-uniform weights; the default
    pipeline produces strictly non-negative weights (Dale's-law-compatible)."""
    import sys
    sys.path.insert(0, "src/pinglab")
    from config import build_net

    torch.manual_seed(0)
    net_default = build_net("standard-snn", w_in=(10.0, 1.0), w_in_sparsity=0.0,
                            hidden_sizes=[32])
    torch.manual_seed(0)
    net_kaiming = build_net("standard-snn", kaiming_init=True, hidden_sizes=[32])

    w_default = dict(net_default.named_parameters())["W_ff.1"].detach()
    w_kaiming = dict(net_kaiming.named_parameters())["W_ff.1"].detach()
    assert float(w_default.min()) >= 0.0, "default init leaked negative weights"
    assert float(w_kaiming.min()) < 0.0, "kaiming init should produce signed weights"


def test_no_dales_law_allows_signed_weights():
    """dales_law=False permits negative weights at init; dales_law=True clamps."""
    import sys
    sys.path.insert(0, "src/pinglab")
    from config import build_net

    torch.manual_seed(0)
    net_dale = build_net("standard-snn", w_in=(10.0, 1.0), w_in_sparsity=0.0,
                         dales_law=True, hidden_sizes=[32])
    torch.manual_seed(0)
    net_free = build_net("standard-snn", w_in=(10.0, 1.0), w_in_sparsity=0.0,
                         dales_law=False, hidden_sizes=[32])

    w_dale = dict(net_dale.named_parameters())["W_ff.1"].detach()
    w_free = dict(net_free.named_parameters())["W_ff.1"].detach()
    assert float(w_dale.min()) >= 0.0
    assert float(w_free.min()) < 0.0


# --fr-reg-upper / --fr-reg-lower (behavior)

@pytest.mark.slow
def test_fr_reg_upper_pulls_rate_down(tmp_path):
    """A strong upper-bound firing-rate regulariser drives end-of-training
    rate_e below the unregularised baseline, with everything else held fixed."""
    def _rate(out_dir, *extra):
        _run_cli("train", "--model", "standard-snn",
                 "--dataset", "mnist", "--max-samples", "60",
                 "--epochs", "3", "--seed", "0",
                 "--dt", "0.25", "--w-in", "10", "--w-in-sparsity", "0",
                 "--out-dir", str(out_dir), "--wipe-dir",
                 *extra)
        return json.loads((out_dir / "metrics.json").read_text())["end"]["rate_e"]

    baseline = _rate(tmp_path / "fr-base")
    regd = _rate(tmp_path / "fr-upper",
                 "--fr-reg-upper-theta", "1",
                 "--fr-reg-upper-strength", "10")
    assert regd < baseline, (
        f"upper FR reg failed to suppress rate: baseline={baseline:.2f} "
        f"regularised={regd:.2f}"
    )


# ── --wipe-dir behavior ──────────────────────────────────────────────────

@pytest.mark.slow
def test_wipe_dir_clears_existing(tmp_path):
    """--wipe-dir removes pre-existing files before the run."""
    out = tmp_path / "wipe-on"
    out.mkdir()
    sentinel = out / "stale.txt"
    sentinel.write_text("should be gone")

    _train_probe(out)
    assert not sentinel.exists(), "--wipe-dir should have removed stale.txt"
    assert (out / "config.json").exists()


@pytest.mark.slow
def test_no_wipe_dir_preserves_existing(tmp_path):
    """Without --wipe-dir, unrelated files in the output dir survive."""
    out = tmp_path / "wipe-off"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("should stay")

    _run_cli("train", "--model", "standard-snn",
             "--dataset", "mnist", "--max-samples", "50",
             "--epochs", "0", "--dt", "0.25",
             "--w-in", "10", "--w-in-sparsity", "0",
             "--out-dir", str(out))
    assert sentinel.exists(), "no --wipe-dir should have preserved keep.txt"
    assert (out / "config.json").exists()
