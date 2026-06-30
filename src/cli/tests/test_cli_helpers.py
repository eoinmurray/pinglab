"""Unit tests for assorted CLI helpers that were uncovered.

Targets pure-ish functions that don't need a GPU or a full training loop:
encoders, seed plumbing, key helpers, scan-var dispatch, CLI parsing.
"""


import numpy as np
import torch

import cli as O
from cli import (
    encode_image_spikes,
    encode_images_poisson,
    encode_smnist,
    encode_batch,
    seed_everything,
    primary_hid_key,
    primary_inh_key,
    _shd_cache_dir,
    _apply_scan_var,
    _auto_device,
    parse_args,
)


class TestEncodeImageSpikes:
    def test_shape_and_binary(self):
        pixels = np.array([0.5, 1.0, 0.0], dtype=np.float32)
        spikes = encode_image_spikes(
            pixels,
            T_steps=200,
            dt=1.0,
            base_rate=10.0,
            stim_rate=100.0,
            step_on_ms=50.0,
            step_off_ms=150.0,
            seed=42,
        )
        assert spikes.shape == (200, 3)
        assert spikes.dtype == torch.float32
        assert set(spikes.unique().tolist()) <= {0.0, 1.0}

    def test_zero_pixels_emit_no_spikes(self):
        pixels = np.zeros(5, dtype=np.float32)
        spikes = encode_image_spikes(
            pixels,
            T_steps=100,
            dt=1.0,
            base_rate=50.0,
            stim_rate=200.0,
            step_on_ms=20.0,
            step_off_ms=80.0,
            seed=1,
        )
        assert spikes.sum() == 0.0

    def test_seed_determinism(self):
        pixels = np.array([0.5, 0.5], dtype=np.float32)
        a = encode_image_spikes(
            pixels, 100, 1.0, 10.0, 80.0, step_on_ms=20.0, step_off_ms=80.0, seed=42
        )
        b = encode_image_spikes(
            pixels, 100, 1.0, 10.0, 80.0, step_on_ms=20.0, step_off_ms=80.0, seed=42
        )
        assert torch.equal(a, b)

    def test_different_seeds_diverge(self):
        pixels = np.array([0.5], dtype=np.float32)
        a = encode_image_spikes(
            pixels, 200, 1.0, 50.0, 200.0, step_on_ms=20.0, step_off_ms=180.0, seed=1
        )
        b = encode_image_spikes(
            pixels, 200, 1.0, 50.0, 200.0, step_on_ms=20.0, step_off_ms=180.0, seed=2
        )
        assert not torch.equal(a, b)

    def test_dt_rebins_same_event_stream(self):
        # Same seed/rate at two dts → same total spike count (within ±1 from
        # boundary clipping) because spike *times* are dt-invariant.
        pixels = np.array([1.0], dtype=np.float32)
        coarse = encode_image_spikes(
            pixels,
            T_steps=100,
            dt=1.0,
            base_rate=50.0,
            stim_rate=50.0,
            step_on_ms=0.0,
            step_off_ms=100.0,
            seed=7,
        )
        fine = encode_image_spikes(
            pixels,
            T_steps=1000,
            dt=0.1,
            base_rate=50.0,
            stim_rate=50.0,
            step_on_ms=0.0,
            step_off_ms=100.0,
            seed=7,
        )
        assert abs(coarse.sum().item() - fine.sum().item()) <= 1


class TestEncodeImagesPoisson:
    def test_shape_and_binary(self):
        images = torch.full((4, 16), 0.5)
        out = encode_images_poisson(images, T_steps=50, dt=1.0, max_rate_hz=100.0)
        assert out.shape == (50, 4, 16)
        assert set(out.unique().tolist()) <= {0.0, 1.0}

    def test_rate_proportional_to_intensity(self):
        # Mean firing rate over many steps ≈ pixel × max_rate × dt/1000.
        torch.manual_seed(0)
        images = torch.tensor([[0.0, 0.5, 1.0]])
        out = encode_images_poisson(images, T_steps=20000, dt=1.0, max_rate_hz=200.0)
        rates = out.mean(dim=0).squeeze()  # per-pixel mean spike prob
        # p = pixel * max_rate * dt/1000 = pixel * 0.2
        assert rates[0].item() == 0.0
        assert abs(rates[1].item() - 0.1) < 0.01
        assert abs(rates[2].item() - 0.2) < 0.01

    def test_generator_determinism(self):
        images = torch.full((2, 8), 0.7)
        g1 = torch.Generator().manual_seed(123)
        g2 = torch.Generator().manual_seed(123)
        a = encode_images_poisson(
            images, T_steps=30, dt=1.0, max_rate_hz=50.0, generator=g1
        )
        b = encode_images_poisson(
            images, T_steps=30, dt=1.0, max_rate_hz=50.0, generator=g2
        )
        assert torch.equal(a, b)

    def test_clamps_above_one(self):
        # Pixels > 1 should be clamped, not throw or produce p > 1 weirdness.
        images = torch.full((1, 4), 5.0)
        out = encode_images_poisson(images, T_steps=20, dt=1.0, max_rate_hz=100.0)
        assert out.shape == (20, 1, 4)
        assert set(out.unique().tolist()) <= {0.0, 1.0}


class TestEncodeSmnist:
    def test_shape_with_t_ms_per_row(self):
        # 28 rows × 10 ms / dt=1.0 = 280 steps, 28 cols.
        images = torch.zeros(2, 784)
        out = encode_smnist(images, dt=1.0, max_rate_hz=100.0, t_ms_per_row=10.0)
        assert out.shape == (280, 2, 28)

    def test_zero_image_no_spikes(self):
        images = torch.zeros(1, 784)
        out = encode_smnist(images, dt=1.0, max_rate_hz=200.0)
        assert out.sum().item() == 0.0

    def test_row_temporal_structure(self):
        # Row k's pixels drive timesteps [k*steps_per_row, (k+1)*steps_per_row).
        # Build an image where only row 5 is non-zero.
        torch.manual_seed(0)
        img = torch.zeros(1, 28, 28)
        img[0, 5, :] = 1.0
        out = encode_smnist(
            img.reshape(1, 784), dt=1.0, max_rate_hz=500.0, t_ms_per_row=10.0
        )
        # Row 5 → timesteps 50..59. All other timesteps must be zero.
        before = out[:50].sum().item()
        during = out[50:60].sum().item()
        after = out[60:].sum().item()
        assert before == 0.0 and after == 0.0
        assert during > 0.0


class TestEncodeBatch:
    def test_passthrough_for_3d(self):
        # (B, T, N_in) → (T, B, N_in)
        x = torch.randn(2, 30, 700)
        out = encode_batch(x, dt=1.0, use_smnist=False)
        assert out.shape == (30, 2, 700)
        assert torch.equal(out, x.permute(1, 0, 2))

    def test_routes_smnist(self):
        images = torch.zeros(1, 784)
        out = encode_batch(images, dt=1.0, use_smnist=True)
        # Smnist encoder produces (T, B, 28).
        assert out.shape[1:] == (1, 28)


class TestSeedEverything:
    def test_none_is_noop(self):
        # Should not raise; should not change torch state in a sticky way.
        before = torch.rand(1).item()
        seed_everything(None)
        after = torch.rand(1).item()
        # Just that it didn't crash; values may differ.
        assert isinstance(before, float) and isinstance(after, float)

    def test_round_trip_determinism(self):
        seed_everything(2024)
        a_torch = torch.rand(3).tolist()
        a_np = np.random.rand(3).tolist()
        seed_everything(2024)
        b_torch = torch.rand(3).tolist()
        b_np = np.random.rand(3).tolist()
        assert a_torch == b_torch
        assert a_np == b_np


class TestPrimaryKeys:
    def test_single_layer_hid(self):
        rec = {"hid": 1, "input": 2}
        assert primary_hid_key(rec) == "hid"

    def test_multi_layer_picks_deepest(self):
        rec = {"hid_1": 1, "hid_2": 2, "hid_3": 3}
        assert primary_hid_key(rec) == "hid_3"

    def test_no_hid_falls_back(self):
        # Function returns 'hid' even if absent — caller's responsibility to
        # check; this documents the behavior.
        rec = {"input": 1}
        assert primary_hid_key(rec) == "hid"

    def test_inh_returns_none_when_absent(self):
        assert primary_inh_key({"hid": 1}) is None

    def test_inh_picks_deepest(self):
        rec = {"inh_1": 1, "inh_2": 2}
        assert primary_inh_key(rec) == "inh_2"


class TestShdCacheDir:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("PINGLAB_SHD_DIR", raising=False)
        assert _shd_cache_dir() == "/tmp/shd/SHD"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PINGLAB_SHD_DIR", "/data/shd")
        assert _shd_cache_dir() == "/data/shd"


class TestApplyScanVar:
    def test_tau_gaba_updates_decay(self):
        old_tau = O.M.tau_gaba
        old_decay = O.M.decay_gaba
        try:
            _apply_scan_var("tau_gaba", 5.0)
            assert O.M.tau_gaba == 5.0
            assert abs(O.M.decay_gaba - np.exp(-O.M.dt / 5.0)) < 1e-12
        finally:
            O.M.tau_gaba = old_tau
            O.M.decay_gaba = old_decay

    def test_tau_ampa_updates_decay(self):
        old_tau = O.M.tau_ampa
        old_decay = O.M.decay_ampa
        try:
            _apply_scan_var("tau_ampa", 1.5)
            assert O.M.tau_ampa == 1.5
            assert abs(O.M.decay_ampa - np.exp(-O.M.dt / 1.5)) < 1e-12
        finally:
            O.M.tau_ampa = old_tau
            O.M.decay_ampa = old_decay

    def test_unknown_scan_var_ignored(self):
        # Unknown scan vars are ignored (only tau_*, config params are handled).
        # This is correct because scan vars like stim-overdrive don't mutate globals.
        _apply_scan_var("does-not-exist", 999.0)  # Should not raise


class TestAutoDevice:
    def test_returns_torch_device(self):
        d = _auto_device()
        assert isinstance(d, torch.device)
        assert d.type in {"cpu", "cuda", "mps"}


class TestParseArgs:
    def test_train_subparser(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv", ["cli", "train", "--epochs", "3", "--lr", "0.05"]
        )
        args = parse_args()
        assert args.epochs == 3
        assert args.lr == 0.05

    def test_train_default_lr(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["cli", "train"])
        args = parse_args()
        assert args.lr == 0.01
        assert args.epochs == 0

