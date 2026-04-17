"""FrozenEncoder dt-invariance: spike patterns across dt values must agree
after OR-pooling. If this drifts, dt-sweep calibration results are lying.
"""
import pytest
import torch

from oscilloscope import FrozenEncoder, downsample_spikes_or


def _same_batch(B=4, n_in=16, seed=0):
    torch.manual_seed(seed)
    return torch.rand(B, n_in)


class TestDownsampleOr:
    def test_k_equals_one_is_identity(self):
        spk = torch.randint(0, 2, (10, 2, 3)).float()
        out = downsample_spikes_or(spk, 0.25, 0.25)
        assert torch.equal(out, spk)

    def test_or_pools_within_each_block(self):
        """Any '1' inside a block of k steps should survive to the pooled output."""
        spk = torch.zeros((6, 1, 1))
        spk[1] = 1.0   # middle of first block of 3
        spk[5] = 1.0   # last step of second block
        out = downsample_spikes_or(spk, dt_ref=0.1, dt_target=0.3)
        assert out.shape == (2, 1, 1)
        assert out[0].item() == 1.0
        assert out[1].item() == 1.0

    def test_trims_leftover_steps(self):
        spk = torch.zeros((7, 1, 1))
        out = downsample_spikes_or(spk, 0.1, 0.3)
        assert out.shape == (2, 1, 1)  # 7 // 3 = 2 blocks, 1 step dropped


class TestFrozenEncoder:
    def test_dt_ratio_1_round_trip(self):
        """At target dt == ref dt, no downsampling happens — spikes pass through."""
        enc = FrozenEncoder(dt_ref=0.1, t_ms=50.0, base_seed=42)
        X = _same_batch()
        a = enc(X, dt=0.1, use_smnist=False)
        expected_T = int(50.0 / 0.1)
        assert a.shape == (expected_T, X.shape[0], X.shape[1])

    @pytest.mark.parametrize("dt_target", [0.2, 0.5, 1.0])
    def test_spike_pattern_invariant_across_dt(self, dt_target):
        """Call at dt_ref, then reset and call at dt_target; the latter must
        equal OR-pooling the former. This is the core FrozenEncoder promise."""
        dt_ref = 0.1
        t_ms = 100.0
        enc = FrozenEncoder(dt_ref=dt_ref, t_ms=t_ms, base_seed=42)
        X = _same_batch()

        spk_ref = enc(X, dt=dt_ref, use_smnist=False)

        enc.reset()
        spk_target = enc(X, dt=dt_target, use_smnist=False)

        pooled = downsample_spikes_or(spk_ref, dt_ref, dt_target)
        assert torch.equal(spk_target, pooled)

    def test_reset_rewinds_batch_index(self):
        """Two separate 'sweeps' over the same batch sequence must produce
        identical spikes — that's what reset() exists for."""
        enc = FrozenEncoder(dt_ref=0.1, t_ms=50.0, base_seed=42)
        X1, X2 = _same_batch(seed=1), _same_batch(seed=2)

        a1 = enc(X1, dt=0.1, use_smnist=False)
        a2 = enc(X2, dt=0.1, use_smnist=False)

        enc.reset()
        b1 = enc(X1, dt=0.1, use_smnist=False)
        b2 = enc(X2, dt=0.1, use_smnist=False)

        assert torch.equal(a1, b1)
        assert torch.equal(a2, b2)

    def test_different_base_seeds_diverge(self):
        enc_a = FrozenEncoder(dt_ref=0.1, t_ms=50.0, base_seed=1)
        enc_b = FrozenEncoder(dt_ref=0.1, t_ms=50.0, base_seed=2)
        X = _same_batch()
        a = enc_a(X, dt=0.1, use_smnist=False)
        b = enc_b(X, dt=0.1, use_smnist=False)
        assert not torch.equal(a, b)
