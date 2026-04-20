"""FrozenEncoder dt-invariance: spike patterns across dt values must agree
after the paper's count-preserving transport (Parthasarathy et al. §2.1 /
Fig 1B / §2.3). If this drifts, dt-sweep calibration results are lying.
"""
import pytest
import torch

from oscilloscope import (
    FROZEN_MODES,
    FrozenEncoder,
    downsample_spikes_count,
    transport_spikes_bin,
    upsample_spikes_zeropad,
)


def _same_batch(B=4, n_in=16, seed=0):
    torch.manual_seed(seed)
    return torch.rand(B, n_in)


class TestDownsampleCount:
    def test_k_equals_one_is_identity(self):
        spk = torch.randint(0, 2, (10, 2, 3)).float()
        out = downsample_spikes_count(spk, 0.25, 0.25)
        assert torch.equal(out, spk)

    def test_sum_pools_within_block(self):
        spk = torch.zeros((6, 1, 1))
        spk[0] = 1.0
        spk[1] = 1.0
        spk[2] = 1.0  # first block of 3: sum = 3
        spk[5] = 1.0  # second block: sum = 1
        out = downsample_spikes_count(spk, dt_ref=0.1, dt_target=0.3)
        assert out.shape == (2, 1, 1)
        assert out[0].item() == 3.0
        assert out[1].item() == 1.0

    def test_trims_leftover_steps(self):
        spk = torch.zeros((7, 1, 1))
        out = downsample_spikes_count(spk, 0.1, 0.3)
        assert out.shape == (2, 1, 1)  # 7 // 3 = 2 blocks, 1 step dropped


class TestUpsampleZeropad:
    def test_k_equals_one_is_identity(self):
        spk = torch.randint(0, 2, (10, 2, 3)).float()
        out = upsample_spikes_zeropad(spk, 0.25, 0.25)
        assert torch.equal(out, spk)

    def test_expands_and_places_at_first_subbin(self):
        """Paper Fig 1B: each reference spike lands at the first fine sub-step
        of its block; the rest of the block is zero."""
        spk = torch.tensor([[1.0], [0.0], [1.0]]).unsqueeze(1)  # (3,1,1)
        out = upsample_spikes_zeropad(spk, dt_ref=0.3, dt_target=0.1)
        assert out.shape == (9, 1, 1)
        assert out[0].item() == 1.0 and out[3].item() == 0.0 and out[6].item() == 1.0
        # zeros in between
        for i in (1, 2, 4, 5, 7, 8):
            assert out[i].item() == 0.0

    def test_preserves_total_spike_count(self):
        torch.manual_seed(0)
        spk = (torch.rand(12, 2, 3) < 0.3).float()
        out = upsample_spikes_zeropad(spk, dt_ref=0.4, dt_target=0.1)
        assert out.shape == (48, 2, 3)
        assert out.sum().item() == spk.sum().item()


class TestTransportBin:
    def test_identity_when_equal(self):
        spk = torch.randint(0, 2, (8, 1, 2)).float()
        out = transport_spikes_bin(spk, 0.1, 0.1)
        assert torch.equal(out, spk)

    def test_dispatches_to_zeropad_when_finer(self):
        spk = torch.randint(0, 2, (6, 1, 1)).float()
        out = transport_spikes_bin(spk, dt_ref=0.2, dt_target=0.1)
        assert out.shape == (12, 1, 1)
        assert out.sum().item() == spk.sum().item()

    def test_dispatches_to_sumpool_when_coarser(self):
        spk = torch.ones((6, 1, 1))
        out = transport_spikes_bin(spk, dt_ref=0.1, dt_target=0.3)
        assert out.shape == (2, 1, 1)
        assert out.sum().item() == 6.0  # count preserved (no leftover here)

    def test_roundtrip_through_finer_and_back(self):
        """Zero-pad then sum-pool with the same k recovers the original."""
        torch.manual_seed(1)
        spk = (torch.rand(8, 2, 3) < 0.5).float()
        fine = transport_spikes_bin(spk, dt_ref=0.4, dt_target=0.1)
        back = transport_spikes_bin(fine, dt_ref=0.1, dt_target=0.4)
        assert torch.equal(back, spk)

    def test_rejects_non_integer_ratio_either_direction(self):
        spk = torch.zeros((6, 1, 1))
        with pytest.raises(ValueError):
            transport_spikes_bin(spk, dt_ref=0.1, dt_target=0.25)
        with pytest.raises(ValueError):
            transport_spikes_bin(spk, dt_ref=0.25, dt_target=0.1)


class TestFrozenEncoder:
    def test_identity_at_reference_dt(self):
        enc = FrozenEncoder(dt_ref=0.1, t_ms=50.0, base_seed=42)
        X = _same_batch()
        a = enc(X, dt=0.1, use_smnist=False)
        expected_T = int(50.0 / 0.1)
        assert a.shape == (expected_T, X.shape[0], X.shape[1])

    @pytest.mark.parametrize("dt_target", [0.2, 0.5, 1.0])
    def test_zero_pad_mode_sum_pools_to_coarser(self, dt_target):
        """When eval-dt is coarser than train-dt, zero-pad mode must sum-pool
        the reference stream — matches downsample_spikes_count."""
        dt_ref = 0.1
        t_ms = 100.0
        enc = FrozenEncoder(dt_ref=dt_ref, t_ms=t_ms, base_seed=42)
        X = _same_batch()

        spk_ref = enc(X, dt=dt_ref, use_smnist=False)
        enc.reset()
        spk_target = enc(X, dt=dt_target, use_smnist=False)

        pooled = downsample_spikes_count(spk_ref, dt_ref, dt_target)
        assert torch.equal(spk_target, pooled)

    @pytest.mark.parametrize("dt_target", [0.1, 0.05])
    def test_zero_pad_mode_expands_to_finer(self, dt_target):
        """When eval-dt is finer than train-dt, zero-pad mode must expand the
        reference stream with zeros between spikes."""
        dt_ref = 0.2
        t_ms = 50.0
        enc = FrozenEncoder(dt_ref=dt_ref, t_ms=t_ms, base_seed=42)
        X = _same_batch()

        spk_ref = enc(X, dt=dt_ref, use_smnist=False)
        enc.reset()
        spk_target = enc(X, dt=dt_target, use_smnist=False)

        expanded = upsample_spikes_zeropad(spk_ref, dt_ref, dt_target)
        assert torch.equal(spk_target, expanded)
        assert spk_target.sum().item() == spk_ref.sum().item()

    def test_reset_rewinds_batch_index(self):
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

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            FrozenEncoder(dt_ref=0.1, t_ms=10.0, mode="nonsense")

    def test_modes_registered(self):
        assert set(FROZEN_MODES) == {"zero-pad", "resample"}


class TestFrozenEncoderResample:
    def test_shape_matches_target_dt(self):
        dt_target, t_ms = 0.5, 100.0
        enc = FrozenEncoder(dt_ref=0.1, t_ms=t_ms, base_seed=42, mode="resample")
        X = _same_batch()
        spk = enc(X, dt=dt_target, use_smnist=False)
        expected_T = int(t_ms / dt_target)
        assert spk.shape == (expected_T, X.shape[0], X.shape[1])

    def test_not_equal_to_zero_pad(self):
        """resample draws fresh noise at target dt; it should diverge from the
        zero-pad path that reuses the reference stream."""
        dt_ref, dt_target, t_ms = 0.1, 0.5, 100.0
        enc_re = FrozenEncoder(dt_ref=dt_ref, t_ms=t_ms, base_seed=42, mode="resample")
        enc_zp = FrozenEncoder(dt_ref=dt_ref, t_ms=t_ms, base_seed=42, mode="zero-pad")
        X = _same_batch()
        a = enc_re(X, dt=dt_target, use_smnist=False)
        b = enc_zp(X, dt=dt_target, use_smnist=False)
        assert not torch.equal(a, b)

    def test_reset_reproduces_stream(self):
        enc = FrozenEncoder(dt_ref=0.1, t_ms=50.0, base_seed=3, mode="resample")
        X = _same_batch()
        a = enc(X, dt=0.25, use_smnist=False)
        enc.reset()
        b = enc(X, dt=0.25, use_smnist=False)
        assert torch.equal(a, b)
