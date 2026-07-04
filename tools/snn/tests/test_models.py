import models as M
import pytest
import torch
from config import build_net
from torch import nn


@pytest.fixture(autouse=True)
def _small_model_sizes():
    """Keep tests fast by shrinking model constants."""
    old = (M.N_IN, M.N_HID, M.N_INH, M.N_OUT, M.HIDDEN_SIZES, M.T_ms, M.T_steps)
    M.N_IN = 16
    M.N_HID = 32
    M.N_INH = 8
    M.N_OUT = 10
    M.HIDDEN_SIZES = [32]
    M.T_ms = 50.0
    M.T_steps = int(M.T_ms / M.dt)
    yield
    (M.N_IN, M.N_HID, M.N_INH, M.N_OUT, M.HIDDEN_SIZES, M.T_ms, M.T_steps) = old


class TestBuildNetRegistry:
    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_net("nonexistent")

    def test_ping_instantiates(self):
        net = build_net("ping", hidden_sizes=[32])
        assert isinstance(net, nn.Module)


class TestCOBANetFrozenWeights:
    def test_recurrent_weights_are_frozen(self):
        """COBANet W_ee/W_ei/W_ie must have requires_grad=False after construction."""
        net = build_net("ping", hidden_sizes=[32])
        for name in ["W_ee", "W_ei", "W_ie"]:
            pdict = getattr(net, name)
            assert isinstance(pdict, nn.ParameterDict)
            for key, param in pdict.items():
                assert not param.requires_grad, (
                    f"{name}[{key}] should be frozen (requires_grad=False)"
                )

    def test_recurrent_weights_survive_optimizer_step(self):
        """Freezing means an optimizer.step() on all params leaves them unchanged."""
        net = build_net("ping", hidden_sizes=[32])
        snapshots = {
            f"{name}_{k}": p.detach().clone()
            for name in ["W_ee", "W_ei", "W_ie"]
            for k, p in getattr(net, name).items()
        }
        trainable = [p for p in net.parameters() if p.requires_grad]
        assert len(trainable) > 0, "expected at least one trainable param"
        opt = torch.optim.SGD(trainable, lr=1.0)
        for p in trainable:
            p.grad = torch.ones_like(p)
        opt.step()
        for name in ["W_ee", "W_ei", "W_ie"]:
            for k, p in getattr(net, name).items():
                assert torch.equal(p, snapshots[f"{name}_{k}"]), (
                    f"{name}[{k}] changed after optimizer step"
                )


class TestSeedReproducibility:
    def test_same_seed_gives_same_weights(self):
        def _weights(net):
            return [p.detach().clone() for p in net.parameters()]

        torch.manual_seed(123)
        a = _weights(build_net("ping", hidden_sizes=[32]))
        torch.manual_seed(123)
        b = _weights(build_net("ping", hidden_sizes=[32]))
        assert len(a) == len(b)
        for pa, pb in zip(a, b):
            assert torch.equal(pa, pb)


class TestDalesLawClamp:
    """Dale's-law forward-pass clamp: when signed_weights=False (default
    ``dales_law=True`` at build time) the forward pass must use
    max(W_ff, 0) rather than the raw stored weights. This is the
    mechanism that lets nb049's optimiser push trainable weights below
    zero in storage without those negative entries propagating into the
    network's dynamics."""

    def test_negative_w_ff_zeroes_out_in_forward(self):
        """If Dale's law is on, replacing a positive W_ff entry with a
        large negative one shouldn't change forward output — the clamp
        treats it as zero."""
        torch.manual_seed(0)
        net = build_net("ping", hidden_sizes=[32], dales_law=True)
        net.recording = False
        assert not net.signed_weights, (
            "dales_law=True should produce signed_weights=False"
        )
        spikes = (torch.rand(M.T_steps, 1, M.N_IN) < 0.2).float()
        with torch.no_grad():
            ref_logits = net.forward(input_spikes=spikes)

        # Zero a few W_ff[0] entries in storage, run again — should match.
        with torch.no_grad():
            net.W_ff[0].data[:3, :3] = 0.0
            forced_zero_logits = net.forward(input_spikes=spikes)

        # Now overwrite those same entries with large negatives and rerun
        # — the clamp on forward must collapse them to zero, producing
        # the same output as the explicit-zero case.
        with torch.no_grad():
            net.W_ff[0].data[:3, :3] = -100.0
            clamped_logits = net.forward(input_spikes=spikes)

        assert torch.allclose(forced_zero_logits, clamped_logits, atol=1e-6), (
            "negative W_ff entries should clamp to zero in forward, but the "
            "output differed from explicit-zero entries"
        )
        # And the ref (with the original positive values) should differ from
        # both — otherwise the test is vacuous (the W entries we zeroed had
        # no effect on the output).
        assert not torch.allclose(ref_logits, clamped_logits, atol=1e-6), (
            "test is vacuous: zeroing some W_ff entries had no effect on output"
        )

    def test_signed_weights_disables_clamp(self):
        """With dales_law=False, the forward pass uses raw signed weights.
        Negative entries are NOT zeroed out, so they must propagate."""
        torch.manual_seed(0)
        net = build_net("ping", hidden_sizes=[32], dales_law=False)
        assert net.signed_weights
        spikes = (torch.rand(M.T_steps, 1, M.N_IN) < 0.2).float()

        with torch.no_grad():
            net.W_ff[0].data[:3, :3] = 0.0
            zero_logits = net.forward(input_spikes=spikes)
            net.W_ff[0].data[:3, :3] = -100.0
            neg_logits = net.forward(input_spikes=spikes)

        # The two should differ — without the clamp, -100 ≠ 0.
        assert not torch.allclose(zero_logits, neg_logits, atol=1e-6), (
            "signed_weights=True should let negative W_ff entries change "
            "the forward output, but zero and -100 gave identical logits"
        )
