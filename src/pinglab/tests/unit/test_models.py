import pytest
import torch
from torch import nn

import models as M
from config import build_net


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

    @pytest.mark.parametrize("name", ["ping", "snntorch-clone", "cuba"])
    def test_each_registered_model_instantiates(self, name):
        net = build_net(name, hidden_sizes=[32])
        assert isinstance(net, nn.Module)


class TestPINGNetFrozenWeights:
    def test_recurrent_weights_are_frozen(self):
        """PINGNet W_ee/W_ei/W_ie must have requires_grad=False after construction."""
        net = build_net("ping", hidden_sizes=[32])
        for name in ["W_ee", "W_ei", "W_ie"]:
            pdict = getattr(net, name)
            assert isinstance(pdict, nn.ParameterDict)
            for key, param in pdict.items():
                assert not param.requires_grad, \
                    f"{name}[{key}] should be frozen (requires_grad=False)"

    def test_recurrent_weights_survive_optimizer_step(self):
        """Freezing means an optimizer.step() on all params leaves them unchanged."""
        net = build_net("ping", hidden_sizes=[32])
        snapshots = {
            f"{name}_{k}": p.detach().clone()
            for name in ["W_ee", "W_ei", "W_ie"]
            for k, p in getattr(net, name).items()
        }
        # Optimize over all params, nudge with a fake loss from a trainable param
        trainable = [p for p in net.parameters() if p.requires_grad]
        assert len(trainable) > 0, "expected at least one trainable param"
        opt = torch.optim.SGD(trainable, lr=1.0)
        for p in trainable:
            p.grad = torch.ones_like(p)
        opt.step()
        for name in ["W_ee", "W_ei", "W_ie"]:
            for k, p in getattr(net, name).items():
                assert torch.equal(p, snapshots[f"{name}_{k}"]), \
                    f"{name}[{k}] changed after optimizer step"


class TestSeedReproducibility:
    @pytest.mark.parametrize("name", ["ping", "snntorch-clone"])
    def test_same_seed_gives_same_weights(self, name):
        def _weights(net):
            return [p.detach().clone() for p in net.parameters()]
        torch.manual_seed(123)
        a = _weights(build_net(name, hidden_sizes=[32]))
        torch.manual_seed(123)
        b = _weights(build_net(name, hidden_sizes=[32]))
        assert len(a) == len(b)
        for pa, pb in zip(a, b):
            assert torch.equal(pa, pb)

    def test_different_seeds_give_different_weights(self):
        torch.manual_seed(1)
        a = [p.detach().clone() for p in build_net("snntorch-clone", hidden_sizes=[32]).parameters()]
        torch.manual_seed(2)
        b = [p.detach().clone() for p in build_net("snntorch-clone", hidden_sizes=[32]).parameters()]
        # At least one parameter tensor must differ
        assert any(not torch.equal(pa, pb) for pa, pb in zip(a, b))


class TestKaimingMode:
    def test_kaiming_flag_switches_snntorch_canonical_to_tutorial(self):
        net = build_net("snntorch-clone", kaiming_init=True, hidden_sizes=[32])
        assert net.tutorial_readout is True
        assert net.reset_mode == "subtract"
