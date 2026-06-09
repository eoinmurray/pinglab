import pytest
import torch

from config import Config


class TestConfigDefaults:
    def test_instantiates_with_defaults(self):
        c = Config()
        assert c.n_e == 1024
        assert c.n_i == 256
        assert c.sim_ms == 600.0
        assert c.device == "cpu"

    def test_visible_ms_is_sim_minus_burn_in(self):
        c = Config(sim_ms=600.0, burn_in_ms=100.0)
        assert c.visible_ms == 500.0
        c2 = Config(sim_ms=400.0, burn_in_ms=50.0)
        assert c2.visible_ms == 350.0

    def test_torch_device_property(self):
        c = Config(device="cpu")
        assert c.torch_device == torch.device("cpu")

    def test_active_panels_is_per_instance(self):
        """Mutable default must not be shared between instances."""
        a = Config()
        b = Config()
        a.active_panels.append("custom")
        assert "custom" not in b.active_panels

    def test_overrides_take_effect(self):
        c = Config(n_e=128, n_i=32, sim_ms=200.0, seed=7)
        assert (c.n_e, c.n_i, c.sim_ms, c.seed) == (128, 32, 200.0, 7)
