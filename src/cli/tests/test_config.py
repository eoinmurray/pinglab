import torch
from config import Config


class TestConfigDefaults:
    def test_instantiates_with_defaults(self):
        c = Config()
        assert c.n_e == 1024
        assert c.n_i == 256
        assert c.sim_ms == 600.0
        assert c.device == "cpu"

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


class TestConfigMutations:
    """Test the documented mutation methods (apply_frame_param, sync_from_model)."""

    def test_apply_frame_param_w_ei_mean(self):
        c = Config(w_ei=(1.0, 0.1))
        c.apply_frame_param("w_ei_mean", 2.5)
        assert c.w_ei == (2.5, 0.1)

    def test_apply_frame_param_w_ie_mean(self):
        c = Config(w_ie=(3.0, 0.3))
        c.apply_frame_param("w_ie_mean", 1.2)
        assert c.w_ie == (1.2, 0.3)

    def test_apply_frame_param_ei_strength(self):
        c = Config(ei_ratio=2.0)
        c.apply_frame_param("ei_strength", 1.0)
        assert c.w_ei == (1.0, 0.1)
        assert c.w_ie == (2.0, 0.2)  # ei_ratio * (1.0, 0.1)

    def test_apply_frame_param_bias(self):
        c = Config(bias=0.0001)
        c.apply_frame_param("bias", 0.0005)
        assert c.bias == 0.0005

    def test_apply_frame_param_unknown_raises(self):
        c = Config()
        import pytest

        with pytest.raises(ValueError, match="Unknown frame param"):
            c.apply_frame_param("tau_gaba", 5.0)

    def test_sync_from_model(self):
        c = Config(n_e=1024, n_i=256)
        c.sync_from_model(n_hid=512, n_inh=128)
        assert c.n_e == 512
        assert c.n_i == 128
