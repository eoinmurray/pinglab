"""Coverage-focused tests for config.py.

Complements test_config.py (which covers the Config dataclass surface:
defaults, torch_device, apply_frame_param, sync_from_model). This file
exercises the module-level machinery instead: model-globals setup, dt
pinning, NPZ snapshot serialisation, the network builder, the simulation
runners, and the record/weight extraction helpers.

Everything here is deliberately tiny (dt=0.1, t_ms=100, hidden=[16..32],
single forward passes, synthetic tensors) so the whole file runs in a
couple of seconds. The shared conftest autouse fixture already resets the
models.py globals between tests, so we never need our own reset.
"""

import config as C
import models as M
import numpy as np
import pytest
import torch
from config import (
    _build_sim_net,
    _extract_records,
    build_net,
    extract_weights,
    make_net,
    run_sim,
    run_sim_image,
    save_snapshot_npz,
    set_sim_dt,
    setup_model_globals,
)

# Small, fast simulation parameters shared across tests.
DT = 0.1
T_MS = 100.0


# ---------------------------------------------------------------------------
# setup_model_globals / set_sim_dt
# ---------------------------------------------------------------------------


class TestSetupModelGlobals:
    def test_normal_list_multi_layer(self):
        setup_model_globals([16, 32])
        # N_HID is the LAST (deepest) layer; N_INH is that // 4.
        assert M.N_HID == 32
        assert M.N_INH == 8
        assert M.HIDDEN_SIZES == [16, 32]

    def test_single_layer(self):
        setup_model_globals([16])
        assert M.N_HID == 16
        assert M.N_INH == 4
        assert M.HIDDEN_SIZES == [16]

    def test_empty_list_defaults_to_256(self):
        # Falsy hidden_sizes -> default [256].
        setup_model_globals([])
        assert M.N_HID == 256
        assert M.N_INH == 64
        assert M.HIDDEN_SIZES == [256]

    def test_none_defaults_to_256(self):
        setup_model_globals(None)
        assert M.N_HID == 256
        assert M.HIDDEN_SIZES == [256]


class TestSetSimDt:
    def test_pins_dt_and_derived_steps(self):
        set_sim_dt(DT, T_MS)
        assert M.dt == pytest.approx(0.1)
        assert M.T_ms == pytest.approx(100.0)
        assert M.T_steps == int(T_MS / DT)  # 1000

    def test_coerces_to_float_and_int(self):
        set_sim_dt(1, 50)  # ints in
        assert isinstance(M.dt, float)
        assert isinstance(M.T_ms, float)
        assert isinstance(M.T_steps, int)
        assert M.T_steps == 50


# ---------------------------------------------------------------------------
# save_snapshot_npz
# ---------------------------------------------------------------------------


class TestSaveSnapshotNpz:
    def test_single_layer_rec_with_inhibition(self, tmp_path):
        # Single-layer rec: 'hid'/'inh'/'input' + an extra voltage field.
        T = 5
        rec = {
            "hid": torch.zeros(T, 8),
            "inh": torch.zeros(T, 2),
            "input": torch.zeros(T, 8),
            "v_e": np.zeros((T, 8), dtype=np.float32),
        }
        out = tmp_path / "snap.npz"
        save_snapshot_npz(out, rec, dt=DT, n_e=8, n_i=2, label=3)

        with np.load(out) as data:
            assert data["dt"] == pytest.approx(0.1)
            assert int(data["n_e"]) == 8
            assert int(data["n_i"]) == 2
            assert int(data["label"]) == 3
            # Canonicalised spike names.
            assert data["spk_e"].shape == (T, 8)
            assert data["spk_i"].shape == (T, 2)
            # 'input' renamed to 'input_spikes'.
            assert data["input_spikes"].shape == (T, 8)
            # Other fields kept under original name.
            assert data["v_e"].shape == (T, 8)

    def test_no_inhibition_creates_empty_spk_i(self, tmp_path):
        # rec with no 'inh' key -> spk_i is an empty (T, 0) array.
        T = 4
        rec = {"hid": np.zeros((T, 6), dtype=np.float32)}
        out = tmp_path / "noinh.npz"
        save_snapshot_npz(out, rec, dt=DT, n_e=6, n_i=0)

        with np.load(out) as data:
            assert data["spk_e"].shape == (T, 6)
            assert data["spk_i"].shape == (T, 0)

    def test_display_fallback_supplies_input_spikes(self, tmp_path):
        # rec without 'input'; display arg fills input_spikes.
        T = 3
        rec = {"hid": np.zeros((T, 4), dtype=np.float32)}
        display = torch.ones(T, 4)
        out = tmp_path / "disp.npz"
        save_snapshot_npz(out, rec, dt=DT, n_e=4, n_i=0, display=display)

        with np.load(out) as data:
            assert data["input_spikes"].shape == (T, 4)
            assert data["input_spikes"].sum() == pytest.approx(T * 4)

    def test_numpy_rec_values_pass_through(self, tmp_path):
        # numpy arrays (no .numpy()) must serialise unchanged.
        T = 2
        rec = {
            "hid": np.ones((T, 3), dtype=np.float32),
            "inh": np.ones((T, 1), dtype=np.float32),
        }
        out = tmp_path / "npy.npz"
        save_snapshot_npz(out, rec, dt=0.25, n_e=3, n_i=1)
        with np.load(out) as data:
            assert data["spk_e"].sum() == pytest.approx(T * 3)
            assert data["spk_i"].shape == (T, 1)


# ---------------------------------------------------------------------------
# build_net / _build_sim_net
# ---------------------------------------------------------------------------


class TestBuildNet:
    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_net("does-not-exist")

    def test_ping_single_layer_dales_law(self):
        setup_model_globals([32])
        net = build_net("ping", hidden_sizes=[32], dales_law=True)
        # W_ff: input->hid, hid->out => 2 feedforward matrices.
        assert len(net.W_ff) == 2
        # First matrix maps N_IN -> 32.
        assert net.W_ff[0].shape[1] == 32
        # Last matrix maps 32 -> N_OUT.
        assert net.W_ff[-1].shape[0] == 32
        assert net.signed_weights is False  # dales_law True

    def test_ping_signed_weights_when_no_dales(self):
        net = build_net("ping", hidden_sizes=[16], dales_law=False)
        assert net.signed_weights is True

    def test_multi_layer_hidden_sizes(self):
        net = build_net("ping", hidden_sizes=[16, 32])
        # input->h1, h1->h2, h2->out => 3 feedforward matrices.
        assert len(net.W_ff) == 3
        assert net.W_ff[0].shape[1] == 16
        assert net.W_ff[1].shape == (16, 32)
        assert net.W_ff[-1].shape[0] == 32
        assert net.hidden_sizes == [16, 32]

    def test_ei_layers_subset(self):
        # Two hidden layers, only layer 1 gets E-I structure.
        net = build_net("ping", hidden_sizes=[16, 32], ei_layers=[1])
        assert net.ei_layers == {1}
        assert set(net.W_ei.keys()) == {"1"}

    def test_explicit_weight_specs_and_ei_strength(self):
        # Exercise the w_in / w_ee / w_ei / w_ie / w_ii spec branches plus
        # the ei_strength derivation and trainable flags.
        net = build_net(
            "ping",
            hidden_sizes=[16],
            w_in=(0.3, 0.06),
            w_in_sparsity=0.5,
            w_ee=(0.1, 0.01),
            w_ii=(0.2, 0.02),
            ei_strength=1.0,
            ei_ratio=2.0,
            sparsity=0.2,
            trainable_w_ei=True,
            trainable_w_ie=True,
            trainable_w_ii=True,
        )
        assert len(net.W_ff) == 2
        # ei_strength wired trainable E-I weights.
        assert net.W_ei["1"].requires_grad is True
        assert net.W_ie["1"].requires_grad is True

    def test_sparsity_only_branch(self):
        # No explicit E-I specs and no ei_strength but sparsity>0 -> setdefault.
        net = build_net("ping", hidden_sizes=[16], sparsity=0.3)
        assert len(net.W_ff) == 2

    def test_device_and_n_inh_per_layer(self):
        net = build_net(
            "ping",
            hidden_sizes=[16],
            device="cpu",
            n_inh_per_layer={1: 8},
        )
        # Explicit N_I override honoured.
        assert net.W_ei["1"].shape[1] == 8

    def test_build_sim_net_spike_input_true(self):
        setup_model_globals([16])
        set_sim_dt(DT, T_MS)
        net = _build_sim_net("ping", spike_input=True, hidden_sizes=[16])
        # With spike_input, real input synapses are built (W_ff[0] non-zero-able).
        assert len(net.W_ff) == 2

    def test_build_sim_net_conductance_drive(self):
        setup_model_globals([16])
        net = _build_sim_net("ping", spike_input=False, hidden_sizes=[16])
        # Conductance path zeroes W_in -> input matrix is all zeros.
        assert np.allclose(net.W_ff[0].data.cpu().numpy(), 0.0)


# ---------------------------------------------------------------------------
# make_net
# ---------------------------------------------------------------------------


class TestMakeNet:
    def test_default_w_in(self):
        cfg_obj = C.Config(n_e=16, n_i=4, seed=1)
        net = make_net(cfg_obj)
        assert net.recording is True
        assert len(net.W_ff) == 2

    def test_legacy_4tuple_w_in(self):
        cfg_obj = C.Config(n_e=16, n_i=4)
        # 4-tuple (mean, std, dist, sparsity) form.
        net = make_net(cfg_obj, w_in=(0.3, 0.06, "normal", 0.9))
        assert net.recording is True

    def test_short_w_in_pair(self):
        cfg_obj = C.Config(n_e=16, n_i=4)
        net = make_net(cfg_obj, w_in=(0.3, 0.06))
        assert net.recording is True

    def test_non_default_n_i_triggers_per_layer_override(self):
        # n_i != n_e // 4 -> n_inh_per_layer={1: n_i}.
        cfg_obj = C.Config(n_e=16, n_i=8)  # 16//4 == 4, so 8 differs
        net = make_net(cfg_obj)
        assert net.W_ei["1"].shape[1] == 8


# ---------------------------------------------------------------------------
# run_sim
# ---------------------------------------------------------------------------


class TestRunSim:
    def test_conductance_drive_path(self):
        # Default path: no override, no spikes -> make_step_drive builds ext_g.
        C.cfg.n_e = 16
        C.cfg.n_i = 4
        C.cfg.sim_ms = T_MS
        rec, display, weights = run_sim(DT, t_e_ping=0.3)
        assert "hid" in rec
        assert isinstance(weights, dict)
        # display is the conductance drive: (T_steps, n_e).
        assert display.shape[0] == int(T_MS / DT)
        assert display.shape[1] == 16

    def test_ext_g_override_path(self):
        C.cfg.n_e = 16
        C.cfg.n_i = 4
        C.cfg.sim_ms = T_MS
        set_sim_dt(DT, T_MS)
        T = 30  # shorter than sim_ms/dt -> exercises the min() clamp
        override = torch.zeros(T, 16)
        rec, display, weights = run_sim(
            DT, t_e_ping=0.3, ext_g_override=override
        )
        assert display.shape == (T, 16)
        # T_steps clamped down to the supplied tensor length.
        assert M.T_steps == T

    def test_ext_g_override_as_numpy(self):
        C.cfg.n_e = 16
        C.cfg.n_i = 4
        C.cfg.sim_ms = T_MS
        override = np.zeros((20, 16), dtype=np.float32)
        rec, display, weights = run_sim(
            DT, t_e_ping=0.3, ext_g_override=override
        )
        assert display.shape == (20, 16)

    def test_input_spikes_path(self):
        C.cfg.n_e = 16
        C.cfg.n_i = 4
        C.cfg.sim_ms = T_MS
        set_sim_dt(DT, T_MS)
        T = 25
        # Input spikes feed THROUGH W_in; shape (T, B, N_IN).
        spikes = torch.zeros(T, 1, M.N_IN)
        rec, display, weights = run_sim(
            DT, t_e_ping=0.3, input_spikes=spikes, t_e_async=0.0006
        )
        assert display.shape[0] == T
        assert M.T_steps == T


# ---------------------------------------------------------------------------
# run_sim_image
# ---------------------------------------------------------------------------


class TestRunSimImage:
    def test_image_forward_returns_prediction(self):
        C.cfg.n_e = 16
        C.cfg.n_i = 4
        C.cfg.sim_ms = T_MS
        set_sim_dt(DT, T_MS)
        # Flat image vector; N_IN is set from image.shape[0].
        image = np.zeros(12, dtype=np.float32)
        rec, pred, net = run_sim_image(DT, image)
        assert M.N_IN == 12
        assert isinstance(pred, int)
        assert 0 <= pred < M.N_OUT
        assert "hid" in rec


# ---------------------------------------------------------------------------
# _extract_records / extract_weights
# ---------------------------------------------------------------------------


class TestExtractRecords:
    def test_stacks_lists_and_copies_tensors(self):
        setup_model_globals([16])
        set_sim_dt(DT, T_MS)
        net = _build_sim_net("ping", spike_input=False, hidden_sizes=[16])
        net.recording = True
        # forward() loops M.T_steps times, so the drive tensor must be that long.
        M.T_steps = 20
        ext_g = torch.zeros(20, 16)
        with torch.no_grad():
            net.forward(ext_g=ext_g)
        rec = _extract_records(net)
        # Every recorded value became a numpy array.
        assert all(isinstance(v, np.ndarray) for v in rec.values())
        assert "hid" in rec


class TestExtractWeights:
    def test_new_style_w_ff_and_dicts(self):
        net = build_net("ping", hidden_sizes=[16, 32], ei_layers=[1])
        weights = extract_weights(net)
        # Multi-layer W_ff -> W_in, W_ff_2..., W_out named entries.
        assert "W_in" in weights
        assert "W_out" in weights
        # Single-key ParameterDicts collapse to the dict name.
        assert "W_ei" in weights
        assert "W_ie" in weights
        # All values are flat numpy arrays.
        assert all(isinstance(v, np.ndarray) and v.ndim == 1 for v in weights.values())

    def test_multi_key_dicts_get_suffixed(self):
        # Two E-I layers -> W_ei has keys '1' and '2' -> suffixed names.
        net = build_net("ping", hidden_sizes=[16, 32], ei_layers=[1, 2])
        weights = extract_weights(net)
        assert "W_ei_1" in weights
        assert "W_ei_2" in weights


# ---------------------------------------------------------------------------
# Module-level __getattr__ aliases
# ---------------------------------------------------------------------------


class TestModuleGetattr:
    def test_cfg_field_alias(self):
        C.cfg.n_e = 128
        assert C.N_E == 128  # resolves via _CFG_ALIASES

    def test_tuple_alias(self):
        C.cfg.w_ei = (0.7, 0.07)
        assert C.W_EI == (0.7, 0.07)

    def test_device_alias(self):
        assert C.DEVICE == C.cfg.torch_device

    def test_artifact_root_alias_is_path(self):
        from pathlib import Path

        assert isinstance(C.ARTIFACT_ROOT, Path)

    def test_unknown_attr_raises(self):
        with pytest.raises(AttributeError, match="no attribute"):
            _ = C.DOES_NOT_EXIST


# ---------------------------------------------------------------------------
# build_config / _sync_globals_from_cfg
# ---------------------------------------------------------------------------


class _Args:
    """Minimal argparse-Namespace stand-in for build_config."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class TestBuildConfig:
    def test_defaults_without_optional_attrs(self):
        # Bare args: only the getattr-with-default branches fire.
        c = C.build_config(_Args())
        assert c.artifact_root == str(C.DEFAULT_ARTIFACT_ROOT)
        assert c.raster_mode == "scatter"
        # _sync_globals_from_cfg installed this Config as the module cfg.
        assert C.cfg is c

    def test_full_args_wire_every_branch(self):
        args = _Args(
            out_dir="/tmp/pinglab-test-out",
            n_hidden=[16, 32],
            device="cpu",
            raster="line",
            drive=0.5,
            ei_ratio=3.0,
            ei_strength=1.0,
            w_ei=[0.4, 0.04],
            w_ie=[0.8, 0.08],
            w_ee=[0.1, 0.01],
            w_in=[0.3],  # single value -> std derived as 10%
            input="synthetic-spikes",
            sparsity=0.25,
            w_in_sparsity=0.9,
            bias=0.0005,
            t_ms=150.0,
        )
        c = C.build_config(args)
        assert c.artifact_root == "/tmp/pinglab-test-out"
        # n_hidden list -> last element is n_e.
        assert c.n_e == 32
        assert c.n_i == 8
        assert c.raster_mode == "line"
        assert c.t_e_async == pytest.approx(0.5)
        assert c.ei_ratio == pytest.approx(3.0)
        # Explicit w_ei/w_ie/w_ee override the ei_strength-derived ones.
        assert c.w_ei == (0.4, 0.04)
        assert c.w_ie == (0.8, 0.08)
        assert c.w_ee == (0.1, 0.01)
        # Single-value w_in -> (mean, 10% mean).
        assert c.w_in_spikes == (0.3, pytest.approx(0.03))
        assert c.sparsity == pytest.approx(0.25)
        assert c.w_in_sparsity == pytest.approx(0.9)
        assert c.bias == pytest.approx(0.0005)
        assert c.sim_ms == pytest.approx(150.0)
        # synthetic-spikes without n_input -> N_IN mirrors n_e.
        assert M.N_IN == 32

    def test_n_input_sets_module_global(self):
        args = _Args(n_input=20, input="synthetic-spikes")
        C.build_config(args)
        assert M.N_IN == 20

    def test_dataset_input_sets_rate_globals(self):
        args = _Args(input="dataset", spike_rate=123.0)
        C.build_config(args)
        assert M.max_rate_hz == pytest.approx(123.0)
        assert M.p_scale == pytest.approx(123.0 * M.dt / 1000.0)

    def test_ei_strength_without_explicit_specs(self):
        # ei_strength present, no w_ei/w_ie -> derived tuples.
        args = _Args(ei_strength=2.0, ei_ratio=2.0)
        c = C.build_config(args)
        assert c.w_ei == (2.0, pytest.approx(0.2))
        assert c.w_ie == (4.0, pytest.approx(0.4))
