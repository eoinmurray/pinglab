"""Structural gates for the exp022 matched_mid_v1 training generation."""

import pytest
from experiments import exp022, exp022_matched


def test_matched_registry_preserves_all_five_families_and_87_cells():
    assert len(exp022.MATCHED_CELLS) == 87
    assert {c["family"] for c in exp022.MATCHED_CELLS} == {
        "canonical", "theta_u", "tau_gaba", "dt", "init",
    }
    assert len({c["name"] for c in exp022.MATCHED_CELLS}) == 87


def test_matched_recipe_freezes_shared_scales():
    for recipe in exp022.MATCHED_MODEL_RECIPES.values():
        assert recipe["--w-in"] == "0.9"
        assert recipe["--readout-w-out-scale"] == "225"
    assert exp022.MATCHED_MODEL_RECIPES["coba"]["--ei-strength"] == "0"
    assert exp022.MATCHED_MODEL_RECIPES["coba"]["--v-grad-dampen"] == "1"
    assert exp022.MATCHED_MODEL_RECIPES["ping"]["--ei-strength"] == "1"
    assert exp022.MATCHED_MODEL_RECIPES["ping"]["--v-grad-dampen"] == "1000"


def test_canary_is_only_theta_off_seeds_43_and_44():
    cells = exp022.matched_canary_cells()
    assert len(cells) == 4
    assert {(c["model"], c["seed"]) for c in cells} == {
        ("coba", 43), ("coba", 44), ("ping", 43), ("ping", 44),
    }
    assert all(c["family"] == "theta_u" and c["theta_u"] is None for c in cells)


def test_remaining_batch_excludes_all_six_gate_cells():
    args = exp022_matched.parse_args(["--batch", "remaining"])
    cells = exp022_matched.selected_cells(args)
    assert len(cells) == 81
    assert not ({c["name"] for c in cells} & (
        exp022_matched.STAGE1_NAMES | exp022_matched.CANARY_NAMES
    ))


def test_expected_configs_record_every_recipe_varying_parameter():
    required = {
        "max_samples", "epochs", "seed", "dt", "t_ms", "tau_gaba_ms",
        "batch_size", "lr", "ei_strength", "v_grad_dampen", "w_in",
        "w_in_sparsity", "readout_w_out_scale", "fr_reg_upper_theta",
        "fr_reg_upper_strength", "trainable_w_ei", "trainable_w_ie",
    }
    for cell in exp022.MATCHED_CELLS:
        cfg = exp022.matched_expected_config(cell)
        assert required <= cfg.keys()
        assert cfg["w_in"] == pytest.approx([0.9, 0.09])
        assert cfg["readout_w_out_scale"] == 225.0
