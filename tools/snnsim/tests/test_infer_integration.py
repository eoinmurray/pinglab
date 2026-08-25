"""Integration tests for infer() and infer_and_snapshot() functions.

Tests parameter propagation, M module globals setup, output artifacts,
accuracy validation, and dataset-specific handling.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import models as M
import numpy as np
import pytest
import torch
from infer import infer, infer_and_snapshot
from train import train

# Whole file trains real (tiny) torch models via fixtures and runs inference —
# minutes of wall-clock. Marked slow so `pytest -m "not slow"` is a true fast lane.
pytestmark = pytest.mark.slow


@pytest.fixture
def tmp_output_dir():
    """Temporary directory for inference artifacts."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def trained_weights(tmp_output_dir):
    """Pre-trained weights for infer tests.

    Trains a tiny model on mnist to produce weights.pth.
    """
    train_dir = tmp_output_dir / "train"
    train_dir.mkdir(exist_ok=True)

    train(
        model_name="ping",
        dt=0.1,
        t_ms=100.0,
        epochs=2,
        dataset="mnist",
        max_samples=100,  # enough for stratified split
        lr=0.01,
        hidden_sizes=[32],
        seed=42,
        out_dir=train_dir,
    )
    weights_path = train_dir / "weights.pth"
    assert weights_path.exists(), "Training should produce weights.pth"
    return weights_path


class TestInferParameterPropagation:
    """Test that infer() parameters propagate correctly."""

    def test_dt_used_in_encoding(self, trained_weights, tmp_output_dir):
        """--dt should be used for spike encoding during inference."""
        dt_value = 0.3
        # Verify that different dt values still allow inference to complete
        result = infer(
            model_name="ping",
            dt=dt_value,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
        )
        assert "acc" in result

    def test_t_ms_propagates_to_m_module(self, trained_weights, tmp_output_dir):
        """--t-ms should set M.T_ms."""
        t_ms_value = 120.0
        infer(
            model_name="ping",
            dt=0.1,
            t_ms=t_ms_value,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
        )
        assert M.T_ms == t_ms_value

    def test_hidden_sizes_propagates_to_m_module(self, trained_weights):
        """An explicit hidden_sizes should set M.N_HID, M.N_INH, M.HIDDEN_SIZES.

        Must match the checkpoint's own shape ([32]); infer() now rebuilds the
        network to the loaded weights, so a mismatched override would fail
        load_state_dict rather than silently building the wrong size.
        """
        hidden_sizes_value = [32]
        infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=hidden_sizes_value,
        )
        assert M.N_HID == 32
        assert M.N_INH == 8  # 32 // 4
        assert M.HIDDEN_SIZES == [32]

    def test_seed_makes_deterministic_encoding(self, trained_weights, tmp_output_dir):
        """seed parameter should make Poisson encoding deterministic."""
        result1 = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            seed=42,
        )
        result2 = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            seed=42,
        )
        # Same seed should produce same accuracy (deterministic encoding)
        assert result1["acc"] == result2["acc"]


class TestMModuleGlobalsInference:
    """Test that M module globals are correctly initialized for inference."""

    def test_n_hid_n_inh_from_hidden_sizes(self, trained_weights):
        """M.N_INH should be derived from N_HID (n_e // 4) via hidden_sizes."""
        hidden_sizes = [32]  # must match the checkpoint infer() rebuilds against
        infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=hidden_sizes,
        )
        assert M.N_HID == 32
        assert M.N_INH == 8  # 32 // 4

    def test_hidden_sizes_none_derives_from_checkpoint(self, trained_weights):
        """hidden_sizes=None should recover the checkpoint's own layer sizes.

        The fixture trains at [32]; infer() reads that back from weights.pth
        rather than falling to the mnist smart default (1024), which would
        mismatch the saved weights.
        """
        infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=None,
        )
        assert M.N_HID == 32
        assert M.HIDDEN_SIZES == [32]

    def test_t_steps_computed_correctly(self, trained_weights):
        """M.T_steps should be int(T_ms / dt)."""
        dt = 0.2
        t_ms = 200.0
        expected_steps = int(t_ms / dt)
        infer(
            model_name="ping",
            dt=dt,
            t_ms=t_ms,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
        )
        assert M.T_steps == expected_steps


class TestInferReturnDict:
    """Test that infer() returns correctly structured accuracy dict."""

    def test_infer_returns_dict_with_acc_key(self, trained_weights):
        """infer() should return {'acc': float, 'rates_hz': dict, 'hid_rate_hz': float}."""
        result = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
        )
        assert isinstance(result, dict)
        assert "acc" in result
        assert isinstance(result["acc"], (float, np.floating))

    def test_infer_returns_rates_hz_dict(self, trained_weights):
        """infer() should return rates_hz dict."""
        result = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
        )
        assert "rates_hz" in result
        assert isinstance(result["rates_hz"], dict)

    def test_infer_returns_hid_rate_hz_float(self, trained_weights):
        """infer() should return hid_rate_hz as float or None."""
        result = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
        )
        assert "hid_rate_hz" in result
        assert result["hid_rate_hz"] is None or isinstance(result["hid_rate_hz"], (float, np.floating))

    def test_acc_is_in_valid_range(self, trained_weights):
        """Accuracy should be between 0 and 100."""
        result = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
        )
        assert 0.0 <= result["acc"] <= 100.0


class TestInferAndSnapshot:
    """Test infer_and_snapshot() function."""

    def test_infer_and_snapshot_creates_npz(self, trained_weights, tmp_output_dir):
        """infer_and_snapshot should create snapshot.npz."""
        infer_and_snapshot(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            hidden_sizes=[32],
            out_dir=tmp_output_dir,
        )
        snapshot_path = tmp_output_dir / "snapshot.npz"
        assert snapshot_path.exists(), "snapshot.npz should be created"

    def test_snapshot_npz_has_required_fields(self, trained_weights, tmp_output_dir):
        """snapshot.npz should contain spk_e, spk_i, dt, n_e, n_i."""
        infer_and_snapshot(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            hidden_sizes=[32],
            out_dir=tmp_output_dir,
        )
        snapshot_path = tmp_output_dir / "snapshot.npz"
        data = np.load(snapshot_path)

        # Check required fields
        required_fields = {"spk_e", "spk_i", "dt", "n_e", "n_i"}
        assert required_fields.issubset(set(data.files)), (
            f"snapshot.npz missing required fields. Got: {set(data.files)}"
        )

    def test_snapshot_npz_spk_e_shape(self, trained_weights, tmp_output_dir):
        """spk_e should have shape (T_steps, n_e)."""
        infer_and_snapshot(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            hidden_sizes=[32],
            out_dir=tmp_output_dir,
        )
        snapshot_path = tmp_output_dir / "snapshot.npz"
        data = np.load(snapshot_path)

        spk_e = data["spk_e"]
        n_e = int(data["n_e"])
        dt = float(data["dt"])
        t_ms = 100.0
        # round, not int: int(100.0 / 0.1) == 999 from float error, but the sim
        # runs the physically-correct 1000 steps.
        expected_t_steps = round(t_ms / dt)

        assert spk_e.shape == (expected_t_steps, n_e), (
            f"spk_e shape {spk_e.shape} doesn't match expected "
            f"({expected_t_steps}, {n_e})"
        )

    def test_snapshot_npz_spk_i_shape(self, trained_weights, tmp_output_dir):
        """spk_i should have shape (T_steps, n_i) or be empty."""
        infer_and_snapshot(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            hidden_sizes=[32],
            out_dir=tmp_output_dir,
        )
        snapshot_path = tmp_output_dir / "snapshot.npz"
        data = np.load(snapshot_path)

        spk_i = data["spk_i"]
        n_i = int(data["n_i"])
        dt = float(data["dt"])
        t_ms = 100.0
        expected_t_steps = round(t_ms / dt)  # round: see spk_e_shape test

        # spk_i should match n_i (second dimension)
        assert spk_i.shape[0] == expected_t_steps
        assert spk_i.shape[1] == n_i

    def test_snapshot_metadata_consistent(self, trained_weights, tmp_output_dir):
        """dt, n_e, n_i in snapshot should match M globals."""
        infer_and_snapshot(
            model_name="ping",
            dt=0.15,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            hidden_sizes=[32],
            out_dir=tmp_output_dir,
        )
        snapshot_path = tmp_output_dir / "snapshot.npz"
        data = np.load(snapshot_path)

        # dt is stored as float32 in the npz, so 0.15 does not round-trip exactly.
        assert float(data["dt"]) == pytest.approx(0.15)
        assert int(data["n_e"]) == M.N_HID
        assert int(data["n_i"]) == M.N_INH


class TestInferOutputArtifacts:
    """Test output files created by infer()."""

    def test_infer_creates_metrics_json(self, trained_weights, tmp_output_dir):
        """infer() should create metrics.json when out_dir is provided."""
        infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
            out_dir=tmp_output_dir,
        )
        metrics_path = tmp_output_dir / "metrics.json"
        assert metrics_path.exists(), "metrics.json should be created when out_dir exists"

    def test_infer_metrics_json_structure(self, trained_weights, tmp_output_dir):
        """metrics.json should have correct structure."""
        infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
            out_dir=tmp_output_dir,
        )
        with open(tmp_output_dir / "metrics.json") as f:
            metrics = json.load(f)

        assert metrics["mode"] == "infer"
        assert "best_acc" in metrics
        assert "n_correct" in metrics
        assert "n_total" in metrics
        assert "config" in metrics


class TestInferHiddenSizesAutoDetection:
    """Test auto-detection of hidden_sizes per dataset."""

    def test_hidden_sizes_auto_mnist(self, tmp_output_dir):
        """MNIST should have its own default hidden size."""
        # First train with MNIST
        train_dir = tmp_output_dir / "train_mnist"
        train_dir.mkdir()
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=1,
            dataset="mnist",
            max_samples=50,
            lr=0.01,
            hidden_sizes=None,  # auto
            seed=42,
            out_dir=train_dir,
        )
        weights_path = train_dir / "weights.pth"

        # Now infer with same auto-detection
        infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=weights_path,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=None,
        )
        # MNIST default is 1024 (from DATASET_N_HIDDEN_DEFAULTS)
        assert M.N_HID == 1024


class TestInferAccuracyValidation:
    """Test that inferred accuracy is reasonable."""

    @pytest.mark.slow
    def test_trained_model_achieves_reasonable_accuracy(self, trained_weights):
        """Model trained on mnist should beat the random baseline."""
        result = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=None,  # Use all test data
            hidden_sizes=[32],
        )
        # The shared fixture trains only briefly on a small mnist subset, so
        # this is a smoke check that learning happened at all: beat chance
        # (10% for 10 classes) with a small margin.
        assert result["acc"] > 12.0, (
            f"Expected accuracy > 12% on mnist after training, "
            f"got {result['acc']:.1f}%"
        )

    @pytest.mark.slow
    def test_untrained_model_near_random_baseline(self, tmp_output_dir):
        """Random (untrained) weights should give near-random accuracy."""
        # Create weights without training
        from config import build_net

        # Build the untrained net at mnist's input size so its W_in matches the
        # net infer() builds for dataset="mnist" (n_in=784); the autouse fixture
        # otherwise leaves M.N_IN at the module default of 64.
        M.N_IN = 784
        net = build_net("ping", hidden_sizes=[32])
        weights_dir = tmp_output_dir / "untrained"
        weights_dir.mkdir()
        weights_path = weights_dir / "weights.pth"
        torch.save(net.state_dict(), weights_path)

        result = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=weights_path,
            dataset="mnist",
            max_samples=100,
            hidden_sizes=[32],
        )
        # Random baseline for 10-class is 10%, allow 0-30% for variance
        assert 0.0 <= result["acc"] <= 30.0


class TestInferDatasetSpecific:
    """Test infer() with different datasets."""

    def test_infer_sets_n_in_for_mnist(self, tmp_output_dir):
        """MNIST inference should set N_IN=784."""
        # Train a tiny MNIST model first
        train_dir = tmp_output_dir / "train_mnist_ni"
        train_dir.mkdir()
        train(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            epochs=1,
            dataset="mnist",
            max_samples=100,
            hidden_sizes=[32],
            seed=42,
            out_dir=train_dir,
        )
        weights_path = train_dir / "weights.pth"

        infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=weights_path,
            dataset="mnist",
            max_samples=50,
        )
        assert M.N_IN == 784


class TestInferWeightsLoading:
    """Test weight loading behavior."""

    def test_infer_requires_load_weights(self, trained_weights):
        """infer() should require load_weights to be set."""
        with pytest.raises(AssertionError):
            infer(
                model_name="ping",
                dt=0.1,
                t_ms=100.0,
                load_weights=None,  # Should fail
                dataset="mnist",
                max_samples=50,
            )

    def test_infer_loads_state_dict_correctly(self, trained_weights):
        """Loaded state dict should be applied to network."""
        result = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
        )
        # If loading worked, we should get a result with valid accuracy
        assert "acc" in result


class TestInferRateComputation:
    """Test firing rate computation during inference."""

    def test_rates_hz_dict_has_populations(self, trained_weights):
        """rates_hz dict should have population keys."""
        result = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
        )
        rates_hz = result["rates_hz"]
        # Should have at least one hidden population rate
        hid_keys = [k for k in rates_hz.keys() if k.startswith("hid")]
        assert len(hid_keys) > 0, "Should compute rates for hidden populations"

    def test_hid_rate_hz_is_positive(self, trained_weights):
        """hid_rate_hz should be positive when neurons are firing."""
        result = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            hidden_sizes=[32],
        )
        hid_rate = result["hid_rate_hz"]
        # Trained network should have some activity
        if hid_rate is not None:
            assert hid_rate > 0.0, "Hidden neurons should have positive firing rate"


class TestInferDeterminism:
    """Test that infer() is deterministic given same seed and weights."""

    def test_same_weights_seed_gives_same_accuracy(self, trained_weights):
        """Same weights and seed should give identical accuracy."""
        result1 = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=100,
            seed=123,
        )
        result2 = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=100,
            seed=123,
        )
        # Deterministic encoding should give same accuracy
        assert result1["acc"] == result2["acc"]

    def test_different_seeds_may_give_different_accuracy(self, trained_weights):
        """Different seeds (different Poisson encodings) may give different accuracy."""
        result1 = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            seed=42,
        )
        result2 = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            seed=999,
        )
        # May be same by chance, but could differ due to different encodings
        # This is a loose test — just verify both are valid
        assert 0 <= result1["acc"] <= 100
        assert 0 <= result2["acc"] <= 100


class TestInferDalesLaw:
    """Test dales_law parameter in infer()."""

    def test_infer_with_dales_law_true(self, trained_weights):
        """dales_law=True should work with trained weights."""
        result = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            dales_law=True,
        )
        assert result["acc"] >= 0.0

    def test_infer_with_dales_law_false(self, trained_weights):
        """dales_law=False should work (signed weights)."""
        result = infer(
            model_name="ping",
            dt=0.1,
            t_ms=100.0,
            load_weights=trained_weights,
            dataset="mnist",
            max_samples=50,
            dales_law=False,
        )
        assert result["acc"] >= 0.0
