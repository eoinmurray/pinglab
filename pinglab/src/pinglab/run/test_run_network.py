"""Tests for run_network function."""

import numpy as np
import pytest
from pinglab.run import run_network
from pinglab.types import NetworkConfig
from pinglab.inputs import tonic


class TestRunNetwork:
    """Integration tests for network simulation."""

    def make_config(self, **overrides) -> NetworkConfig:
        """Create a minimal network config for testing."""
        defaults = {
            "dt": 0.1,
            "T": 100.0,  # 100ms simulation
            "N_E": 10,
            "N_I": 5,
            "g_ei": 0.5,
            "g_ie": 1.0,
            "seed": 42,
            # LIF parameters
            "V_init": -65.0,
            "E_L": -65.0,
            "E_e": 0.0,
            "E_i": -80.0,
            "C_m_E": 1.0,
            "g_L_E": 0.1,
            "C_m_I": 1.0,
            "g_L_I": 0.1,
            "V_th": -50.0,
            "V_reset": -65.0,
            # Synaptic
            "tau_ampa": 5.0,
            "tau_gaba": 10.0,
            "t_ref_E": 2.0,
            "t_ref_I": 1.0,
            "delay_ei": 1.0,
            "delay_ie": 1.0,
            "delay_ee": 1.0,
            "delay_ii": 1.0,
        }
        defaults.update(overrides)
        return NetworkConfig(**defaults)

    def test_no_spikes_with_subthreshold_input(self):
        """With very low input, neurons should not spike."""
        config = self.make_config()
        num_steps = int(config.T / config.dt)
        external_input = tonic(
            N_E=config.N_E, N_I=config.N_I,
            I_E=0.0, I_I=0.0, noise_std=0.0, num_steps=num_steps, seed=42
        )

        result = run_network(config, external_input)

        assert len(result.spikes.times) == 0, "No spikes expected with zero input"

    def test_basic_spiking_with_strong_input(self):
        """With strong input, neurons should spike."""
        config = self.make_config()
        num_steps = int(config.T / config.dt)
        external_input = tonic(
            N_E=config.N_E, N_I=config.N_I,
            I_E=5.0, I_I=5.0, noise_std=0.5, num_steps=num_steps, seed=42
        )

        result = run_network(config, external_input)

        assert len(result.spikes.times) > 0, "Spikes expected with strong input"
        assert len(result.spikes.ids) == len(result.spikes.times)
        assert result.spikes.types is not None
        assert len(result.spikes.types) == len(result.spikes.times)

    def test_deterministic_with_same_seed(self):
        """Same seed should produce identical spike trains."""
        results = []
        for _ in range(2):
            config = self.make_config(seed=123)
            num_steps = int(config.T / config.dt)
            external_input = tonic(
                N_E=config.N_E, N_I=config.N_I,
                I_E=3.0, I_I=3.0, noise_std=0.5, num_steps=num_steps, seed=456
            )
            result = run_network(config, external_input)
            results.append(result)

        np.testing.assert_array_equal(
            results[0].spikes.times, results[1].spikes.times,
            "Spike times should be identical with same seed"
        )
        np.testing.assert_array_equal(
            results[0].spikes.ids, results[1].spikes.ids,
            "Spike IDs should be identical with same seed"
        )

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different spike trains."""
        results = []
        for seed in [42, 43]:
            config = self.make_config(seed=seed)
            num_steps = int(config.T / config.dt)
            external_input = tonic(
                N_E=config.N_E, N_I=config.N_I,
                I_E=3.0, I_I=3.0, noise_std=0.5, num_steps=num_steps, seed=seed
            )
            result = run_network(config, external_input)
            results.append(result)

        # Should have different spike trains (very unlikely to be identical)
        if len(results[0].spikes.times) > 0 and len(results[1].spikes.times) > 0:
            # At least one difference expected
            times_differ = not np.array_equal(
                results[0].spikes.times, results[1].spikes.times
            )
            ids_differ = not np.array_equal(
                results[0].spikes.ids, results[1].spikes.ids
            )
            assert times_differ or ids_differ, "Different seeds should produce different results"

    def test_refractory_period_limits_firing_rate(self):
        """Neurons should not fire faster than 1/t_ref."""
        t_ref = 5.0  # 5ms refractory = max 200 Hz
        config = self.make_config(t_ref_E=t_ref, t_ref_I=t_ref, T=500.0)
        num_steps = int(config.T / config.dt)
        # Very strong input to maximize firing
        external_input = tonic(
            N_E=config.N_E, N_I=config.N_I,
            I_E=20.0, I_I=20.0, noise_std=0.0, num_steps=num_steps, seed=42
        )

        result = run_network(config, external_input)

        if len(result.spikes.times) > 0:
            # Check each neuron's ISIs
            for nid in range(config.N_E + config.N_I):
                neuron_times = result.spikes.times[result.spikes.ids == nid]
                if len(neuron_times) > 1:
                    isis = np.diff(np.sort(neuron_times))
                    min_isi = np.min(isis)
                    # Allow small tolerance for dt discretization
                    assert min_isi >= t_ref - config.dt, \
                        f"Neuron {nid} has ISI {min_isi}ms < refractory {t_ref}ms"

    def test_dt_stability_check_raises_error(self):
        """Too large dt should raise ValueError."""
        # With g_L=0.1 and C_m=1.0, tau_mem = 10ms
        # dt > tau/5 = 2ms should raise error
        config = self.make_config(dt=3.0)  # Too large
        # Use ceil to match run_network's calculation
        num_steps = int(np.ceil(config.T / config.dt))
        external_input = tonic(
            N_E=config.N_E, N_I=config.N_I,
            I_E=1.0, I_I=1.0, noise_std=0.0, num_steps=num_steps, seed=42
        )

        with pytest.raises(ValueError, match="too large for numerical stability"):
            run_network(config, external_input)

    def test_spike_types_are_correct(self):
        """Spike types should correctly identify E (0) and I (1) neurons."""
        config = self.make_config()
        num_steps = int(config.T / config.dt)
        external_input = tonic(
            N_E=config.N_E, N_I=config.N_I,
            I_E=5.0, I_I=5.0, noise_std=0.5, num_steps=num_steps, seed=42
        )

        result = run_network(config, external_input)

        if len(result.spikes.times) > 0:
            for spike_id, spike_type in zip(result.spikes.ids, result.spikes.types):
                expected_type = 0 if spike_id < config.N_E else 1
                assert spike_type == expected_type, \
                    f"Neuron {spike_id} should have type {expected_type}, got {spike_type}"

    def test_broadcast_external_input(self):
        """1D external input should be broadcast to all neurons."""
        config = self.make_config()
        num_steps = int(config.T / config.dt)
        # 1D input (same for all neurons)
        external_input = np.full(num_steps, 3.0)

        result = run_network(config, external_input)

        # Should run without error and produce spikes
        assert len(result.spikes.times) > 0, "Should produce spikes with broadcast input"

    def test_per_neuron_external_input(self):
        """2D external input should work correctly."""
        config = self.make_config()
        num_steps = int(config.T / config.dt)
        N = config.N_E + config.N_I
        # 2D input (different per neuron)
        external_input = np.random.RandomState(42).uniform(2.0, 4.0, (num_steps, N))

        result = run_network(config, external_input)

        # Should run without error
        assert result.spikes is not None

    def test_spike_times_within_simulation_range(self):
        """All spike times should be within [0, T)."""
        config = self.make_config()
        num_steps = int(config.T / config.dt)
        external_input = tonic(
            N_E=config.N_E, N_I=config.N_I,
            I_E=5.0, I_I=5.0, noise_std=0.5, num_steps=num_steps, seed=42
        )

        result = run_network(config, external_input)

        if len(result.spikes.times) > 0:
            assert np.all(result.spikes.times >= 0), "Spike times should be >= 0"
            assert np.all(result.spikes.times < config.T), f"Spike times should be < T={config.T}"

    def test_spike_ids_within_valid_range(self):
        """All spike IDs should be valid neuron indices."""
        config = self.make_config()
        num_steps = int(config.T / config.dt)
        N = config.N_E + config.N_I
        external_input = tonic(
            N_E=config.N_E, N_I=config.N_I,
            I_E=5.0, I_I=5.0, noise_std=0.5, num_steps=num_steps, seed=42
        )

        result = run_network(config, external_input)

        if len(result.spikes.ids) > 0:
            assert np.all(result.spikes.ids >= 0), "Spike IDs should be >= 0"
            assert np.all(result.spikes.ids < N), f"Spike IDs should be < N={N}"
