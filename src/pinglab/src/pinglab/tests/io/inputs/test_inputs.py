import numpy as np
import pytest

from pinglab.io import (
    add_pulse_train_to_input,
    external_spike_train,
    oscillating,
    pulse,
    ramp,
)
from pinglab.backends.types import Spikes


def test_ramp_shape_and_seed():
    out1 = ramp(
        N_E=2,
        N_I=1,
        I_E_start=0.5,
        I_E_end=1.0,
        I_I_start=0.2,
        I_I_end=0.8,
        noise_std_E=0.1,
        noise_std_I=0.1,
        num_steps=5,
        dt=1.0,
        seed=123,
    )
    out2 = ramp(
        N_E=2,
        N_I=1,
        I_E_start=0.5,
        I_E_end=1.0,
        I_I_start=0.2,
        I_I_end=0.8,
        noise_std_E=0.1,
        noise_std_I=0.1,
        num_steps=5,
        dt=1.0,
        seed=123,
    )
    assert out1.shape == (5, 3)
    np.testing.assert_allclose(out1, out2)
    assert np.all(out1 >= 0.0), "Ramp input should be clamped to non-negative values"


def test_ramp_invalid_args():
    with pytest.raises(ValueError):
        ramp(
            N_E=-1,
            N_I=1,
            I_E_start=0.0,
            I_E_end=0.0,
            I_I_start=0.0,
            I_I_end=0.0,
            noise_std_E=0.0,
            noise_std_I=0.0,
            num_steps=1,
            dt=1.0,
            seed=0,
        )
    with pytest.raises(ValueError):
        ramp(
            N_E=1,
            N_I=1,
            I_E_start=0.0,
            I_E_end=0.0,
            I_I_start=0.0,
            I_I_end=0.0,
            noise_std_E=-0.1,
            noise_std_I=0.0,
            num_steps=1,
            dt=1.0,
            seed=0,
        )
    with pytest.raises(ValueError):
        ramp(
            N_E=1,
            N_I=1,
            I_E_start=0.0,
            I_E_end=0.0,
            I_I_start=0.0,
            I_I_end=0.0,
            noise_std_E=0.0,
            noise_std_I=0.0,
            num_steps=0,
            dt=1.0,
            seed=0,
        )


def test_oscillating_phase_offset():
    out = oscillating(
        N_E=1,
        N_I=1,
        I_E=0.0,
        I_I=0.0,
        noise_std=0.0,
        num_steps=100,
        dt=1.0,
        seed=0,
        oscillation_freq=10.0,
        oscillation_amplitude=1.0,
        oscillation_phase=0.0,
        phase_offset_I=np.pi,
    )
    # With pi offset, E and I should be opposite at t=0
    assert out[0, 0] == pytest.approx(0.0)
    assert out[0, 1] == pytest.approx(0.0)
    # At quarter period, sin and sin+pi are opposite signs
    quarter = int((1000.0 / 10.0) / 4.0)
    assert out[quarter, 0] == pytest.approx(-out[quarter, 1])


def test_add_pulse_to_input_and_spike_delta():
    base = np.zeros((10, 3))
    pulse_input = pulse.add_pulse_to_input(
        base,
        target_neurons=np.array([1, 2]),
        pulse_t=2.0,
        pulse_width_ms=4.0,
        pulse_amp=5.0,
        dt=1.0,
    )
    # Pulse should affect steps [2, 6)
    assert np.all(pulse_input[2:6, 1:3] == 5.0)
    assert np.all(pulse_input[:2] == 0.0)
    assert np.all(pulse_input[6:] == 0.0)

    spikes = Spikes(
        times=np.array([1.0, 2.5, 3.0, 7.0]),
        ids=np.array([1, 1, 2, 2]),
    )
    delta = pulse.compute_spike_delta(
        spikes,
        target_neurons=np.array([1, 2]),
        pulse_t=2.0,
        pre_window_ms=1.0,
        post_window_ms=2.0,
    )
    assert delta == 1


def test_add_pulse_train_to_input():
    base = np.zeros((12, 4))
    out = add_pulse_train_to_input(
        base,
        target_neurons=np.array([0, 1]),
        pulse_t=2.0,
        pulse_width_ms=2.0,
        pulse_amp=3.0,
        pulse_interval_ms=4.0,
        dt=1.0,
    )
    assert np.all(out[2:4, 0:2] == 3.0)
    assert np.all(out[6:8, 0:2] == 3.0)


def test_external_spike_train_shape_seed_and_spikes():
    out1, spikes_e_1, spikes_i_1 = external_spike_train(
        N_E=4,
        N_I=2,
        I_E_base=0.1,
        I_I_base=0.0,
        noise_std_E=0.0,
        noise_std_I=0.0,
        num_steps=200,
        dt=1.0,
        seed=3,
        lambda0_hz=30.0,
        mod_depth=0.4,
        envelope_freq_hz=6.0,
        phase_rad=0.3,
        w_in=0.2,
        tau_in_ms=3.0,
        return_spikes=True,
    )
    out2, spikes_e_2, spikes_i_2 = external_spike_train(
        N_E=4,
        N_I=2,
        I_E_base=0.1,
        I_I_base=0.0,
        noise_std_E=0.0,
        noise_std_I=0.0,
        num_steps=200,
        dt=1.0,
        seed=3,
        lambda0_hz=30.0,
        mod_depth=0.4,
        envelope_freq_hz=6.0,
        phase_rad=0.3,
        w_in=0.2,
        tau_in_ms=3.0,
        return_spikes=True,
    )

    assert out1.shape == (200, 6)
    assert spikes_e_1.shape == (200, 4)
    assert spikes_i_1.shape == (200, 2)
    assert np.all(spikes_i_1 == 0)
    np.testing.assert_allclose(out1, out2)
    assert np.array_equal(spikes_e_1, spikes_e_2)
    assert np.array_equal(spikes_i_1, spikes_i_2)
    assert np.all(out1[:, :4] >= 0.0)


def test_external_spike_train_independence_sanity():
    _, spikes_e, _ = external_spike_train(
        N_E=6,
        N_I=0,
        I_E_base=0.0,
        I_I_base=0.0,
        noise_std_E=0.0,
        noise_std_I=0.0,
        num_steps=4000,
        dt=1.0,
        seed=11,
        lambda0_hz=40.0,
        mod_depth=0.0,
        envelope_freq_hz=5.0,
        phase_rad=0.0,
        w_in=0.2,
        tau_in_ms=3.0,
        return_spikes=True,
    )
    assert spikes_e.shape == (4000, 6)
    assert not np.array_equal(spikes_e[:, 0], spikes_e[:, 1])
    corr = np.corrcoef(spikes_e[:, 0].astype(float), spikes_e[:, 1].astype(float))[0, 1]
    assert abs(float(corr)) < 0.2
