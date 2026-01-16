import numpy as np
import pytest

from pinglab.inputs import tonic, oscillating, pulse
from pinglab.types import Spikes


def test_tonic_shape_and_seed():
    out1 = tonic(N_E=2, N_I=1, I_E=1.0, I_I=2.0, noise_std=0.1, num_steps=5, seed=123)
    out2 = tonic(N_E=2, N_I=1, I_E=1.0, I_I=2.0, noise_std=0.1, num_steps=5, seed=123)
    assert out1.shape == (5, 3)
    np.testing.assert_allclose(out1, out2)


def test_tonic_invalid_args():
    with pytest.raises(ValueError):
        tonic(N_E=-1, N_I=1, I_E=0.0, I_I=0.0, noise_std=0.0, num_steps=1, seed=0)
    with pytest.raises(ValueError):
        tonic(N_E=1, N_I=-1, I_E=0.0, I_I=0.0, noise_std=0.0, num_steps=1, seed=0)
    with pytest.raises(ValueError):
        tonic(N_E=1, N_I=1, I_E=0.0, I_I=0.0, noise_std=-0.1, num_steps=1, seed=0)
    with pytest.raises(ValueError):
        tonic(N_E=1, N_I=1, I_E=0.0, I_I=0.0, noise_std=0.0, num_steps=0, seed=0)


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
