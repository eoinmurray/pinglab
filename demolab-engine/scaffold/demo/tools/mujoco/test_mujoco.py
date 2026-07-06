"""Tests for the mujoco tool's physics primitives (data-in/data-out). These call
`simulate_*` directly — the same functions the CLI handlers use — with no
Renderer, so they're headless-safe and exercise the real code path."""

import mujoco
import numpy as np

from tools.mujoco.tool import (
    CARTPOLE_XML,
    DOUBLE_PENDULUM_XML,
    simulate_cartpole,
    simulate_double_pendulum,
)


def test_cartpole_pole_falls():
    model = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
    result = simulate_cartpole(model, theta0=0.15, duration=10.0)
    assert result.fall_step is not None                 # it fell
    assert abs(result.angles[-1]) > np.pi / 3           # past the 60° threshold


def test_cartpole_is_deterministic():
    model = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
    a = simulate_cartpole(model, theta0=0.15, duration=3.0)
    b = simulate_cartpole(model, theta0=0.15, duration=3.0)
    assert np.array_equal(a.angles, b.angles)
    assert np.array_equal(a.cart_x, b.cart_x)


def test_double_pendulum_diverges_from_tiny_offset():
    model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
    result = simulate_double_pendulum(
        model,
        theta1=2.0,
        theta2=2.0,
        epsilon=1e-3,
        duration=6.0,
        separation_threshold=0.1,
    )
    assert result.sep_step is not None                  # trajectories crossed the threshold
    assert result.separations[-1] > 0.1                 # and stayed diverged
