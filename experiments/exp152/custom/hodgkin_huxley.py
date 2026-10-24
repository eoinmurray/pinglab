"""Hodgkin–Huxley membrane dynamics for the joint controller.

Adapted from snnlab/examples/hodgkin-huxley/example.py. Retains its ionic
channels, steady-state gates, RK4 and upward zero-crossing spike detection.
AMPA/GABA conductances drive reversal-potential currents at every RK4 stage;
both populations use physical milliseconds without clock scaling.
A native CPU kernel executes the same RK4 updates with steps <= 0.025 ms.
"""

import hashlib
import math
import os
import platform
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import torch
from snnlab import extensions

MAX_STEP_MS = 0.025
DEFINITION = "pinglab.exp152.hodgkin_huxley/v2"
PARAMETERS = (
    "capacitance_nf",
    "g_na_us",
    "g_k_us",
    "g_leak_us",
    "e_na_mv",
    "e_k_mv",
    "e_leak_mv",
    "excitatory_reversal_mv",
    "inhibitory_reversal_mv",
)
DEFAULT_PARAMETERS = (0.01, 1.2, 0.36, 0.003, 50.0, -77.0, -54.387, 0.0, -80.0)


def _vtrap(x):
    small = x.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1 + x / 2 + x.square() / 12, safe / -torch.expm1(-safe))


def rates(voltage):
    return (
        (_vtrap((voltage + 40) / 10), 4 * torch.exp(-(voltage + 65) / 18)),
        (0.07 * torch.exp(-(voltage + 65) / 20), torch.sigmoid((voltage + 35) / 10)),
        (0.1 * _vtrap((voltage + 55) / 10), 0.125 * torch.exp(-(voltage + 65) / 80)),
    )


def initial_state(context):
    voltage = torch.full(
        context.shape,
        getattr(context, "config", {}).get("initial_voltage_mv", -65.0),
        device=context.device,
        dtype=context.dtype,
    )
    state = {"voltage": voltage}
    for gate, (alpha, beta) in zip(("m", "h", "n"), rates(voltage)):
        state[gate] = alpha / (alpha + beta)
    return state


def _derivative(values, excitatory, inhibitory, parameters):
    voltage, m, h, n = values
    capacitance, g_na, g_k, g_leak, e_na, e_k, e_leak, e_exc, e_inh = parameters
    ionic = (
        g_na * m**3 * h * (voltage - e_na)
        + g_k * n**4 * (voltage - e_k)
        + g_leak * (voltage - e_leak)
    )
    gates = tuple(
        alpha * (1 - gate) - beta * gate
        for gate, (alpha, beta) in zip((m, h, n), rates(voltage))
    )
    current = excitatory * (e_exc - voltage) + inhibitory * (e_inh - voltage)
    return ((current - ionic) / capacitance, *gates)


def reference_step(context):
    keys = ("voltage", "m", "h", "n")
    values = tuple(context.state[key] for key in keys)
    excitatory = context.excitatory.to(values[0].dtype)
    inhibitory = context.inhibitory.to(values[0].dtype)
    parameters = tuple(
        context.config.get(key, value)
        for key, value in zip(PARAMETERS, DEFAULT_PARAMETERS)
    )
    substeps = max(1, math.ceil(context.dt_ms / MAX_STEP_MS))
    dt = context.dt_ms / substeps
    crossed = torch.zeros_like(values[0], dtype=torch.bool)
    for _ in range(substeps):
        k1 = _derivative(values, excitatory, inhibitory, parameters)
        k2 = _derivative(
            tuple(y + dt / 2 * k for y, k in zip(values, k1)),
            excitatory,
            inhibitory,
            parameters,
        )
        k3 = _derivative(
            tuple(y + dt / 2 * k for y, k in zip(values, k2)),
            excitatory,
            inhibitory,
            parameters,
        )
        k4 = _derivative(
            tuple(y + dt * k for y, k in zip(values, k3)),
            excitatory,
            inhibitory,
            parameters,
        )
        updated = tuple(
            y + dt / 6 * (a + 2 * b + 2 * c + d)
            for y, a, b, c, d in zip(values, k1, k2, k3, k4)
        )
        crossed = crossed | ((values[0] < 0) & (updated[0] >= 0))
        values = updated
    return {**context.state, **dict(zip(keys, values))}, crossed.to(values[0].dtype)


@lru_cache(maxsize=1)
def register_operation():
    """Build and load a Torch CPU operator before control starts, never in a tick."""
    source = Path(__file__).with_name("hh_kernel.c")
    bridge = source.with_name("hh_operator.cpp")
    identity = (
        source.read_bytes()
        + bridge.read_bytes()
        + (platform.platform() + torch.__version__).encode()
    )
    digest = hashlib.sha256(identity).hexdigest()[:20]
    directory = Path(tempfile.gettempdir()) / "pinglab-exp152-hh" / digest
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / "hh_kernel.so"
    if not library.exists():
        temporary = library.with_name(f"hh_kernel.{os.getpid()}.so")
        object_file = directory / f"kernel.{os.getpid()}.o"
        torch_root = Path(torch.__file__).resolve().parent
        subprocess.run(
            [
                "cc",
                "-O3",
                "-std=c99",
                "-fPIC",
                "-c",
                str(source),
                "-o",
                str(object_file),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            subprocess.run(
                [
                    "c++",
                    "-O3",
                    "-std=c++17",
                    "-fPIC",
                    "-dynamiclib" if platform.system() == "Darwin" else "-shared",
                    str(bridge),
                    str(object_file),
                    "-I" + str(torch_root / "include"),
                    "-L" + str(torch_root / "lib"),
                    "-Wl,-rpath," + str(torch_root / "lib"),
                    "-D_GLIBCXX_USE_CXX11_ABI="
                    + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
                    "-ltorch_cpu",
                    "-ltorch",
                    "-lc10",
                    "-o",
                    str(temporary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            os.replace(temporary, library)
        finally:
            object_file.unlink(missing_ok=True)
    torch.ops.load_library(str(library))

    @torch.library.register_fake("pinglab_exp152::hh_advance")
    def fake(voltage, m, h, n, excitatory, inhibitory, dt_ms, parameters):
        return voltage.new_empty((5, *voltage.shape))


def step(context):
    keys = ("voltage", "m", "h", "n")
    parameters = [
        context.config.get(key, value)
        for key, value in zip(PARAMETERS, DEFAULT_PARAMETERS)
    ]
    result = torch.ops.pinglab_exp152.hh_advance(
        *(context.state[key] for key in keys),
        context.excitatory,
        context.inhibitory,
        float(context.dt_ms),
        parameters,
    )
    voltage, m, h, n, spikes = result.unbind(0)
    return {**context.state, **dict(zip(keys, (voltage, m, h, n)))}, spikes


@lru_cache(maxsize=1)
def register():
    register_operation()
    extensions.register_neuron(
        DEFINITION,
        step,
        initialize=initial_state,
        input_unit="uS",
        state_units={"m": "1", "h": "1", "n": "1"},
        capture_safe=True,
    )
