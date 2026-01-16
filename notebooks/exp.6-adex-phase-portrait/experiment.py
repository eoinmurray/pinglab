import sys
from pathlib import Path
import shutil
import yaml
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT

from lib.model import LocalConfig
from pinglab.plots.styles import save_both, figsize


def main() -> None:
    """
    1) Pick the model + parameters

    - Use the AdEx equations (V, w) and a constant external current I_ext.
    - Choose parameter values (C_m, g_L, E_L, V_T, Delta_T, tau_w, a, b, V_reset,
    V_peak).
    - For a phase portrait, you only need the subthreshold dynamics, so b,
    V_reset, V_peak are irrelevant unless you want to show reset behavior.

    2) Choose the phase‑plane domain

    - Decide a voltage range (e.g. −80 to +20 mV).
    - Decide a w range wide enough to show the nullclines’ intersection and flow
    (e.g. −200 to +400 in current units).
    - Pick grid resolution (e.g. 150–250 points per axis).

    3) Build the vector field

    - For every grid point (V, w), compute:
        - dV/dt = (g_L (E_L − V) + g_L Δ_T exp((V − V_T)/Δ_T) − w + I_ext) / C_m
        - dw/dt = (a (V − E_L) − w) / τ_w
    - This gives a vector (dV/dt, dw/dt) at each point.

    4) Compute nullclines

    - V‑nullcline (dV/dt = 0):
    w = g_L (E_L − V) + g_L Δ_T exp((V − V_T)/Δ_T) + I_ext
    - w‑nullcline (dw/dt = 0):
    w = a (V − E_L)
    - Plot both on the same axes.

    5) Plot the phase portrait

    - Draw a streamplot or quiver plot of the vector field.
    - Overlay both nullclines with distinct colors.
    - Optionally annotate the fixed point at the nullcline intersection.

    6) Interpret

    - The fixed point is where the nullclines cross.
    - The shape of the V‑nullcline (from the exponential term) shows the sharp
    onset regime.
    - If you later vary I_ext, you can show how the fixed point moves and whether
    it crosses into spiking.
    """
    data_path = ARTIFACTS_ROOT / Path(__file__).parent.name
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open() as f:
        data = yaml.safe_load(f)
    config = LocalConfig.model_validate(data)

    C_m = config.adex.C_m
    g_L = config.adex.g_L
    E_L = config.adex.E_L
    Delta_T = config.adex.Delta_T
    V_T = config.adex.V_T
    a = config.adex.a
    b = config.adex.b

    V_min = config.phase_portrait.V_min
    V_max = config.phase_portrait.V_max
    w_min = config.phase_portrait.w_min
    w_max = config.phase_portrait.w_max
    V_points = config.phase_portrait.V_points
    w_points = config.phase_portrait.w_points
    I_ext = config.phase_portrait.I_ext

    def f(V, w):
        exp_arg = np.clip((V - V_T) / Delta_T, -100.0, 50.0)
        dVdt = (g_L * (E_L - V) + g_L * Delta_T * np.exp(exp_arg) - w + I_ext) / C_m
        return dVdt

    def g(V, w):
        dwdt = (a * (V - E_L) - w) / config.adex.tau_w
        return dwdt

    def fg(V, w):
        return f(V, w), g(V, w)

    def V_nullcline(V):
        return g_L * (E_L - V) + g_L * Delta_T * np.exp((V - V_T) / Delta_T) + I_ext

    def w_nullcline(V):
        return a * (V - E_L)

    V_grid, w_grid = np.meshgrid(
        np.linspace(V_min, V_max, V_points),
        np.linspace(w_min, w_max, w_points),
    )

    dVdt_grid, dwdt_grid = fg(V_grid, w_grid)

    field_V_nullcline = V_nullcline(V_grid)
    field_w_nullcline = w_nullcline(V_grid)

    V_axis = np.linspace(V_min, V_max, V_points)
    w_axis = np.linspace(w_min, w_max, w_points)
    V_nullcline_line = V_nullcline(V_axis)
    w_nullcline_line = w_nullcline(V_axis)

    def plot_phase_portrait() -> None:
        plt.figure(figsize=figsize)
        plt.streamplot(
            V_axis,
            w_axis,
            dVdt_grid,
            dwdt_grid,
            density=1.2,
            color="#666666",
            linewidth=0.8,
            arrowsize=1.0,
        )
        step = max(1, V_points // 25)
        plt.quiver(
            V_grid[::step, ::step],
            w_grid[::step, ::step],
            dVdt_grid[::step, ::step],
            dwdt_grid[::step, ::step],
            color="#999999",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.002,
            alpha=0.7,
        )
        plt.plot(V_axis, V_nullcline_line, color="#1f77b4", label="dV/dt = 0")
        plt.plot(V_axis, w_nullcline_line, color="#d62728", label="dw/dt = 0")
        plt.xlim(V_min, V_max)
        plt.ylim(w_min, w_max)
        plt.xlabel("Membrane potential V (mV)")
        plt.ylabel("Adaptation current w")
        plt.title("AdEx Phase Portrait")
        plt.legend(frameon=False)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    save_both(data_path / "adex_phase_portrait.png", plot_phase_portrait)

if __name__ == "__main__":
    main()
