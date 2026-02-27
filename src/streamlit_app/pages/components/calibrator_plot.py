from pathlib import Path
from typing import Any

import streamlit as st

from pinglab.backends.pytorch import simulate_network
from pinglab.io import compile_graph_to_runtime, layer_bounds_from_spec
from pinglab.plots.raster import save_raster


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _raster_output_base() -> Path:
    out_dir = _repo_root() / "src" / "streamlit_app" / ".artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "calibrator_raster"


def _run_simulation_and_render(spec: dict[str, Any]) -> str:
    runtime = compile_graph_to_runtime(spec, backend="pytorch")
    result = simulate_network(runtime)
    layer_bounds = layer_bounds_from_spec(spec)
    out_base = _raster_output_base()

    save_raster(
        result.spikes,
        out_base,
        label="Calibrator",
        xlim=(0.0, float(runtime.config.T)),
        layer_bounds=layer_bounds,
    )
    return str(out_base.with_name(out_base.name + "_light.png"))


def render_calibrator_plot() -> None:
    st.subheader("Output")

    if "calibrator_last_raster_path" not in st.session_state:
        st.session_state["calibrator_last_raster_path"] = ""
    if "calibrator_last_run_error" not in st.session_state:
        st.session_state["calibrator_last_run_error"] = ""

    if st.button("Run", use_container_width=True, type="primary"):
        spec = st.session_state.get("calibrator_config")
        if not isinstance(spec, dict):
            st.session_state["calibrator_last_run_error"] = "No config loaded."
            st.session_state["calibrator_last_raster_path"] = ""
        else:
            try:
                raster_path = _run_simulation_and_render(spec)
                st.session_state["calibrator_last_raster_path"] = raster_path
                st.session_state["calibrator_last_run_error"] = ""
            except Exception as exc:
                st.session_state["calibrator_last_run_error"] = str(exc)
                st.session_state["calibrator_last_raster_path"] = ""

    err = st.session_state.get("calibrator_last_run_error", "")
    if err:
        st.error(f"Run failed: {err}")
        return

    raster_path = st.session_state.get("calibrator_last_raster_path", "")
    if raster_path and Path(raster_path).exists():
        _, img_col, _ = st.columns([0.15, 0.7, 0.15])
        with img_col:
            st.image(raster_path, use_container_width=True)
    else:
        st.info("Press Run to compile, simulate, and render a raster.")
