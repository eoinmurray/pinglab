import json
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from pinglab.io import compile_graph
except Exception:  # pragma: no cover
    compile_graph = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _experiments_root() -> Path:
    return _repo_root() / "src" / "experiments"


def _discover_experiment_configs() -> list[Path]:
    root = _experiments_root()
    if not root.exists():
        return []
    return sorted(p for p in root.glob("*/config.json") if p.is_file())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _get_node_size(spec: dict[str, Any], node_id: str, default: int) -> int:
    nodes = spec.get("nodes", [])
    if not isinstance(nodes, list):
        return default
    for node in nodes:
        if isinstance(node, dict) and str(node.get("id", "")) == node_id:
            return int(node.get("size", default))
    return default


def _set_node_size(spec: dict[str, Any], node_id: str, new_size: int) -> None:
    nodes = spec.setdefault("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("nodes must be a list")
    for node in nodes:
        if isinstance(node, dict) and str(node.get("id", "")) == node_id:
            node["size"] = int(new_size)
            return


def _first_input_key(spec: dict[str, Any]) -> str | None:
    inputs = spec.get("inputs", {})
    if not isinstance(inputs, dict) or not inputs:
        return None
    return sorted(str(k) for k in inputs.keys())[0]


def _init_state() -> None:
    st.session_state.setdefault("calibrator_config", None)
    st.session_state.setdefault("calibrator_config_path", "")
    st.session_state.setdefault("calibrator_validation_msg", "")
    st.session_state.setdefault("calibrator_validation_ok", None)


def _load_config(path: Path) -> None:
    spec = _read_json(path)
    st.session_state["calibrator_config"] = spec
    st.session_state["calibrator_config_path"] = str(path)
    st.session_state["calibrator_validation_msg"] = ""
    st.session_state["calibrator_validation_ok"] = None


def _validate_config(spec: dict[str, Any]) -> tuple[bool, str]:
    if compile_graph is None:
        return False, "Validation unavailable: pinglab compiler import failed."
    try:
        compile_graph(spec)
    except Exception as exc:
        return False, str(exc)
    return True, "Config is valid."


def render_calibrator_form() -> None:
    _init_state()
    st.subheader("Config Form")

    config_paths = _discover_experiment_configs()
    if not config_paths:
        st.error("No experiment configs found under src/experiments/*/config.json")
        return

    options = [str(p.relative_to(_repo_root())) for p in config_paths]
    selected_rel = st.selectbox("Experiment config", options, key="calibrator_selected_config")
    selected_abs = _repo_root() / selected_rel

    load_col, validate_col, save_col = st.columns(3)
    with load_col:
        if st.button("Load", use_container_width=True):
            _load_config(selected_abs)
    with validate_col:
        if st.button("Validate", use_container_width=True):
            spec = st.session_state.get("calibrator_config")
            if spec is None:
                st.session_state["calibrator_validation_ok"] = False
                st.session_state["calibrator_validation_msg"] = "Load a config first."
            else:
                ok, msg = _validate_config(spec)
                st.session_state["calibrator_validation_ok"] = ok
                st.session_state["calibrator_validation_msg"] = msg
    with save_col:
        if st.button("Save", use_container_width=True):
            spec = st.session_state.get("calibrator_config")
            if spec is None:
                st.session_state["calibrator_validation_ok"] = False
                st.session_state["calibrator_validation_msg"] = "Load a config first."
            else:
                ok, msg = _validate_config(spec)
                if ok:
                    _write_json(selected_abs, spec)
                    st.session_state["calibrator_validation_ok"] = True
                    st.session_state["calibrator_validation_msg"] = f"Saved {selected_rel}"
                else:
                    st.session_state["calibrator_validation_ok"] = False
                    st.session_state["calibrator_validation_msg"] = f"Save blocked: {msg}"

    if st.session_state.get("calibrator_config") is None:
        _load_config(selected_abs)

    spec = st.session_state["calibrator_config"]
    if spec is None:
        st.warning("No config loaded.")
        return

    validation_ok = st.session_state.get("calibrator_validation_ok")
    validation_msg = st.session_state.get("calibrator_validation_msg", "")
    if validation_msg:
        if validation_ok:
            st.success(validation_msg)
        else:
            st.error(validation_msg)

    st.caption(f"Editing: `{selected_rel}`")

    sim = spec.setdefault("sim", {})
    execution = spec.setdefault("execution", {})
    inputs = spec.setdefault("inputs", {})

    with st.form("calibrator_graph_form"):
        st.markdown("**Simulation**")
        sim_dt = st.number_input("dt_ms", value=float(sim.get("dt_ms", 0.1)), step=0.01)
        sim_t = st.number_input("T_ms", value=float(sim.get("T_ms", 1000.0)), step=10.0)
        sim_seed = st.number_input("seed", value=int(sim.get("seed", 0)), step=1)
        sim_model = st.selectbox(
            "neuron_model",
            options=["lif", "mqif"],
            index=0 if str(sim.get("neuron_model", "lif")) == "lif" else 1,
        )

        st.markdown("**Execution**")
        perf_mode = st.checkbox(
            "performance_mode",
            value=bool(execution.get("performance_mode", False)),
        )
        max_spikes = st.number_input(
            "max_spikes",
            value=int(execution.get("max_spikes", 30000)),
            step=1000,
            min_value=1000,
        )
        burn_in_ms = st.number_input(
            "burn_in_ms",
            value=float(execution.get("burn_in_ms", 0.0)),
            step=10.0,
        )

        st.markdown("**Population Sizes**")
        n_e = st.number_input(
            "E.size",
            value=int(_get_node_size(spec, "E", 0)),
            min_value=0,
            step=1,
        )
        n_i = st.number_input(
            "I.size",
            value=int(_get_node_size(spec, "I", 0)),
            min_value=0,
            step=1,
        )

        st.markdown("**First Input Program**")
        first_input = _first_input_key(spec)
        if first_input is None:
            st.info("No inputs found in this graph.")
            input_mode = "tonic"
            input_mean = 0.0
            input_std = 0.0
        else:
            input_cfg = inputs.setdefault(first_input, {})
            input_mode = st.selectbox(
                f"{first_input}.mode",
                options=["tonic", "ramp", "sine", "external_spike_train", "pulse", "pulses"],
                index=max(
                    0,
                    ["tonic", "ramp", "sine", "external_spike_train", "pulse", "pulses"].index(
                        str(input_cfg.get("mode", "tonic"))
                    )
                    if str(input_cfg.get("mode", "tonic"))
                    in ["tonic", "ramp", "sine", "external_spike_train", "pulse", "pulses"]
                    else 0,
                ),
            )
            input_mean = st.number_input(
                f"{first_input}.mean",
                value=float(input_cfg.get("mean", 0.0)),
                step=0.1,
            )
            input_std = st.number_input(
                f"{first_input}.std",
                value=float(input_cfg.get("std", 0.0)),
                step=0.1,
                min_value=0.0,
            )

        apply_changes = st.form_submit_button("Apply changes")

        if apply_changes:
            sim["dt_ms"] = float(sim_dt)
            sim["T_ms"] = float(sim_t)
            sim["seed"] = int(sim_seed)
            sim["neuron_model"] = str(sim_model)

            execution["performance_mode"] = bool(perf_mode)
            execution["max_spikes"] = int(max_spikes)
            execution["burn_in_ms"] = float(burn_in_ms)

            _set_node_size(spec, "E", int(n_e))
            _set_node_size(spec, "I", int(n_i))

            if first_input is not None:
                inputs[first_input]["mode"] = str(input_mode)
                inputs[first_input]["mean"] = float(input_mean)
                inputs[first_input]["std"] = float(input_std)

            st.session_state["calibrator_config"] = spec
            st.session_state["calibrator_validation_ok"] = None
            st.session_state["calibrator_validation_msg"] = "Applied in-memory changes."

    with st.expander("Raw JSON (read-only preview)", expanded=False):
        st.code(json.dumps(spec, indent=2), language="json")
