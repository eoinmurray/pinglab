import argparse
import json
import logging
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple

import imageio.v3 as iio
import mujoco
import numpy as np

TEMP_DIR = Path(__file__).resolve().parents[2] / "temp" / "mujoco"

DOUBLE_PENDULUM_XML = """
<mujoco model="double_pendulum_pair">
  <option gravity="0 0 -9.81" timestep="0.002" integrator="RK4"/>
  <visual>
    <global offwidth="640" offheight="368"/>
  </visual>
  <worldbody>
    <light name="top" pos="0 0 3" dir="0 0 -1"/>
    <body name="anchorA" pos="-0.6 0 1.6">
      <joint name="jA1" type="hinge" axis="0 1 0" damping="0.0"/>
      <geom name="armA1" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.012" rgba="0.22 0.45 0.73 1"/>
      <body name="armA2_body" pos="0 0 -0.4">
        <joint name="jA2" type="hinge" axis="0 1 0" damping="0.0"/>
        <geom name="armA2" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.012" rgba="0.22 0.45 0.73 1"/>
        <geom name="bobA" type="sphere" pos="0 0 -0.4" size="0.03" rgba="0.22 0.45 0.73 1"/>
        <site name="tipA" pos="0 0 -0.4"/>
      </body>
    </body>
    <body name="anchorB" pos="0.6 0 1.6">
      <joint name="jB1" type="hinge" axis="0 1 0" damping="0.0"/>
      <geom name="armB1" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.012" rgba="0.80 0.31 0.31 1"/>
      <body name="armB2_body" pos="0 0 -0.4">
        <joint name="jB2" type="hinge" axis="0 1 0" damping="0.0"/>
        <geom name="armB2" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.012" rgba="0.80 0.31 0.31 1"/>
        <geom name="bobB" type="sphere" pos="0 0 -0.4" size="0.03" rgba="0.80 0.31 0.31 1"/>
        <site name="tipB" pos="0 0 -0.4"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

CARTPOLE_XML = """
<mujoco model="cartpole">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <visual>
    <global offwidth="480" offheight="320"/>
  </visual>
  <worldbody>
    <light name="top" pos="0 0 2" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 0.05" rgba="0.92 0.92 0.90 1"/>
    <body name="rail" pos="0 0 0.05">
      <geom name="rail_geom" type="capsule" fromto="-2.5 0 0 2.5 0 0" size="0.005" rgba="0.6 0.6 0.6 1"/>
    </body>
    <body name="cart" pos="0 0 0.05">
      <joint name="slider" type="slide" axis="1 0 0" damping="0.05"/>
      <geom name="cart_geom" type="box" size="0.1 0.05 0.03" rgba="0.28 0.46 0.72 1"/>
      <body name="pole" pos="0 0 0.03">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.005"/>
        <geom name="pole_geom" type="capsule" fromto="0 0 0 0 0 0.6" size="0.012" rgba="0.80 0.31 0.31 1"/>
        <geom name="bob_geom" type="sphere" pos="0 0 0.6" size="0.03" rgba="0.80 0.31 0.31 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _run_provenance() -> dict:
    """Git commit + dirty flag + UTC timestamp for this run, so every committed
    result records which code produced it. Degrades gracefully outside a git repo
    (e.g. a degit'd copy): commit is null.

    The runner-side twin lives in experiments/helpers/provenance.py; kept separate
    because the tool↔runner firewall (§4.5) forbids a tool importing experiments/."""
    here = Path(__file__).resolve().parent

    def git(*args: str) -> str:
        try:
            r = subprocess.run(["git", *args], cwd=here, capture_output=True, text=True, timeout=5)
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            return ""

    commit = git("rev-parse", "HEAD")
    return {
        "commit": commit or None,
        # dirty = uncommitted *code* (tools/runners), not regenerated outputs
        # `:/` = repo-top magic pathspec, so this is correct regardless of the cwd
        # git runs in (here: tools/mujoco). Plain "tools"/"experiments" would resolve
        # relative to cwd and silently match nothing (dirty always False).
        "dirty": bool(git("status", "--porcelain", "--", ":/tools", ":/experiments")) if commit else False,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def setup_run_dir(command: str, args: argparse.Namespace) -> tuple[Path, logging.Logger]:
    run_dir = TEMP_DIR / command
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "output.log"
    logger = logging.getLogger(f"mujoco.{command}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False

    config = {k: v for k, v in vars(args).items() if k != "func"}
    config["_provenance"] = _run_provenance()
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    cli_args = [shlex.quote(a) for a in sys.argv[1:]]
    script = Path(sys.argv[0]).resolve()
    try:
        script_rel = script.relative_to(Path.cwd())
    except ValueError:
        script_rel = script
    run_sh = "#!/bin/sh\n"
    run_sh += "# Regenerated by tools/mujoco/tool.py — reruns this command with the same arguments.\n"
    run_sh += f"exec uv run python {shlex.quote(str(script_rel))} {' '.join(cli_args)}\n"
    run_sh_path = run_dir / "run.sh"
    run_sh_path.write_text(run_sh)
    run_sh_path.chmod(0o755)

    return run_dir, logger


def write_output(run_dir: Path, metrics: dict, manifest: dict) -> None:
    missing = [k for k in manifest.get("headline_metrics", []) if k not in metrics]
    if missing:
        raise ValueError(
            f"manifest declares headline_metrics {missing!r} not present in output.json"
        )
    for field in ("headline_figure", "headline_video"):
        name = manifest.get(field)
        if name is None:
            continue
        path = run_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"manifest declares {field}={name!r} but {path} does not exist"
            )
    (run_dir / "output.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


class CartpoleResult(NamedTuple):
    angles: np.ndarray       # pole angle per step (rad)
    cart_x: np.ndarray       # cart position per step (m)
    fall_step: int | None    # first step past the 60° fall threshold, or None
    dt: float
    n_steps: int


def simulate_cartpole(
    model: mujoco.MjModel,
    theta0: float,
    duration: float,
    fps: float = 60.0,
    on_frame: Callable[[mujoco.MjData], None] | None = None,
) -> CartpoleResult:
    """Step the cartpole from a small pole offset; record the pole angle, cart
    position, and when it falls. Pure physics — pass `on_frame` to observe the
    live MjData at the frame cadence (the CLI uses it to render); tests omit it."""
    data = mujoco.MjData(model)
    data.qpos[1] = theta0  # initial pole offset, radians
    dt = model.opt.timestep
    n_steps = int(round(duration / dt))
    frame_every = max(1, int(round((1.0 / fps) / dt)))
    angles = np.empty(n_steps)
    cart_x = np.empty(n_steps)
    fall_step: int | None = None
    fall_threshold = np.pi / 3  # 60 degrees

    for step in range(n_steps):
        mujoco.mj_step(model, data)
        angles[step] = data.qpos[1]
        cart_x[step] = data.qpos[0]
        if fall_step is None and abs(data.qpos[1]) >= fall_threshold:
            fall_step = step
        if on_frame is not None and step % frame_every == 0:
            on_frame(data)

    return CartpoleResult(angles, cart_x, fall_step, dt, n_steps)


class DoublePendulumResult(NamedTuple):
    separations: np.ndarray   # tip-to-tip separation per step (m)
    sep_step: int | None      # first step past the separation threshold, or None
    dt: float
    n_steps: int


def simulate_double_pendulum(
    model: mujoco.MjModel,
    theta1: float,
    theta2: float,
    epsilon: float,
    duration: float,
    separation_threshold: float,
    fps: float = 60.0,
    on_frame: Callable[[mujoco.MjData], None] | None = None,
) -> DoublePendulumResult:
    """Release two near-identical double pendulums (B offset by `epsilon`) and
    record their tip-to-tip separation over time. Pure physics — `on_frame`
    observes the live MjData at the frame cadence for rendering."""
    data = mujoco.MjData(model)
    data.qpos[0] = theta1
    data.qpos[1] = theta2
    data.qpos[2] = theta1 + epsilon
    data.qpos[3] = theta2

    tipA_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tipA")
    tipB_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tipB")
    anchorA_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "anchorA")
    anchorB_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "anchorB")
    anchorA_xpos = model.body_pos[anchorA_id].copy()
    anchorB_xpos = model.body_pos[anchorB_id].copy()

    dt = model.opt.timestep
    n_steps = int(round(duration / dt))
    frame_every = max(1, int(round((1.0 / fps) / dt)))
    separations = np.empty(n_steps)
    sep_step: int | None = None

    for step in range(n_steps):
        mujoco.mj_step(model, data)
        tipA_local = data.site_xpos[tipA_id] - anchorA_xpos
        tipB_local = data.site_xpos[tipB_id] - anchorB_xpos
        sep = float(np.linalg.norm(tipA_local - tipB_local))
        separations[step] = sep
        if sep_step is None and sep >= separation_threshold:
            sep_step = step
        if on_frame is not None and step % frame_every == 0:
            on_frame(data)

    return DoublePendulumResult(separations, sep_step, dt, n_steps)


def cartpole(args: argparse.Namespace) -> None:
    run_dir, logger = setup_run_dir("cartpole", args)
    logger.info("cartpole start")

    model = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
    renderer = mujoco.Renderer(model, height=320, width=480)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 2.4
    cam.elevation = -12
    cam.azimuth = 90

    frames: list[np.ndarray] = []

    def on_frame(data: mujoco.MjData) -> None:
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

    result = simulate_cartpole(model, args.theta0, args.duration, args.fps, on_frame)

    video_path = run_dir / "cartpole.mp4"
    iio.imwrite(video_path, np.stack(frames), fps=args.fps, codec="libx264")
    logger.info(f"wrote {len(frames)} frames to {video_path.name}")

    fall_time_s = (result.fall_step * result.dt) if result.fall_step is not None else None
    metrics = {
        "n_steps": result.n_steps,
        "n_frames": len(frames),
        "fall_time_s": fall_time_s,
        "final_angle_deg": float(np.degrees(result.angles[-1])),
        "max_cart_displacement_m": float(np.max(np.abs(result.cart_x))),
    }
    manifest = {
        "headline_video": "cartpole.mp4",
        "headline_metrics": [
            "fall_time_s",
            "final_angle_deg",
            "max_cart_displacement_m",
        ],
    }
    write_output(run_dir, metrics, manifest)
    logger.info(f"cartpole done — fall_time_s={fall_time_s}")


def double_pendulum(args: argparse.Namespace) -> None:
    run_dir, logger = setup_run_dir("double_pendulum", args)
    logger.info("double_pendulum start")

    model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
    renderer = mujoco.Renderer(model, height=368, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat = np.array([0.0, 0.0, 1.05])
    cam.distance = 2.8
    cam.elevation = -8
    cam.azimuth = 90

    frames: list[np.ndarray] = []

    def on_frame(data: mujoco.MjData) -> None:
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

    result = simulate_double_pendulum(
        model,
        args.theta1,
        args.theta2,
        args.epsilon,
        args.duration,
        args.separation_threshold,
        args.fps,
        on_frame,
    )

    video_path = run_dir / "double_pendulum.mp4"
    iio.imwrite(video_path, np.stack(frames), fps=args.fps, codec="libx264")
    logger.info(f"wrote {len(frames)} frames to {video_path.name}")

    sep_time_s = (result.sep_step * result.dt) if result.sep_step is not None else None
    metrics = {
        "n_steps": result.n_steps,
        "n_frames": len(frames),
        "separation_time_s": sep_time_s,
        "max_separation_m": float(np.max(result.separations)),
        "final_separation_m": float(result.separations[-1]),
    }
    manifest = {
        "headline_video": "double_pendulum.mp4",
        "headline_metrics": [
            "separation_time_s",
            "max_separation_m",
            "final_separation_m",
        ],
    }
    write_output(run_dir, metrics, manifest)
    logger.info(f"double_pendulum done — separation_time_s={sep_time_s}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mujoco")
    sub = parser.add_subparsers(dest="command", required=True)

    cp = sub.add_parser("cartpole", help="Passive cart-pole released from an offset")
    cp.add_argument("--theta0", type=float, default=0.15, help="Initial pole angle (rad)")
    cp.add_argument("--duration", type=float, default=4.0, help="Simulation duration (s)")
    cp.add_argument("--fps", type=int, default=60, help="Video frame rate")
    cp.set_defaults(func=cartpole)

    dp = sub.add_parser(
        "double_pendulum",
        help="Two double pendulums side by side, identical except for a tiny initial offset",
    )
    dp.add_argument("--theta1", type=float, default=2.0, help="Initial upper-arm angle (rad)")
    dp.add_argument("--theta2", type=float, default=0.0, help="Initial lower-arm angle (rad)")
    dp.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
        help="Initial-condition perturbation applied to the second pendulum's upper angle (rad)",
    )
    dp.add_argument("--duration", type=float, default=8.0, help="Simulation duration (s)")
    dp.add_argument("--fps", type=int, default=60, help="Video frame rate")
    dp.add_argument(
        "--separation-threshold",
        dest="separation_threshold",
        type=float,
        default=0.1,
        help="Tip-to-tip distance at which to record separation time (m)",
    )
    dp.set_defaults(func=double_pendulum)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
