import json
import shutil
import subprocess
import sys
from pathlib import Path

from helpers import provenance  # run.sh reproducer into the committed record (RULES §4.7)

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / "temp"
ARTIFACTS = ROOT / "artifacts" / "data" / "exp002"

# Each entry: (tool dir under tools/, subcommand).
COMMANDS = (
    ("mujoco", "cartpole"),
)


def run_tool(tool: str, command: str) -> None:
    tool_path = ROOT / "tools" / tool / "tool.py"
    # sys.executable is the uv-managed venv python (`task run` wraps us in `uv run`). Not the
    # `sh` package — it doesn't support Windows. Same idiom as playground.py.
    subprocess.run([sys.executable, str(tool_path), command], check=True)


def artifact_dir(tool: str, command: str) -> Path:
    return TEMP / tool / command


def load_manifest(tool: str, command: str) -> dict:
    return json.loads((artifact_dir(tool, command) / "manifest.json").read_text())


def collect_numbers() -> dict:
    numbers: dict = {}
    for tool, command in COMMANDS:
        manifest = load_manifest(tool, command)
        config = json.loads((artifact_dir(tool, command) / "config.json").read_text())
        output = json.loads((artifact_dir(tool, command) / "output.json").read_text())
        numbers[command] = {
            "config": config,
            **{f: output[f] for f in manifest.get("headline_metrics", [])},
        }
    return numbers


def copy_headline_assets() -> None:
    for tool, command in COMMANDS:
        manifest = load_manifest(tool, command)
        src_dir = artifact_dir(tool, command)
        for field in ("headline_figure", "headline_video"):
            name = manifest.get(field)
            if not name:
                continue
            src = src_dir / name
            dst = ARTIFACTS / name
            shutil.copy(src, dst)
            print(f"copied {name} -> {dst}")


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for tool, command in COMMANDS:
        run_tool(tool, command)
    copy_headline_assets()
    numbers_path = ARTIFACTS / "numbers.json"
    numbers_path.write_text(json.dumps(collect_numbers(), indent=2) + "\n")
    print(f"wrote {numbers_path}")

    provenance.write_run_sh(ARTIFACTS)


if __name__ == "__main__":
    main()
