"""Download artifacts from Modal Volume to local posts/_artifacts/.

Usage:
    uv run python src/scripts/download_artifacts.py study.15
    uv run python src/scripts/download_artifacts.py study.14-ping-snn-poisson-input
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

VOLUME_NAME = "pinglab-artifacts"
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "posts" / "_artifacts"


def resolve_study(selector: str) -> str:
    """Resolve a short selector like 'study.15' to full name."""
    candidate = EXPERIMENTS_DIR / selector
    if candidate.is_dir():
        return selector
    matches = sorted(glob.glob(str(EXPERIMENTS_DIR / f"{selector}-*")))
    dirs = [m for m in matches if os.path.isdir(m)]
    if len(dirs) == 1:
        return Path(dirs[0]).name
    if len(dirs) > 1:
        print(f"Ambiguous selector '{selector}'. Matches:", file=sys.stderr)
        for d in dirs:
            print(f"  {Path(d).name}", file=sys.stderr)
        sys.exit(1)
    print(f"No study matching '{selector}'.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python src/scripts/download_artifacts.py <study-selector>")
        sys.exit(1)

    study_name = resolve_study(sys.argv[1])
    local_dest = ARTIFACTS_DIR / study_name

    if local_dest.exists():
        shutil.rmtree(local_dest)
    local_dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable, "-m", "modal", "volume", "get",
            VOLUME_NAME,
            study_name,
            tmp,
            "--force",
        ]
        print(f"Downloading {study_name} from Modal volume...")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"modal volume get failed (exit {result.returncode})", file=sys.stderr)
            sys.exit(1)

        downloaded = Path(tmp) / study_name
        if downloaded.is_dir():
            for item in downloaded.iterdir():
                shutil.move(str(item), str(local_dest / item.name))
        else:
            for item in Path(tmp).iterdir():
                shutil.move(str(item), str(local_dest / item.name))

    print(f"Artifacts downloaded to {local_dest}")


if __name__ == "__main__":
    main()
