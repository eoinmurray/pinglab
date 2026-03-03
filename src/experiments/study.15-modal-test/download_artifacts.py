"""Download artifacts from Modal Volume to local posts/_artifacts/.

Usage:
    uv run python src/experiments/study.15-modal-test/download_artifacts.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

STUDY_NAME = "study.15-modal-test"
LOCAL_DEST = Path(__file__).resolve().parent.parent.parent / "posts" / "_artifacts" / STUDY_NAME
VOLUME_NAME = "pinglab-artifacts"


def main() -> None:
    if LOCAL_DEST.exists():
        shutil.rmtree(LOCAL_DEST)
    LOCAL_DEST.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable, "-m", "modal", "volume", "get",
            VOLUME_NAME,
            STUDY_NAME,
            tmp,
            "--force",
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"modal volume get failed (exit {result.returncode})", file=sys.stderr)
            sys.exit(1)

        downloaded = Path(tmp) / STUDY_NAME
        if downloaded.is_dir():
            for item in downloaded.iterdir():
                shutil.move(str(item), str(LOCAL_DEST / item.name))
        else:
            for item in Path(tmp).iterdir():
                shutil.move(str(item), str(LOCAL_DEST / item.name))

    print(f"Artifacts downloaded to {LOCAL_DEST}")


if __name__ == "__main__":
    main()
