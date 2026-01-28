import sys
import shutil
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT

from lib.model import LocalConfig
from lib.experiment import run_experiment


def main() -> None:
    data_path = ARTIFACTS_ROOT / Path(__file__).parent.name
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open() as f:
        data = yaml.safe_load(f)
    config = LocalConfig.model_validate(data)

    shutil.copy2(config_path, data_path / "config.yaml")

    run_experiment(config, data_path)


if __name__ == "__main__":
    main()
