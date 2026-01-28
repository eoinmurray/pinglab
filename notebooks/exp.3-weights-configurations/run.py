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

    configs_dir = Path(__file__).parent / "configs"
    config_paths = sorted(configs_dir.glob("config-*.yaml"))
    if not config_paths:
        raise FileNotFoundError(f"No config-*.yaml found in {configs_dir}")

    for config_path in config_paths:
        config_name = config_path.stem
        shutil.copy2(config_path, data_path / f"{config_name}.yaml")
        with config_path.open() as f:
            data = yaml.safe_load(f)
        config = LocalConfig.model_validate(data)
        run_experiment(config, data_path, config_name=config_name)


if __name__ == "__main__":
    main()
