from pathlib import Path
import shutil
import yaml

from experiments.settings import ARTIFACTS_ROOT
from .model import LocalConfig
from .experiment_1 import experiment_1
from .experiment_2 import experiment_2


def main() -> None:
    data_path = ARTIFACTS_ROOT / "stn"
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open() as f:
        data = yaml.safe_load(f)
    config = LocalConfig.model_validate(data)

    experiment_1(config, data_path)
    # experiment_2(config, data_path)


if __name__ == "__main__":
    main()
