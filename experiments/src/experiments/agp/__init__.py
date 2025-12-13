from pathlib import Path
import shutil
import yaml

from experiments.settings import ARTIFACTS_ROOT
from .model import LocalConfig
from .experiment_1 import experiment_1
from .experiment_2 import experiment_2
from .experiment_3 import experiment_3
from .experiment_4 import experiment_4
from .experiment_5 import experiment_5
from .experiment_6 import experiment_6


def main() -> None:
    data_path = ARTIFACTS_ROOT / "agp"
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open() as f:
        data = yaml.safe_load(f)
    config = LocalConfig.model_validate(data)

    experiment_1(config, data_path)
    experiment_2(config, data_path)
    experiment_3(config, data_path)
    experiment_4(config, data_path)
    experiment_5(config, data_path)
    experiment_6(config, data_path)


if __name__ == "__main__":
    main()
