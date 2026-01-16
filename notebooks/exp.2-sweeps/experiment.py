import sys
from pathlib import Path
import shutil
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT

from lib.model import LocalConfig
from lib.experiment_1 import experiment_1
from lib.experiment_2 import experiment_2
from lib.experiment_3 import experiment_3
from lib.experiment_4 import experiment_4
from lib.experiment_5 import experiment_5
from lib.experiment_6 import experiment_6


def main() -> None:
    """
    Docstring for this experiment
    """
    data_path = ARTIFACTS_ROOT / Path(__file__).parent.name
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
