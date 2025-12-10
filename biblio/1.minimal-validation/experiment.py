
from pathlib import Path
import shutil
import sys

import yaml

# Add the experiment directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from local.model import LocalConfig
from local.experiment_1 import experiment_1
from local.experiment_2 import experiment_2
from local.experiment_3 import experiment_3
from local.experiment_4 import experiment_4
from local.experiment_5 import experiment_5

def main() -> None:
    root = Path(__file__).parent
    data_path = root / "data"
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.yaml"
    with config_path.open() as f:
        data = yaml.safe_load(f)
    config = LocalConfig.model_validate(data)

    experiment_1(config, data_path)
    experiment_2(config, data_path)
    experiment_3(config, data_path)
    experiment_4(config, data_path)
    experiment_5(config, data_path)

if __name__ == "__main__":
    main()
