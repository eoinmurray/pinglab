
import sys
from pathlib import Path
import shutil
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from local.experiment_1 import experiment_1
from local.experiment_2 import experiment_2
from local.model import LocalConfig

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
    # experiment_2(config, data_path)


if __name__ == "__main__":
    main()
    
