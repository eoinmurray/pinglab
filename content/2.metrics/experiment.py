import numpy as np
from pathlib import Path
import shutil
import sys
import yaml

# Add the experiment directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from local.model import LocalConfig
from local.run_regime import run_regime


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

    run_regime(
        config=config,
        data_path=data_path,
        label="ai"
    )

    new_config = config.model_copy(update={
        "default_inputs": config.default_inputs.model_copy(update={"I_E": 2.0}),
        "base": config.base.model_copy(update={"g_ei": 3.0}),
    })

    run_regime(
        config=new_config,
        data_path=data_path,
        label="osc"
    )
    
if __name__ == "__main__":
    main()
