from pathlib import Path
import shutil
import yaml

from experiments.settings import ARTIFACTS_ROOT
from .model import LocalConfig
from .run_regime import run_regime


def main() -> None:
    data_path = ARTIFACTS_ROOT / "dub"
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(__file__).parent / "config.yaml"
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
