
from pathlib import Path
import shutil
import sys

# Add the experiment directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from pinglab.utils import load_config
from local.experiment_1 import experiment_1
from local.experiment_2 import experiment_2
from local.experiment_3 import experiment_3
from local.experiment_4 import experiment_4
from local.experiment_5 import experiment_5
from local.experiment_6 import experiment_6

def main() -> None:
    root = Path(__file__).parent
    data_path = root / "data"
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    config = load_config(root / "config.yaml")

    # Experiment 1: Vary g_ei
    experiment_1(config, data_path)

    # Experiment 2: Vary I_E
    experiment_2(config, data_path)
    
    # Experiment 3: Firing rates vs I_E (over larger range)
    experiment_3(config, data_path)

    # Experiment 4: PSD
    experiment_4(config, data_path)

    # Experiment 5: Cross corr
    experiment_5(config, data_path)

    # Experiment 6: ISI CV vs I_E
    experiment_6(config, data_path)

if __name__ == "__main__":
    main()
