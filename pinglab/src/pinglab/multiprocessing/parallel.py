from typing import Callable, TypeVar
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

T = TypeVar("T")


def parallel(
    inner: Callable[[dict], T],
    cfgs: list[dict],
    label: str = "pl",
) -> list[T]:
    """
    Execute a function in parallel over a list of configurations.

    Parameters:
        inner: Function to execute for each configuration
        cfgs: List of configuration dictionaries
        label: Progress bar label

    Returns:
        List of results from each function call
    """
    with tqdm_joblib(desc=label, total=len(cfgs)) as _:
        result: list[T] = Parallel(n_jobs=-1)(delayed(inner)(cfg) for cfg in cfgs)
    return result
