from typing import Callable, Iterable, List
from pinglab.types import NetworkResult
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib


def parallel(
    inner: Callable[[dict], NetworkResult],
    cfgs: Iterable[dict],
    label: str = "pl",
) -> List[NetworkResult]:
    with tqdm_joblib(desc=label, total=len(list(cfgs))) as _:
        result: List[NetworkResult] = Parallel(n_jobs=-1)(delayed(inner)(cfg) for cfg in cfgs)
    return result
