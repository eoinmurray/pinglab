
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

def parallel(inner, cfgs, label="pl"):
    with tqdm_joblib(desc=label, total=10) as _:
        return Parallel(n_jobs=-1)(delayed(inner)(cfg) for cfg in cfgs)