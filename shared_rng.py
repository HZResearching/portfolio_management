import numpy as np

_SHARED_SEED = 42

def get_rng():
    return np.random.default_rng(_SHARED_SEED)