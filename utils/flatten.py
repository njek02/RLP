import numpy as np
from typing import Dict


# -----------------------------
# Helper: Flatten MetaWorld observations
# -----------------------------
def flatten_obs(obs: Dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
    if isinstance(obs, dict):
        return np.concatenate([v.ravel() for v in obs.values()])
    return obs
