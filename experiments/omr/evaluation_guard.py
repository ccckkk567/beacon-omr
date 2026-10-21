"""Keep Demo10 root-only evaluation out of training RNG and archive evidence."""
from functools import wraps


def isolated_evaluation(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        if not self.config.omr.get('demo'):
            return function(self, *args, **kwargs)
        import random
        import numpy as np
        import torch
        from .task_manifest import digest
        observer = self.train_observer
        before = digest(observer.state_dict()) if hasattr(observer, 'state_dict') else None
        rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            random.seed(20261024); np.random.seed(20261024); torch.manual_seed(20261024)
            return function(self, *args, **kwargs)
        finally:
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
            if before is not None and digest(observer.state_dict()) != before:
                raise ValueError('Root evaluation mutated training archive/statistics/planner')
    return wrapped
