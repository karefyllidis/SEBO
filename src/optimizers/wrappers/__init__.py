"""
Wrappers for external HPO libraries with a common suggest(X, y, bounds) interface.

- Optuna: TPE/CMA-ES, pruning, beginner-friendly
- Ray Tune: large-scale, distributed; uses OptunaSearch
- HEBO: NeurIPS 2020 BBO winner, strong on noisy objectives
- Hyperopt: TPE, one of the originals
- TuRBO: trust-region BO for high dimensions (50+)
- GA: evolutionary (differential_evolution) to maximize GP-EI; scipy only
"""

from .optuna_solver import load_optuna_config, suggest as optuna_suggest

# Optional wrappers (lazy import on first use to avoid ImportError if not installed)
def hebo_suggest(*args, **kwargs):
    from .hebo_solver import suggest
    return suggest(*args, **kwargs)

def hyperopt_suggest(*args, **kwargs):
    from .hyperopt_solver import suggest
    return suggest(*args, **kwargs)

def turbo_suggest(*args, **kwargs):
    from .turbo_solver import suggest
    return suggest(*args, **kwargs)

def ray_tune_suggest(*args, **kwargs):
    from .ray_tune_solver import suggest
    return suggest(*args, **kwargs)

def ga_suggest(*args, **kwargs):
    from .ga_solver import suggest
    return suggest(*args, **kwargs)

__all__ = [
    "optuna_suggest", "load_optuna_config",
    "hebo_suggest", "hyperopt_suggest", "turbo_suggest", "ray_tune_suggest",
    "ga_suggest",
]
