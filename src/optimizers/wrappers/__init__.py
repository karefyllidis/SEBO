"""
Wrappers for external HPO libraries with a common suggest(X, y, bounds) interface.

- Optuna: TPE/CMA-ES, pruning, beginner-friendly
- Hyperopt: TPE, one of the originals
- TuRBO: trust-region BO for high dimensions (50+)
- DE-GP-EI: scipy differential_evolution to maximize GP-EI (module: de_gp_ei_solver)
"""

from .optuna_solver import load_optuna_config, suggest as optuna_suggest

# Optional wrappers (lazy import on first use to avoid ImportError if not installed)
def hyperopt_suggest(*args, **kwargs):
    from .hyperopt_solver import suggest
    return suggest(*args, **kwargs)

def turbo_suggest(*args, **kwargs):
    from .turbo_solver import suggest
    return suggest(*args, **kwargs)

def de_gp_ei_suggest(*args, **kwargs):
    from .de_gp_ei_solver import suggest
    return suggest(*args, **kwargs)


def ga_suggest(*args, **kwargs):
    """Deprecated alias for :func:`de_gp_ei_suggest` (legacy notebook name *GA*)."""
    return de_gp_ei_suggest(*args, **kwargs)

__all__ = [
    "optuna_suggest", "load_optuna_config",
    "hyperopt_suggest", "turbo_suggest",
    "de_gp_ei_suggest", "ga_suggest",
]
