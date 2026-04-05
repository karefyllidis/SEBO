"""
DE-GP-EI: maximize GP Expected Improvement using scipy's differential_evolution.

User-facing name: **DE-GP-EI** (differential evolution on the GP–EI acquisition).
Same interface as other wrappers: suggest(X, y, bounds) -> x_next (maximization).
Fits a GP to (X, y), then uses differential_evolution to find the point that
maximizes Expected Improvement. No extra dependencies (scipy + sklearn only).

Used in function notebooks (Section 6) and append_results/run_optimizers_on_data.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# Use project EI if available
try:
    from src.optimizers.my_bayesian.acquisition_functions import expected_improvement
except ImportError:
    from scipy.stats import norm
    def expected_improvement(mu, sigma, y_best, xi=0.01):
        mu, sigma = np.asarray(mu), np.asarray(sigma)
        with np.errstate(divide="ignore", invalid="ignore"):
            imp = mu - y_best - xi
            z = np.where(sigma > 1e-12, imp / sigma, 0.0)
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            ei[sigma <= 1e-12] = 0.0
        return ei


def load_de_gp_ei_config(
    config_path: str | Path | None = None,
    function_id: int | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load DE-GP-EI hyperparameters from configs/de_gp_ei_optimizer.yaml."""
    if not _YAML_AVAILABLE:
        return {}
    if config_path is None:
        project_root = project_root or Path(__file__).resolve().parent.parent.parent.parent
        config_path = Path(project_root) / "configs" / "de_gp_ei_optimizer.yaml"
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    merged = dict(data.get("default") or {})
    if function_id is not None and 1 <= function_id <= 8:
        merged.update((data.get(f"function_{function_id}") or {}))
    out: dict[str, Any] = {}
    for k, v in merged.items():
        if k in ("seed", "popsize", "maxiter"):
            out[k] = int(v) if v is not None else (42 if k == "seed" else None)
        elif k in ("atol", "tol", "xi"):
            out[k] = float(v) if v is not None else (0.01 if k == "xi" else 0.0 if k == "atol" else 1e-6)
        elif k == "acquisition":
            out[k] = str(v).lower() if v else "ei"
        else:
            out[k] = v
    return out


def suggest(
    X: np.ndarray,
    y: np.ndarray,
    bounds: list[tuple[float, float]] | None = None,
    *,
    config_path: str | Path | None = None,
    function_id: int | None = None,
    config: dict[str, Any] | None = None,
    acquisition: str = "ei",
    xi: float = 0.01,
    seed: int = 42,
    popsize: int = 15,
    maxiter: int = 100,
    atol: float = 0.0,
    tol: float = 1e-6,
    **kwargs: Any,
) -> np.ndarray:
    """
    Suggest next query by maximizing EI (or GP mean) with differential_evolution.

    Parameters
    ----------
    X : np.ndarray
        Shape (n, d). Observed inputs.
    y : np.ndarray
        Shape (n,). Observed outputs (maximize).
    bounds : list of (low, high) or None
        Per-dimension bounds. If None, use [0, 1]^d.
    config_path, function_id, config
        Optional config file and overrides.
    acquisition : str
        "ei" (Expected Improvement) or "mean" (GP mean).
    xi : float
        EI exploration parameter.
    seed : int
        Random seed for differential_evolution.
    popsize : int
        DE population size (e.g. 15).
    maxiter : int
        Max DE generations.
    atol, tol : float
        DE convergence tolerances.

    Returns
    -------
    x_next : np.ndarray
        Shape (d,) in bounds.
    """
    cfg = load_de_gp_ei_config(config_path=config_path, function_id=function_id)
    if isinstance(config, dict):
        cfg = {**cfg, **config}
    acquisition = str(kwargs.get("acquisition", cfg.get("acquisition", acquisition)))
    xi = float(kwargs.get("xi", cfg.get("xi", xi)))
    seed = int(kwargs.get("seed", cfg.get("seed", seed)))
    popsize = int(kwargs.get("popsize", cfg.get("popsize", popsize)))
    maxiter = int(kwargs.get("maxiter", cfg.get("maxiter", maxiter)))
    atol = float(cfg.get("atol", atol))
    tol = float(cfg.get("tol", tol))

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, d = X.shape
    assert len(y) == n
    if bounds is None:
        bounds = [(0.0, 1.0)] * d
    bounds = [(float(b[0]), float(b[1])) for b in bounds]

    # Fit GP (wider length_scale_bounds to avoid ConvergenceWarning when optimum is near default lower bound)
    rbf = RBF(length_scale=[1.0] * d, length_scale_bounds=(1e-10, 20.0))
    kernel = ConstantKernel(1.0) * rbf
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=5, random_state=seed)
    gp.fit(X, y)
    y_best = float(np.max(y))

    def objective(x: np.ndarray) -> float:
        x = np.asarray(x).reshape(1, -1)
        mu, sigma = gp.predict(x, return_std=True)
        sigma = np.maximum(sigma, 1e-10)
        if acquisition == "mean":
            return -float(mu[0])
        ei = expected_improvement(mu, sigma, y_best, xi=xi)
        return -float(ei[0])

    rng = np.random.default_rng(seed)
    result = differential_evolution(
        objective,
        bounds,
        seed=int(rng.integers(0, 2**31)),
        popsize=popsize,
        maxiter=maxiter,
        atol=atol,
        tol=tol,
        polish=True,
        disp=False,
    )
    x_next = np.asarray(result.x, dtype=np.float64).ravel()
    return np.clip(x_next, 1e-10, 1.0 - 1e-10)
