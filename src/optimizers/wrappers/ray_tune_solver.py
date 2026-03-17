"""
Ray Tune wrapper: suggest one next point given (X, y). Maximization.

Ray Tune is built for large-scale, distributed hyperparameter tuning. Uses
OptunaSearch under the hood for the suggest interface.

Same interface: suggest(X, y, bounds) -> x_next.

Dependency
----------
Requires Ray Tune (and Optuna): pip install "ray[tune]" optuna
Raises ImportError if Ray Tune is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    from ray.tune.search.optuna import OptunaSearch
    try:
        import optuna
        from optuna.distributions import FloatDistribution
    except ImportError:
        optuna = None
    _RAY_TUNE_AVAILABLE = True
except ImportError:
    _RAY_TUNE_AVAILABLE = False


def load_ray_tune_config(
    config_path: str | Path | None = None,
    function_id: int | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load Ray Tune hyperparameters from configs/ray_tune_optimizer.yaml."""
    if not _YAML_AVAILABLE:
        return {}
    if config_path is None:
        project_root = project_root or Path(__file__).resolve().parent.parent.parent.parent
        config_path = Path(project_root) / "configs" / "ray_tune_optimizer.yaml"
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
        if k == "seed":
            out[k] = int(v) if v is not None else 42
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
    seed: int = 42,
) -> np.ndarray:
    """
    Suggest next query using Ray Tune (OptunaSearch). Maximization.

    Parameters
    ----------
    X : np.ndarray
        Shape (n, d). Observed inputs.
    y : np.ndarray
        Shape (n,). Observed outputs (maximize).
    bounds : list of (low, high) or None
        Per-dimension bounds. If None, use [0, 1]^d.
    seed : int
        Random seed.

    Returns
    -------
    x_next : np.ndarray
        Shape (d,) in bounds (clipped to [0, 0.999999] for portal).
    """
    if not _RAY_TUNE_AVAILABLE or optuna is None:
        raise ImportError(
            "Ray Tune and Optuna are required for ray_tune_solver. "
            "Install with: pip install 'ray[tune]' optuna"
        )

    cfg: dict[str, Any] = load_ray_tune_config(config_path=config_path, function_id=function_id)
    if isinstance(config, dict):
        cfg = {**cfg, **config}
    seed = int(cfg.get("seed", seed))

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, d = X.shape
    assert len(y) == n
    if bounds is None:
        bounds = [(0.0, 1.0)] * d

    # Scale to [0, 1]
    X_01 = np.zeros_like(X)
    for j in range(d):
        lo, hi = bounds[j][0], bounds[j][1]
        if abs(hi - lo) < 1e-12:
            X_01[:, j] = 0.5
        else:
            X_01[:, j] = (X[:, j] - lo) / (hi - lo)

    space = {f"x{j}": FloatDistribution(0.0, 1.0) for j in range(d)}
    searcher = OptunaSearch(space, metric="y", mode="max", seed=seed)

    for i in range(n):
        params = {f"x{j}": float(X_01[i, j]) for j in range(d)}
        searcher.add_evaluated_point(parameters=params, value=float(y[i]))

    config = searcher.suggest("next_trial")
    if config is None:
        # Fallback: random
        rng = np.random.default_rng(seed)
        x_01 = rng.random(d).astype(np.float64)
    else:
        x_01 = np.array([float(config[f"x{j}"]) for j in range(d)], dtype=np.float64)

    x_01 = np.clip(x_01, 0.0, 0.999999)

    # Map back to original bounds
    out = np.zeros(d, dtype=np.float64)
    for j in range(d):
        lo, hi = bounds[j][0], bounds[j][1]
        if abs(hi - lo) < 1e-12:
            out[j] = lo
        else:
            out[j] = lo + x_01[j] * (hi - lo)
    return np.clip(out, [b[0] for b in bounds], [min(b[1], 0.999999) for b in bounds])
