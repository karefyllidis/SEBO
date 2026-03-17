"""
HEBO wrapper: suggest one next point given (X, y). Maximization.

HEBO (Heteroscedastic Evolutionary Bayesian Optimization) won the NeurIPS 2020
Black-Box Optimization challenge. Top performer on benchmarks, especially for
noisy real-world objectives.

Same interface as other solvers: suggest(X, y, bounds) -> x_next.

Dependency
----------
Requires HEBO: pip install HEBO
Raises ImportError if HEBO is not installed.
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
    import pandas as pd
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO
    _HEBO_AVAILABLE = True
except ImportError:
    _HEBO_AVAILABLE = False


def load_hebo_config(
    config_path: str | Path | None = None,
    function_id: int | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load HEBO hyperparameters from configs/hebo_optimizer.yaml."""
    if not _YAML_AVAILABLE:
        return {}
    if config_path is None:
        project_root = project_root or Path(__file__).resolve().parent.parent.parent.parent
        config_path = Path(project_root) / "configs" / "hebo_optimizer.yaml"
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
        if k == "model_name":
            out[k] = str(v) if v else "gpy"
        elif k in ("rand_sample", "seed"):
            out[k] = int(v) if v is not None else (4 if k == "rand_sample" else 42)
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
    rand_sample: int | None = None,
    model_name: str | None = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Suggest next query using HEBO. Maximization.

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
    if not _HEBO_AVAILABLE:
        raise ImportError("HEBO is required for hebo_solver. Install with: pip install HEBO")

    cfg: dict[str, Any] = load_hebo_config(config_path=config_path, function_id=function_id)
    if isinstance(config, dict):
        cfg = {**cfg, **config}
    cfg.setdefault("rand_sample", rand_sample if rand_sample is not None else 4)
    cfg.setdefault("model_name", model_name or "gpy")
    cfg.setdefault("seed", seed)
    rand_sample = int(cfg.get("rand_sample", 4))
    model_name = str(cfg.get("model_name", "gpy"))
    seed = int(cfg.get("seed", 42))

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, d = X.shape
    assert len(y) == n
    if bounds is None:
        bounds = [(0.0, 1.0)] * d

    # Scale X to [0,1] for HEBO design space
    X_01 = np.zeros_like(X)
    for j in range(d):
        lo, hi = bounds[j][0], bounds[j][1]
        if abs(hi - lo) < 1e-12:
            X_01[:, j] = 0.5
        else:
            X_01[:, j] = (X[:, j] - lo) / (hi - lo)

    # Design space: [0, 1]^d
    space = DesignSpace().parse([
        {"name": f"x{j}", "type": "num", "lb": 0.0, "ub": 1.0}
        for j in range(d)
    ])

    np.random.seed(seed)
    rs = min(rand_sample, max(1, n))
    try:
        opt = HEBO(space, rand_sample=rs, model_name=model_name, seed=seed)
    except TypeError:
        try:
            opt = HEBO(space, rand_sample=rs, model_name=model_name)
        except TypeError:
            opt = HEBO(space, rand_sample=rs)

    # Warm start with observations (HEBO minimizes by default; negate y for maximization)
    # HEBO's observe: rec_x is DataFrame, obj is (n,) array. For minimization we pass -y.
    df_cols = {f"x{j}": X_01[:, j] for j in range(d)}
    rec_x = pd.DataFrame(df_cols)
    opt.observe(rec_x, -np.asarray(y).reshape(-1, 1))  # minimize -y = maximize y

    rec = opt.suggest(n_suggestions=1)
    x_01 = np.array([float(rec[f"x{j}"].iloc[0]) for j in range(d)], dtype=np.float64)
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
