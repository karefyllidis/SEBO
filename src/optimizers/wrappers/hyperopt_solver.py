"""
Hyperopt wrapper: suggest one next point given (X, y). Maximization.

Hyperopt uses TPE (Tree-structured Parzen Estimators), one of the original HPO
libraries. Supports distributed search via MongoDB.

Same interface as other solvers: suggest(X, y, bounds) -> x_next.

Dependency
----------
Requires Hyperopt: pip install hyperopt
Raises ImportError if Hyperopt is not installed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
    from hyperopt.base import JOB_STATE_DONE
    _HYPEROPT_AVAILABLE = True
except ImportError:
    _HYPEROPT_AVAILABLE = False


def load_hyperopt_config(
    config_path: str | Path | None = None,
    function_id: int | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load Hyperopt hyperparameters from configs/hyperopt_optimizer.yaml."""
    if not _YAML_AVAILABLE:
        return {}
    if config_path is None:
        project_root = project_root or Path(__file__).resolve().parent.parent.parent.parent
        config_path = Path(project_root) / "configs" / "hyperopt_optimizer.yaml"
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
    Suggest next query using Hyperopt TPE. Maximization.

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
    if not _HYPEROPT_AVAILABLE:
        raise ImportError("Hyperopt is required for hyperopt_solver. Install with: pip install hyperopt")

    cfg: dict[str, Any] = load_hyperopt_config(config_path=config_path, function_id=function_id)
    if isinstance(config, dict):
        cfg = {**cfg, **config}
    seed = int(cfg.get("seed", seed))

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, d = X.shape
    assert len(y) == n
    if bounds is None:
        bounds = [(0.0, 1.0)] * d

    # Scale X to [0,1]
    X_01 = np.zeros_like(X)
    for j in range(d):
        lo, hi = bounds[j][0], bounds[j][1]
        if abs(hi - lo) < 1e-12:
            X_01[:, j] = 0.5
        else:
            X_01[:, j] = (X[:, j] - lo) / (hi - lo)

    # Space: [0, 1]^d
    space = {f"x{j}": hp.uniform(f"x{j}", 0.0, 1.0) for j in range(d)}

    # Capture the suggested point from the objective
    suggested = [None]

    def objective(params: dict) -> float:
        x = np.array([float(params[f"x{j}"]) for j in range(d)], dtype=np.float64)
        suggested[0] = x
        # Return dummy loss (we only need the suggestion; Hyperopt minimizes)
        return 0.0

    rstate = np.random.default_rng(seed)
    trials = Trials()

    # Pre-populate trials with (X, y); Hyperopt minimizes so use -y
    now = time.time()
    for i in range(n):
        vals = {f"x{j}": [float(X_01[i, j])] for j in range(d)}
        spec = {f"x{j}": float(X_01[i, j]) for j in range(d)}
        result = {"status": STATUS_OK, "loss": float(-y[i])}
        misc = {"tid": i, "cmd": ("domain_attachment", "FMinIter_Domain"), "idxs": {f"x{j}": [i] for j in range(d)}, "vals": vals}
        doc = {
            "tid": i,
            "spec": spec,
            "result": result,
            "misc": misc,
            "state": JOB_STATE_DONE,
            "owner": None,
            "book_time": now,
            "refresh_time": now,
            "exp_key": None,
        }
        trials.insert_trial_doc(doc)

    # One more eval: TPE will suggest based on our trials; objective captures it
    fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=n + 1,
        trials=trials,
        rstate=rstate,
        verbose=0,
    )

    x_01 = suggested[0]
    if x_01 is None:
        # Fallback: random point
        x_01 = rstate.random(d).astype(np.float64)
    x_01 = np.clip(x_01, 0.0, 0.999999)

    # Map back to original bounds
    out = np.zeros(d, dtype=np.float64)
    for j in range(d):
        lo, hi = bounds[j][0], bounds[j][1]
        if abs(hi - lo) < 1e-12:
            out[j] = lo
        else:
            out[j] = lo + x_01[j] * (hi - lo)
    out = np.clip(out, [b[0] for b in bounds], [min(b[1], 0.999999) for b in bounds])
    try:
        from src.utils.unique_query import ensure_distinct_from_observations

        out = ensure_distinct_from_observations(out, X, bounds, min_l2=1e-3, seed=seed)
    except ImportError:
        pass
    return out
