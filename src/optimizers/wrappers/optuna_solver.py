"""
Optuna wrapper: suggest one next point given (X, y). Maximization.

This module provides the same interface as the project's classical BO solver
(suggest(X, y, bounds) -> x_next) so you can compare Optuna's TPE (and other
samplers) against MyBO on the same challenge data. It does not evaluate the
objective—it only returns the next point Optuna would suggest given the
observed (X, y).

How it works
------------
Existing observations are added to an Optuna study as completed trials
(FrozenTrial). The study then asks for one new trial; we read the suggested
continuous parameters (suggest_float) and return them as a single next point.
Samplers: "tpe" (Tree-structured Parzen Estimator, default), "cmaes", or
"random". The study direction is "maximize" to match the BBO challenge.

Dependency
----------
Requires Optuna: pip install optuna (or pip install -r requirements-benchmark.txt).
Raises ImportError if Optuna is not installed.

Usage
-----
From project root:
    from src.optimizers.wrappers.optuna_solver import suggest as optuna_suggest
    x_next = optuna_suggest(X, y, bounds=[(0, 1)] * d, seed=42)
    # Per-function hyperparameters from configs/optuna_optimizer.yaml:
    x_next = optuna_suggest(X, y, bounds=bounds, function_id=1)

Used by scripts/run_optimizers_on_data.py and by the Function 1 notebook
(Section 6: MyBO vs Optuna comparison).
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    import optuna
    from optuna.trial import FrozenTrial, TrialState
    from optuna.distributions import FloatDistribution
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False


def load_optuna_config(
    config_path: str | Path | None = None,
    function_id: int | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load Optuna hyperparameters from a YAML file. Per-function overrides supported.

    Parameters
    ----------
    config_path : str or Path or None
        Path to YAML file. If None, uses project_root / "configs" / "optuna_optimizer.yaml".
    function_id : int or None
        Challenge function 1..8. If set, merges default with function_{id} section.
    project_root : str or Path or None
        Project root. If None, inferred from this file's location (two levels up from src/optimizers/wrappers/).

    Returns
    -------
    dict
        Merged config (default + function section). Keys: sampler, n_startup_trials, seed, n_ei_candidates.
    """
    if not _YAML_AVAILABLE:
        return {}
    if config_path is None:
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent.parent.parent
        config_path = Path(project_root) / "configs" / "optuna_optimizer.yaml"
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
        if k == "sampler":
            out[k] = str(v) if v is not None else "tpe"
        elif k in ("n_startup_trials", "n_ei_candidates", "seed"):
            out[k] = int(v) if v is not None else (42 if k == "seed" else (10 if k == "n_startup_trials" else 24))
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
    sampler: str = "tpe",
    n_startup_trials: int = 10,
    seed: int = 42,
    **kwargs: Any,
) -> np.ndarray:
    """
    Suggest next query using Optuna. Maximization.

    Parameters
    ----------
    X : np.ndarray
        Shape (n, d). Observed inputs.
    y : np.ndarray
        Shape (n,). Observed outputs (maximize).
    bounds : list of (low, high) or None
        Per-dimension bounds. If None, use [0, 1]^d.
    config_path : str or Path or None
        Path to configs/optuna_optimizer.yaml. If set with function_id, config is loaded and merged.
    function_id : int or None
        Challenge function 1..8. Used to select the function_1..function_8 section in the YAML.
    config : dict or None
        Pre-loaded hyperparameter dict (e.g. from load_optuna_config()). Overridden by explicit kwargs.
    sampler : str
        "tpe" (default) or "cmaes" or "random". Overridden by config when function_id/config_path set.
    n_startup_trials : int
        Number of random trials before TPE (if sampler is tpe).
    seed : int
        Random seed.
    **kwargs
        Any of sampler, n_startup_trials, seed; override config and the above defaults.

    Returns
    -------
    x_next : np.ndarray
        Shape (d,) in bounds (clipped to [0, 0.999999] for portal).
    """
    if not _OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optuna_solver. Install with: pip install optuna")

    # Config file + optional config dict; then explicit sampler/n_startup_trials/seed (and kwargs) override
    cfg: dict[str, Any] = load_optuna_config(config_path=config_path, function_id=function_id)
    if isinstance(config, dict):
        cfg = {**cfg, **config}
    # Signature defaults override config only when caller passes them (we can't detect that, so we apply defaults)
    # so that suggest(X, y) works without a config file. When config_path/function_id is used, config wins unless
    # the caller passes e.g. seed=0.
    cfg.setdefault("sampler", sampler)
    cfg.setdefault("n_startup_trials", n_startup_trials)
    cfg.setdefault("seed", seed)
    cfg["sampler"] = kwargs.get("sampler", cfg["sampler"])
    cfg["n_startup_trials"] = kwargs.get("n_startup_trials", cfg["n_startup_trials"])
    cfg["seed"] = kwargs.get("seed", cfg["seed"])
    for k, v in kwargs.items():
        if k not in ("sampler", "n_startup_trials", "seed"):
            cfg[k] = v
    sampler = str(cfg.get("sampler", "tpe"))
    n_startup_trials = int(cfg.get("n_startup_trials", 10))
    seed = int(cfg.get("seed", 42))

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, d = X.shape
    assert len(y) == n
    if bounds is None:
        bounds = [(0.0, 1.0)] * d

    # Scale X to [0,1] for Optuna if bounds differ
    X_01 = np.zeros_like(X)
    for j in range(d):
        lo, hi = bounds[j][0], bounds[j][1]
        if abs(hi - lo) < 1e-12:
            X_01[:, j] = 0.5
        else:
            X_01[:, j] = (X[:, j] - lo) / (hi - lo)

    if sampler == "tpe":
        sampler_obj = optuna.samplers.TPESampler(
            n_startup_trials=min(n_startup_trials, n),
            seed=seed,
        )
    elif sampler == "cmaes":
        sampler_obj = optuna.samplers.CmaEsSampler(seed=seed)
    elif sampler == "random":
        sampler_obj = optuna.samplers.RandomSampler(seed=seed)
    else:
        sampler_obj = optuna.samplers.TPESampler(n_startup_trials=min(n_startup_trials, n), seed=seed)

    study = optuna.create_study(direction="maximize", sampler=sampler_obj)

    dists = {f"x{j}": FloatDistribution(0.0, 1.0) for j in range(d)}
    now = datetime.datetime.now()
    for i in range(n):
        params = {f"x{j}": float(X_01[i, j]) for j in range(d)}
        trial = FrozenTrial(
            number=i,
            state=TrialState.COMPLETE,
            value=float(y[i]),
            datetime_start=now,
            datetime_complete=now,
            params=params,
            distributions=dists,
            user_attrs={},
            system_attrs={},
            intermediate_values={},
            trial_id=i,
        )
        study.add_trial(trial)

    # Ask for one new trial and read suggested params (we do not evaluate it)
    trial = study.ask()
    x_01 = np.array([trial.suggest_float(f"x{j}", 0.0, 1.0) for j in range(d)], dtype=np.float64)
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
