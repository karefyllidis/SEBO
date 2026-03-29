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
Samplers: "tpe" (Tree-structured Parzen Estimator, default, multivariate mode),
"gp" (GPSampler — GP+EI, direct comparison with MyBO, requires optuna>=3.6),
"cmaes" (CMA-ES evolution strategy), or "random" (baseline).
The study direction is "maximize" to match the BBO challenge.

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


# Sentinel to distinguish "caller explicitly passed a value" from "using the function default".
# This is necessary because the YAML config also sets sampler/seed/n_startup_trials, and we
# need explicit kwargs to win over YAML while still letting YAML win over function defaults.
_UNSET: Any = object()


def suggest(
    X: np.ndarray,
    y: np.ndarray,
    bounds: list[tuple[float, float]] | None = None,
    *,
    config_path: str | Path | None = None,
    function_id: int | None = None,
    config: dict[str, Any] | None = None,
    sampler: Any = _UNSET,
    n_startup_trials: Any = _UNSET,
    seed: Any = _UNSET,
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
        "tpe" (default, multivariate) | "gp" (GP+EI, direct MyBO comparison) |
        "cmaes" | "random". Explicit value always overrides the YAML config.
    n_startup_trials : int
        Number of random trials before TPE kicks in (if sampler is "tpe").
        Explicit value always overrides the YAML config.
    seed : int
        Random seed. Explicit value always overrides the YAML config.
    **kwargs
        Forwarded to sampler construction (ignored for unknown keys).

    Returns
    -------
    x_next : np.ndarray
        Shape (d,) in bounds (clipped to [0, 0.999999] for portal).
    """
    if not _OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optuna_solver. Install with: pip install optuna")

    # Priority order (highest → lowest):
    #   1. Explicit named kwargs passed by the caller  (sampler="gp" in notebook)
    #   2. **kwargs overrides                          (unlikely but supported)
    #   3. YAML / config dict values                   (configs/optuna_optimizer.yaml)
    #   4. Built-in defaults                           ("tpe", 10, 42)
    cfg: dict[str, Any] = load_optuna_config(config_path=config_path, function_id=function_id)
    if isinstance(config, dict):
        cfg = {**cfg, **config}
    # **kwargs also override YAML (level 2)
    for k, v in kwargs.items():
        cfg[k] = v
    # Explicit named parameters override everything (level 1)
    if sampler is not _UNSET:
        cfg["sampler"] = sampler
    if n_startup_trials is not _UNSET:
        cfg["n_startup_trials"] = n_startup_trials
    if seed is not _UNSET:
        cfg["seed"] = seed
    # Apply built-in defaults for anything still missing
    cfg.setdefault("sampler", "tpe")
    cfg.setdefault("n_startup_trials", 10)
    cfg.setdefault("seed", 42)

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

    # Silence study-creation INFO messages and the multivariate experimental warning
    import warnings
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

    if sampler == "tpe":
        # multivariate=True fits a joint density over all dimensions; strictly better
        # than the default univariate mode which treats each dimension independently.
        sampler_obj = optuna.samplers.TPESampler(
            n_startup_trials=min(n_startup_trials, n),
            seed=seed,
            multivariate=True,
        )
    elif sampler == "gp":
        # GPSampler (Optuna ≥ 3.6): internal GP + EI loop — direct apples-to-apples
        # comparison with MyBO (which also uses GP + EI/PI/UCB).
        # deterministic_objective=True: assumes a noise-free objective, giving a much
        # better-conditioned GP fit for small BBO datasets. Without this flag the default
        # noisy GP tends to maximise EI at the search-space boundaries due to high
        # posterior uncertainty everywhere except near observations.
        # n_startup_trials=0: skip the random-initialisation phase so the GP is always
        # used regardless of how many prior observations exist.
        try:
            sampler_obj = optuna.samplers.GPSampler(
                seed=seed,
                deterministic_objective=True,
                n_startup_trials=0,
            )
        except AttributeError:
            raise ImportError("GPSampler requires optuna >= 3.6. Update with: pip install -U optuna")
    elif sampler == "cmaes":
        sampler_obj = optuna.samplers.CmaEsSampler(seed=seed)
    elif sampler == "random":
        sampler_obj = optuna.samplers.RandomSampler(seed=seed)
    else:
        sampler_obj = optuna.samplers.TPESampler(
            n_startup_trials=min(n_startup_trials, n), seed=seed, multivariate=True,
        )

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
