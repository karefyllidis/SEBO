"""
MyBO: classical Bayesian optimization for the BBO challenge.

Uses a single GP surrogate (RBF / Matérn / RBF+WhiteKernel) with EI/PI/UCB
acquisition—the standard sample-efficient loop: fit surrogate → maximise
acquisition → suggest next point. No trust regions, heteroscedastic models, or
ensemble-of-optimisers; this is the classical BO approach (e.g. Jones et al.,
Rasmussen & Williams) as used in the notebooks.

Single entry point: suggest(X, y, bounds, **kwargs) -> x_next.
Replicates the notebook logic: kernel selection by LML, candidate masking,
ensemble or solo acquisition, duplicate fallback.

Hyperparameters from YAML
-------------------------
Pass config_path and/or function_id (1..8) to load per-function settings via
load_mybo_config() from configs/bayesian_optimizer.yaml when that file exists;
if it is missing, the loader returns {} and suggest() uses code defaults. When
the file exists, merge default with function_N sections; explicit kwargs to
suggest() override the merged config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb
from skopt.sampler import Sobol, Lhs

# Optional: use project warping if available (run from project root)
try:
    from src.utils.warping import apply_output_warping
except ImportError:
    apply_output_warping = None

# Optional: YAML for config
try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

# Keys that suggest() accepts from config (all hyperparameters except X, y, bounds)
_SUGGEST_CONFIG_KEYS = frozenset({
    "output_warping", "kernel", "optimize_kernel", "n_restarts", "length_scale_bounds",
    "xi", "kappa", "solo_strategy", "use_ensemble_aquisation", "agree_threshold",
    "n_cand", "candidate_method", "min_dist_threshold", "boundary_margin",
    "duplicate_threshold", "seed",
})


def load_mybo_config(
    config_path: str | Path | None = None,
    function_id: int | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load MyBO hyperparameters from a YAML file. Per-function overrides supported.

    Parameters
    ----------
    config_path : str or Path or None
        Path to YAML file. If None, uses project_root / "configs" / "bayesian_optimizer.yaml".
    function_id : int or None
        Challenge function 1..8. If set, merges default with function_{id} section.
    project_root : str or Path or None
        Project root. If None, inferred from this file's location (repo root: four parents up from this file).

    Returns
    -------
    dict
        All keys from merged config (default + function section). Used by suggest() and by
        notebooks for GP kernel params (constant_kernel_scale, length_scale, gp_alpha, etc.).
        Lists for *_bounds are converted to tuples; numeric types coerced.
    """
    if not _YAML_AVAILABLE:
        return {}
    if config_path is None:
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent.parent.parent
        config_path = Path(project_root) / "configs" / "bayesian_optimizer.yaml"
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    # Merge default + function section
    merged = dict(data.get("default") or {})
    if function_id is not None and 1 <= function_id <= 8:
        key = f"function_{function_id}"
        merged.update((data.get(key) or {}))
    # Coerce types for suggest() keys and notebook GP kernel keys
    out = {}
    for k, v in merged.items():
        if v is None and k in ("output_warping", "kernel", "n_cand"):
            out[k] = None
        elif k == "length_scale_bounds" and isinstance(v, (list, tuple)) and len(v) >= 2:
            out[k] = (float(v[0]), float(v[1]))
        elif k in ("constant_scale_bounds", "white_noise_bounds") and isinstance(v, (list, tuple)) and len(v) >= 2:
            out[k] = (float(v[0]), float(v[1]))
        elif k == "boundary_margin" and isinstance(v, (int, float)):
            out[k] = float(v)
        elif k in ("xi", "kappa", "agree_threshold", "min_dist_threshold", "duplicate_threshold",
                   "constant_kernel_scale", "length_scale", "gp_alpha", "matern_nu", "white_noise_level"):
            out[k] = float(v)
        elif k in ("n_restarts", "n_cand", "seed"):
            out[k] = int(v) if v is not None else None
        elif k in ("optimize_kernel", "use_ensemble_aquisation"):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


def _get_candidates(n_cand: int, dim: int, method: str = "sobol", seed: int = 42) -> np.ndarray:
    """Space-filling candidates in [0, 1]^dim. method in ('sobol', 'lhs')."""
    if method == "sobol":
        # skopt Sobol: skip=0, randomize; n must be power of 2 for balance
        sampler = Sobol(skip=0, randomize=True)
        n = max(n_cand, 2 ** max(10, int(np.ceil(np.log2(dim + 1)))))
        n = 2 ** int(np.ceil(np.log2(n)))
        space = [(0.0, 1.0)] * dim
        pts = sampler.generate(space, n)
        pts = np.array(pts, dtype=np.float64)[:n_cand]
    else:
        sampler = Lhs(criterion="maximin", iterations=10)
        space = [(0.0, 1.0)] * dim
        pts = np.array(sampler.generate(space, n_cand), dtype=np.float64)
    return np.clip(pts, 1e-10, 1.0 - 1e-10)


def suggest(
    X: np.ndarray,
    y: np.ndarray,
    bounds: list[tuple[float, float]] | None = None,
    *,
    config_path: str | Path | None = None,
    function_id: int | None = None,
    config: dict[str, Any] | None = None,
    # warping
    output_warping: str | None = None,
    # kernel
    kernel: str | None = None,
    optimize_kernel: bool = True,
    n_restarts: int = 10,
    length_scale_bounds: tuple[float, float] = (1e-10, 20.0),
    # acquisition
    xi: float = 0.15,
    kappa: float = 3.0,
    solo_strategy: str = "EI",
    use_ensemble: bool = True,
    agree_threshold: float = 0.15,
    # candidates
    n_cand: int | None = None,
    candidate_method: str = "sobol",
    min_dist_threshold: float = 0.05,
    boundary_margin: float = 0.0,
    # fallback
    duplicate_threshold: float = 1e-3,
    seed: int = 42,
) -> np.ndarray:
    """
    Suggest next query point given observations (X, y). Maximization.

    Hyperparameters can be read from a YAML config (per function): pass
    config_path and function_id (1..8), or pass a pre-loaded config dict.
    Any explicit keyword argument overrides the config.

    Parameters
    ----------
    X : np.ndarray
        Shape (n, d). Observed inputs.
    y : np.ndarray
        Shape (n,). Observed outputs (maximize).
    bounds : list of (low, high) or None
        Per-dimension bounds. If None, use [0, 1]^d.
    config_path : str or Path or None
        Path to configs/bayesian_optimizer.yaml. If set with function_id, config is loaded and merged.
    function_id : int or None
        Challenge function 1..8. Used to select the function_1..function_8 section in the YAML.
    config : dict or None
        Pre-loaded hyperparameter dict (e.g. from load_mybo_config()). Overridden by explicit kwargs.
    output_warping : str or None
        None/"none": no warp. "log" or "boxcox": warp y before GP fit.
    kernel : str or None
        None: auto-select by LML among RBF, Matern, RBF+White. Else "RBF"|"Matern"|"RBF+WhiteKernel".
    optimize_kernel : bool
        Optimize kernel hyperparameters.
    n_restarts : int
        Restarts for kernel optimizer.
    length_scale_bounds, xi, kappa, solo_strategy, use_ensemble, agree_threshold
        As in notebooks.
    n_cand : int or None
        Number of candidate points. Default: 2^14 for d<=4, 2^16 for d>4.
    candidate_method : str
        "sobol" or "lhs".
    min_dist_threshold : float
        Mask candidates within this L2 distance of any observation.
    boundary_margin : float
        If > 0, mask candidates with any coord in [0,margin] or [1-margin,1].
    duplicate_threshold : float
        If suggested point is within this distance of an observation, use high-distance fallback.
    seed : int
        Random seed.

    Returns
    -------
    x_next : np.ndarray
        Shape (d,) in [0, 0.999999]^d (portal format).
    """
    # Merge config (YAML or dict) then explicit kwargs (kwargs override config)
    merged: dict[str, Any] = {}
    if config is not None:
        merged.update({k: v for k, v in config.items() if k in _SUGGEST_CONFIG_KEYS})
    if config_path is not None or function_id is not None:
        merged.update(load_mybo_config(config_path=config_path, function_id=function_id))
    explicit = {
        "output_warping": output_warping, "kernel": kernel, "optimize_kernel": optimize_kernel,
        "n_restarts": n_restarts, "length_scale_bounds": length_scale_bounds,
        "xi": xi, "kappa": kappa, "solo_strategy": solo_strategy, "use_ensemble": use_ensemble,
        "agree_threshold": agree_threshold, "n_cand": n_cand, "candidate_method": candidate_method,
        "min_dist_threshold": min_dist_threshold, "boundary_margin": boundary_margin,
        "duplicate_threshold": duplicate_threshold, "seed": seed,
    }
    merged.update(explicit)
    output_warping = merged.get("output_warping")
    kernel = merged.get("kernel")
    optimize_kernel = merged.get("optimize_kernel", True)
    n_restarts = merged.get("n_restarts", 10)
    length_scale_bounds = merged.get("length_scale_bounds", (1e-10, 20.0))
    xi = merged.get("xi", 0.15)
    kappa = merged.get("kappa", 3.0)
    solo_strategy = merged.get("solo_strategy", "EI")
    # Config key is use_ensemble_aquisation; suggest() kwarg remains use_ensemble
    use_ensemble = merged.get("use_ensemble", merged.get("use_ensemble_aquisation", True))
    agree_threshold = merged.get("agree_threshold", 0.15)
    n_cand = merged.get("n_cand")
    candidate_method = merged.get("candidate_method", "sobol")
    min_dist_threshold = merged.get("min_dist_threshold", 0.05)
    boundary_margin = merged.get("boundary_margin", 0.0)
    duplicate_threshold = merged.get("duplicate_threshold", 1e-3)
    seed = merged.get("seed", 42)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, d = X.shape
    assert len(y) == n, "X and y length mismatch"
    if bounds is None:
        bounds = [(0.0, 1.0)] * d

    # Warp y if requested
    if apply_output_warping and output_warping and output_warping.strip().lower() not in ("none", ""):
        y_fit, _, _ = apply_output_warping(y, mode=output_warping)
    else:
        y_fit = y.copy()

    best_idx = np.argmax(y_fit)
    best_y = float(y_fit[best_idx])
    best_x_so_far = X[best_idx]

    # Candidate count
    if n_cand is None:
        n_cand = 2 ** 16 if d > 4 else 2 ** 14
    n_cand = min(n_cand, 2 ** 18)

    # Scale to [0,1] if bounds differ
    X_01 = X.copy()
    for j in range(d):
        lo, hi = bounds[j][0], bounds[j][1]
        if hi != lo:
            X_01[:, j] = (X[:, j] - lo) / (hi - lo)
        else:
            X_01[:, j] = 0.5

    # Minimization surrogate: fit on -y_fit
    y_neg = -y_fit

    # Kernels
    c = ConstantKernel(1.0, constant_value_bounds=(1e-10, 1e10))
    rbf = RBF(length_scale=1.0, length_scale_bounds=length_scale_bounds)
    matern = Matern(nu=1.5, length_scale_bounds=length_scale_bounds)
    white = WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-12, 1e1))

    kernel_RBF = c * rbf
    kernel_Matern = c * matern
    kernel_RBF_noise = c * rbf + white

    optimizer = "fmin_l_bfgs_b" if optimize_kernel else None
    restarts = n_restarts if optimize_kernel else 0
    alpha = 1e-6

    gp_RBF = GaussianProcessRegressor(
        kernel=kernel_RBF, alpha=alpha, optimizer=optimizer, n_restarts_optimizer=restarts, normalize_y=True
    )
    gp_Matern = GaussianProcessRegressor(
        kernel=kernel_Matern, alpha=alpha, optimizer=optimizer, n_restarts_optimizer=restarts, normalize_y=True
    )
    gp_RBF_noise = GaussianProcessRegressor(
        kernel=kernel_RBF_noise, alpha=alpha, optimizer=optimizer, n_restarts_optimizer=restarts, normalize_y=True
    )

    gp_RBF.fit(X_01, y_neg)
    gp_Matern.fit(X_01, y_neg)
    gp_RBF_noise.fit(X_01, y_neg)

    # Best by LML (sklearn minimizes negative LML)
    lml_RBF = gp_RBF.log_marginal_likelihood_value_
    lml_Matern = gp_Matern.log_marginal_likelihood_value_
    lml_RBF_noise = gp_RBF_noise.log_marginal_likelihood_value_
    lmls = [(lml_RBF, gp_RBF, "RBF"), (lml_Matern, gp_Matern, "Matérn"), (lml_RBF_noise, gp_RBF_noise, "RBF+WhiteKernel")]
    lmls.sort(key=lambda t: t[0], reverse=True)
    best_name = lmls[0][2]
    best_gp_neg = lmls[0][1]

    # Candidates in [0,1]^d
    candidate_pts = _get_candidates(n_cand, d, method=candidate_method, seed=seed)

    # Mask: too close to observations
    dists = np.sqrt(((candidate_pts[:, None, :] - X_01[None, :, :]) ** 2).sum(axis=2))
    min_dist_cand = np.min(dists, axis=1)
    too_close = min_dist_cand < min_dist_threshold
    near_boundary = False
    if boundary_margin > 0:
        near_boundary = (
            (candidate_pts < boundary_margin).any(axis=1) | (candidate_pts > 1.0 - boundary_margin).any(axis=1)
        )
    masked = too_close | near_boundary

    # High-distance fallback
    sum_d = dists.sum(axis=1)
    next_x_high_dist = candidate_pts[np.argmax(sum_d)].ravel()

    # Acquisition (skopt: minimize LCB so we pass GP on -y; y_opt = -best_y)
    EI_vals = gaussian_ei(candidate_pts, best_gp_neg, y_opt=-best_y, xi=xi)
    PI_vals = gaussian_pi(candidate_pts, best_gp_neg, y_opt=-best_y, xi=xi)
    LCB_vals = gaussian_lcb(candidate_pts, best_gp_neg, kappa=kappa)
    UCB_vals = -LCB_vals

    EI_vals = np.where(masked, -np.inf, EI_vals)
    PI_vals = np.where(masked, -np.inf, PI_vals)
    UCB_vals = np.where(masked, -np.inf, UCB_vals)

    def pick(acq, fallback):
        idx = np.argmax(acq)
        if masked[idx]:
            return fallback
        return candidate_pts[idx].ravel()

    x_best_EI = pick(EI_vals, next_x_high_dist)
    x_best_PI = pick(PI_vals, next_x_high_dist)
    x_best_UCB = pick(UCB_vals, next_x_high_dist)

    if use_ensemble:
        pts = np.array([x_best_EI, x_best_PI, x_best_UCB])
        d12 = np.linalg.norm(pts[0] - pts[1])
        d13 = np.linalg.norm(pts[0] - pts[2])
        d23 = np.linalg.norm(pts[1] - pts[2])
        max_d = max(d12, d13, d23)
        if max_d >= agree_threshold:
            next_x = np.mean(pts, axis=0)
        else:
            next_x = x_best_EI.copy()
    else:
        m = {"EI": x_best_EI, "PI": x_best_PI, "UCB": x_best_UCB}
        next_x = m.get(solo_strategy, x_best_EI).copy()

    next_x = np.clip(next_x, 0.0, 0.999999)

    # Duplicate check
    dist_to_obs = np.sqrt(((X_01 - next_x) ** 2).sum(axis=1))
    nearest_dist = dist_to_obs.min()
    if nearest_dist < duplicate_threshold:
        next_x = np.clip(next_x_high_dist, 0.0, 0.999999)

    # Map back to original bounds if not [0,1]
    out = next_x.copy()
    for j in range(d):
        lo, hi = bounds[j][0], bounds[j][1]
        if abs(hi - lo) > 1e-12:
            out[j] = lo + next_x[j] * (hi - lo)
    out = np.clip(out, [b[0] for b in bounds], [min(b[1], 0.999999) for b in bounds])
    return out
