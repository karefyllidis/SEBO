"""
TuRBO wrapper: suggest one next point given (X, y). Maximization.

TuRBO (Trust Region Bayesian Optimization) excels at high-dimensional problems
(50+ parameters) where standard Bayesian optimization struggles.

Uses BoTorch TuRBO-1 implementation. Same interface: suggest(X, y, bounds) -> x_next.

One-shot vs full TuRBO
----------------------
Each call fits a GP, builds a trust region around the **current** best training point,
and returns one candidate. Full TuRBO-1 in the literature **persists** trust-region
state (expand/shrink ``length``) across iterations; this wrapper **does not** carry
state between calls. Use it for ``(X, y) -> x_next`` snapshots; for long runs,
either call it inside your own loop and optionally adapt the TR size externally,
or use a multi-step BoTorch TuRBO pipeline.

Dependency
----------
Requires BoTorch (and PyTorch, GPyTorch): pip install botorch
Raises ImportError if BoTorch is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    import torch
    import gpytorch
    from gpytorch.constraints import Interval
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from torch.quasirandom import SobolEngine
    from botorch.acquisition import qLogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.generation import MaxPosteriorSampling
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    _TURBO_AVAILABLE = True
except ImportError:
    _TURBO_AVAILABLE = False


def load_turbo_config(
    config_path: str | Path | None = None,
    function_id: int | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load TuRBO hyperparameters from configs/turbo_optimizer.yaml."""
    if not _YAML_AVAILABLE:
        return {}
    if config_path is None:
        project_root = project_root or Path(__file__).resolve().parent.parent.parent.parent
        config_path = Path(project_root) / "configs" / "turbo_optimizer.yaml"
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
        if k == "acqf":
            out[k] = str(v) if v else "ts"
        elif k in ("num_restarts", "raw_samples", "seed"):
            out[k] = int(v) if v is not None else (10 if k == "num_restarts" else (512 if k == "raw_samples" else 42))
        elif k == "n_candidates":
            out[k] = int(v) if v is not None else None
        else:
            out[k] = v
    return out


@dataclass
class _TurboState:
    """TuRBO-1 state for one suggest call (no history across calls)."""
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    best_value: float = -float("inf")


def suggest(
    X: np.ndarray,
    y: np.ndarray,
    bounds: list[tuple[float, float]] | None = None,
    *,
    config_path: str | Path | None = None,
    function_id: int | None = None,
    config: dict[str, Any] | None = None,
    acqf: str = "ts",
    seed: int = 42,
    num_restarts: int = 10,
    raw_samples: int = 512,
    n_candidates: int | None = None,
) -> np.ndarray:
    """
    Suggest next query using TuRBO-1. Maximization.

    Parameters
    ----------
    X : np.ndarray
        Shape (n, d). Observed inputs.
    y : np.ndarray
        Shape (n,). Observed outputs (maximize).
    bounds : list of (low, high) or None
        Per-dimension bounds. If None, use [0, 1]^d.
    acqf : str
        "ts" (Thompson sampling) or "ei" (Expected Improvement).
    seed : int
        Random seed.

    Returns
    -------
    x_next : np.ndarray
        Shape (d,) in bounds (clipped to [0, 0.999999] for portal).
    """
    if not _TURBO_AVAILABLE:
        raise ImportError("BoTorch is required for turbo_solver. Install with: pip install botorch")

    cfg: dict[str, Any] = load_turbo_config(config_path=config_path, function_id=function_id)
    if isinstance(config, dict):
        cfg = {**cfg, **config}
    cfg.setdefault("acqf", acqf)
    cfg.setdefault("seed", seed)
    cfg.setdefault("num_restarts", num_restarts)
    cfg.setdefault("raw_samples", raw_samples)
    cfg.setdefault("n_candidates", n_candidates)
    acqf = str(cfg.get("acqf", "ts"))
    seed = int(cfg.get("seed", 42))
    num_restarts = int(cfg.get("num_restarts", 10))
    raw_samples = int(cfg.get("raw_samples", 512))
    n_candidates = cfg.get("n_candidates")

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

    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    X_t = torch.tensor(X_01, dtype=dtype, device=device)
    if X_t.dim() == 1:
        X_t = X_t.unsqueeze(0)
    Y_t = torch.tensor(y, dtype=dtype, device=device).reshape(-1, 1)
    # Normalize Y for numerical stability
    train_Y = (Y_t - Y_t.mean()) / (Y_t.std() + 1e-8)

    best_value = float(Y_t.max())
    state = _TurboState(dim=d, batch_size=1, best_value=best_value)

    # Fit GP
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(
        MaternKernel(nu=2.5, ard_num_dims=d, lengthscale_constraint=Interval(0.005, 4.0))
    )
    model = SingleTaskGP(X_t, train_Y, covar_module=covar_module, likelihood=likelihood)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Trust region bounds around best point
    x_center = X_t[Y_t.argmax().item(), :].clone()
    with torch.no_grad():
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        if weights.dim() == 0:
            weights = weights.unsqueeze(0)  # 1D case: keep as (1,) for len()/broadcast
        weights = weights / (weights.mean() + 1e-8)
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)) + 1e-8)
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        n_cand = n_candidates if n_candidates is not None else min(5000, max(2000, 200 * d))
        sobol = SobolEngine(d, scramble=True, seed=seed)
        pert = sobol.draw(n_cand).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert
        prob_perturb = min(20.0 / d, 1.0)
        mask = torch.rand(n_cand, d, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        if len(ind) > 0:
            mask[ind, torch.randint(0, d, size=(len(ind),), device=device)] = 1
        X_cand = x_center.expand(n_cand, d).clone()
        X_cand[mask] = pert[mask]
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():
            X_next = thompson_sampling(X_cand, num_samples=1)
    else:
        ei = qLogExpectedImprovement(model, train_Y.max())
        X_next, _ = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    x_01 = X_next.squeeze().cpu().numpy()
    x_01 = np.atleast_1d(np.asarray(x_01, dtype=np.float64)).ravel()
    x_01 = np.clip(x_01, 0.0, 0.999999)

    # Map back to original bounds (ensure x_01 has d elements for indexing)
    x_01 = np.atleast_1d(x_01).ravel()[:d]
    if x_01.size < d:
        x_01 = np.resize(x_01, d)
    out = np.zeros(d, dtype=np.float64)
    for j in range(d):
        lo, hi = bounds[j][0], bounds[j][1]
        if abs(hi - lo) < 1e-12:
            out[j] = lo
        else:
            out[j] = lo + x_01[j] * (hi - lo)
    result = np.clip(out, [b[0] for b in bounds], [min(b[1], 0.999999) for b in bounds])
    result = np.atleast_1d(result).ravel()

    try:
        from src.utils.unique_query import ensure_distinct_from_observations

        result = ensure_distinct_from_observations(
            result, X, bounds, min_l2=float(cfg.get("min_l2_from_obs", 1e-3)), seed=seed
        )
    except ImportError:
        pass
    return result
