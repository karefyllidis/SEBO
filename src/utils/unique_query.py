"""
Ensure the next query is not (near-)identical to any existing observation.

Portal submissions use six decimal places; optimisers can still return a training
point exactly (e.g. TuRBO trust-region centre). This module enforces a minimum
L2 distance in normalised [0, 1]^d between the proposal and all rows of *X*.
"""

from __future__ import annotations

import numpy as np

try:
    from skopt.sampler import Sobol
except ImportError:
    Sobol = None  # type: ignore[misc, assignment]


def _X_to_01(X: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    out = np.zeros_like(X)
    for j in range(d):
        lo, hi = bounds[j][0], bounds[j][1]
        if abs(hi - lo) < 1e-12:
            out[:, j] = 0.5
        else:
            out[:, j] = (X[:, j] - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _x_to_01(x: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    d = x.size
    out = np.zeros(d, dtype=np.float64)
    for j in range(d):
        lo, hi = bounds[j][0], bounds[j][1]
        if abs(hi - lo) < 1e-12:
            out[j] = 0.5
        else:
            out[j] = (x[j] - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _01_to_x(x_01: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    x_01 = np.asarray(x_01, dtype=np.float64).ravel()
    d = x_01.size
    out = np.zeros(d, dtype=np.float64)
    for j in range(d):
        lo, hi = bounds[j][0], bounds[j][1]
        if abs(hi - lo) < 1e-12:
            out[j] = lo
        else:
            out[j] = lo + x_01[j] * (hi - lo)
    return np.clip(out, [b[0] for b in bounds], [min(b[1], 0.999999) for b in bounds])


def _min_l2_to_data(x_01: np.ndarray, X_01: np.ndarray) -> float:
    if X_01.size == 0:
        return float("inf")
    d = np.sqrt(((X_01 - x_01) ** 2).sum(axis=1))
    return float(d.min())


def _sobol_space_filling(n_cand: int, d: int, seed: int) -> np.ndarray:
    if Sobol is None:
        rng = np.random.default_rng(seed)
        return rng.uniform(1e-10, 1.0 - 1e-10, size=(n_cand, d))
    sampler = Sobol(skip=0, randomize=True)
    n = max(n_cand, 2 ** int(np.ceil(np.log2(max(16, n_cand)))))
    space = [(0.0, 1.0)] * d
    pts = np.array(sampler.generate(space, n), dtype=np.float64)[:n_cand]
    return np.clip(pts, 1e-10, 1.0 - 1e-10)


def _perturb_until_clear(
    x0_01: np.ndarray,
    X_01: np.ndarray,
    *,
    min_l2: float,
    rng: np.random.Generator,
    d: int,
) -> np.ndarray | None:
    """Try small random moves from *x0_01* until min distance to *X_01* >= *min_l2*."""
    if _min_l2_to_data(x0_01, X_01) >= min_l2:
        return x0_01
    scales = np.linspace(0.0005, 0.08, 40)
    for scale in scales:
        for _ in range(24):
            u = rng.normal(size=d)
            u /= np.linalg.norm(u) + 1e-12
            cand = np.clip(x0_01 + scale * u, 1e-10, 1.0 - 1e-10)
            if _min_l2_to_data(cand, X_01) >= min_l2:
                return cand
    return None


def ensure_distinct_from_observations(
    x_next: np.ndarray,
    X: np.ndarray,
    bounds: list[tuple[float, float]],
    *,
    min_l2: float = 1e-3,
    seed: int = 42,
    n_sobol: int = 8192,
    fallback_01: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return a point in *bounds* that is at least *min_l2* (Euclidean in normalised
    [0,1]^d) away from every row of *X*.

    Order: accept if already clear; else small local perturbations of *x_next*;
    else try *fallback_01* (in [0,1]^d); else Sobol candidate with largest
    minimum-distance to *X*.
    """
    x_next = np.asarray(x_next, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    d = x_next.size
    rng = np.random.default_rng(seed)
    if X.size == 0 or len(X) == 0:
        return _01_to_x(_x_to_01(x_next, bounds), bounds)

    bounds = [(float(b[0]), float(b[1])) for b in bounds]
    X_01 = _X_to_01(X, bounds)

    primary_01 = _x_to_01(x_next, bounds)
    fixed = _perturb_until_clear(primary_01, X_01, min_l2=min_l2, rng=rng, d=d)
    if fixed is not None:
        return _01_to_x(fixed, bounds)

    if fallback_01 is not None:
        fb = np.asarray(fallback_01, dtype=np.float64).ravel()
        if fb.size == d:
            fb = np.clip(fb, 0.0, 1.0)
            fixed = _perturb_until_clear(
                fb, X_01, min_l2=min_l2, rng=np.random.default_rng(seed + 1), d=d
            )
            if fixed is not None:
                return _01_to_x(fixed, bounds)

    pts = _sobol_space_filling(n_sobol, d, seed + 2)
    dists = np.sqrt(((pts[:, None, :] - X_01[None, :, :]) ** 2).sum(axis=2))
    min_dist = dists.min(axis=1)
    j = int(np.argmax(min_dist))
    x_01_pick = pts[j]
    return _01_to_x(x_01_pick, bounds)
