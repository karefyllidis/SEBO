"""
BayesianOptimizer: stateful suggest/observe API wrapping the MyBO GP surrogate.

The functional core (`my_gp_skopt.suggest`) does the heavy lifting — kernel
selection by LML, ensemble acquisition (EI/PI/UCB), Sobol/LHS candidates,
duplicate masking, output warping.  This class adds the statefulness needed
for real optimization loops: accumulate (X, y), call suggest(), call observe().

Usage
-----
    from src.optimizers.optimizer import BayesianOptimizer

    bounds = [(10, 500), (2, 30), (0.01, 0.5), (0.1, 0.99)]
    opt = BayesianOptimizer(bounds=bounds, xi=0.01, kappa=0.75)

    # Option A — explicit loop (most transparent for notebooks/demos)
    opt.fit(X_warm, y_warm)               # pre-load warm-start data
    for _ in range(n_iter):
        x_next = opt.suggest()            # GP + ensemble acquisition
        y_next = objective(x_next)
        opt.observe(x_next, y_next)

    best_x, best_y = opt.best

    # Option B — one-liner with built-in LHS warm-start
    opt.run(objective, n_iter=20, n_init=10, verbose=True)

See notebooks/demo_sklearn_hpo.ipynb for an end-to-end example tuning a
RandomForestClassifier on a real dataset and comparing against random search.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.optimizers.my_bayesian.my_gp_skopt import suggest as _mybo_suggest


class BayesianOptimizer:
    """Stateful GP-based Bayesian Optimiser with ensemble acquisition.

    Wraps ``my_gp_skopt.suggest()`` behind a ``fit / suggest / observe`` API
    suited to sequential black-box optimization loops.  Maximizes the
    objective (higher y is better).

    Parameters
    ----------
    bounds : list of (low, high)
        Per-dimension search bounds.  The optimizer normalises to [0, 1]^d
        internally; returned points are in the original space.
    output_warping : str or None
        ``"log"`` or ``"boxcox"`` for skewed objectives; ``None`` (default)
        for no warping.
    kernel : str or None
        Force a kernel family: ``"RBF"``, ``"Matern"``, ``"RBF+WhiteKernel"``.
        ``None`` (default) selects the best by log-marginal likelihood.
    optimize_kernel : bool
        Whether to MLE-tune kernel hyperparameters.  Default ``True``.
    n_restarts : int
        Number of L-BFGS-B restarts for kernel hyperparameter optimisation.
    xi : float
        Exploration parameter ξ for EI and PI.  Lower = more exploitation.
    kappa : float
        Exploration parameter κ for UCB.  Lower = more exploitation.
    use_ensemble : bool
        If ``True`` (default), run EI+PI+UCB and use the centroid when the
        three argmaxes disagree (max pairwise L2 ≥ ``agree_threshold``);
        otherwise use EI alone.
    agree_threshold : float
        Ensemble agreement threshold (max pairwise L2 in normalised space).
    candidate_method : str
        ``"sobol"`` (default) or ``"lhs"`` for candidate pool sampling.
    n_cand : int or None
        Candidate pool size.  ``None`` → 2^14 for d ≤ 4, 2^16 for d > 4.
    min_dist_threshold : float
        Mask candidates within this L2 (normalised) of any observation.
    seed : int
        Random seed for candidate sampling and fallback selection.
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        *,
        output_warping: str | None = None,
        kernel: str | None = None,
        optimize_kernel: bool = True,
        n_restarts: int = 10,
        xi: float = 0.01,
        kappa: float = 0.75,
        use_ensemble: bool = True,
        agree_threshold: float = 0.22,
        candidate_method: str = "sobol",
        n_cand: int | None = None,
        min_dist_threshold: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.bounds = list(bounds)
        self._d = len(bounds)
        self._suggest_kwargs: dict[str, Any] = {
            "output_warping": output_warping,
            "kernel": kernel,
            "optimize_kernel": optimize_kernel,
            "n_restarts": n_restarts,
            "xi": xi,
            "kappa": kappa,
            "use_ensemble": use_ensemble,
            "agree_threshold": agree_threshold,
            "candidate_method": candidate_method,
            "n_cand": n_cand,
            "min_dist_threshold": min_dist_threshold,
            "seed": seed,
        }
        self._X: list[np.ndarray] = []
        self._y: list[float] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianOptimizer":
        """Pre-load observations (e.g. warm-start data).  Chainable.

        Replaces any observations already stored.  To *add* observations
        incrementally use ``observe()`` instead.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2 or X.shape[1] != self._d:
            raise ValueError(f"X must be shape (n, {self._d}); got {X.shape}")
        if len(y) != len(X):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        self._X = list(X)
        self._y = list(y)
        return self

    def observe(self, x: np.ndarray | list[float], y: float) -> None:
        """Record one evaluated point (x, y).

        Parameters
        ----------
        x : array-like, shape (d,)
            The evaluated input point, in the original (un-normalised) space.
        y : float
            The corresponding objective value (higher is better).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != self._d:
            raise ValueError(f"x must have {self._d} elements; got {x.size}")
        self._X.append(x.copy())
        self._y.append(float(y))

    def suggest(self) -> np.ndarray:
        """Return the next recommended query point.

        Uses the GP surrogate fit on all current observations, followed by
        ensemble acquisition (EI/PI/UCB) over a Sobol/LHS candidate pool.

        Returns
        -------
        x_next : np.ndarray, shape (d,)
            Next query point in the original bounds.

        Raises
        ------
        RuntimeError
            If fewer than 2 observations have been recorded (GP cannot be fit).
        """
        if len(self._X) < 2:
            raise RuntimeError(
                f"BayesianOptimizer needs at least 2 observations to fit a GP "
                f"(currently has {len(self._X)}).  Call observe() or fit() first."
            )
        X_arr = np.array(self._X, dtype=np.float64)
        y_arr = np.array(self._y, dtype=np.float64)
        return _mybo_suggest(X_arr, y_arr, bounds=self.bounds, **self._suggest_kwargs)

    # ------------------------------------------------------------------
    # Full optimization loop (convenience wrapper)
    # ------------------------------------------------------------------

    def run(
        self,
        objective: Callable[[np.ndarray], float],
        n_iter: int,
        n_init: int = 5,
        *,
        init_method: str = "lhs",
        verbose: bool = True,
    ) -> "BayesianOptimizer":
        """Full optimization loop: LHS warm-start followed by BO iterations.

        If observations are already loaded (via ``fit()`` or prior ``observe()``
        calls), skips the warm-start and goes straight to BO iterations.

        Parameters
        ----------
        objective : callable
            Maps x (shape (d,)) → float.  Maximization.
        n_iter : int
            Number of BO iterations after warm-start.
        n_init : int
            Number of space-filling warm-start evaluations.
        init_method : str
            ``"lhs"`` (default, maximin Latin Hypercube) or ``"random"``.
        verbose : bool
            Print progress.
        """
        if len(self._X) == 0:
            X_warm = self._sample_init(n_init, method=init_method)
            for i, x in enumerate(X_warm):
                y = objective(x)
                self.observe(x, y)
                if verbose:
                    print(f"  init {i + 1:2d}/{n_init}: y={y:.6f}  best={self.best_y:.6f}")

        for i in range(n_iter):
            x_next = self.suggest()
            y_next = objective(x_next)
            self.observe(x_next, y_next)
            if verbose:
                print(f"  iter {i + 1:2d}/{n_iter}: y={y_next:.6f}  best={self.best_y:.6f}")

        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_init(self, n: int, method: str = "lhs") -> np.ndarray:
        """Sample *n* initial points within bounds using LHS or uniform random."""
        if method == "lhs":
            from skopt.sampler import Lhs
            sampler = Lhs(criterion="maximin", iterations=50)
            pts_01 = np.array(
                sampler.generate([(0.0, 1.0)] * self._d, n), dtype=np.float64
            )
        else:
            rng = np.random.default_rng(self._suggest_kwargs.get("seed", 42))
            pts_01 = rng.random((n, self._d))

        out = np.zeros_like(pts_01)
        for j, (lo, hi) in enumerate(self.bounds):
            out[:, j] = lo + pts_01[:, j] * (hi - lo)
        return out

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def X(self) -> np.ndarray:
        """All observed inputs, shape (n, d)."""
        if not self._X:
            return np.empty((0, self._d), dtype=np.float64)
        return np.array(self._X, dtype=np.float64)

    @property
    def y(self) -> np.ndarray:
        """All observed outputs, shape (n,)."""
        return np.array(self._y, dtype=np.float64)

    @property
    def best_y(self) -> float:
        """Best observed output so far (``-inf`` if no observations)."""
        return float(np.max(self._y)) if self._y else float("-inf")

    @property
    def best(self) -> tuple[np.ndarray, float]:
        """(best_x, best_y): the point with the highest observed output.

        Raises
        ------
        RuntimeError
            If no observations have been recorded yet.
        """
        if not self._X:
            raise RuntimeError("No observations recorded yet.")
        idx = int(np.argmax(self._y))
        return self._X[idx].copy(), float(self._y[idx])

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._X)

    def __repr__(self) -> str:
        n = len(self._X)
        best = f"{self.best_y:.4f}" if n > 0 else "n/a"
        ens = self._suggest_kwargs.get("use_ensemble", True)
        acq = "EI+PI+UCB ensemble" if ens else self._suggest_kwargs.get("solo_strategy", "EI")
        return f"BayesianOptimizer(d={self._d}, n_obs={n}, best_y={best}, acq={acq})"
