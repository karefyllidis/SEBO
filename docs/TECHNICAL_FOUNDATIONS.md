# Technical Foundations

Reference for the design decisions, key literature, and library choices behind SEBO.

---

## Core Approach

- **Method:** Bayesian optimisation (BO) with a **Gaussian process (GP) surrogate** and an acquisition function (EI, PI, UCB).
- **Why BO:** Standard sample-efficient framework for expensive black-box objectives with limited evaluations. The loop (fit surrogate → maximise acquisition → evaluate → update) has theoretical regret bounds and is well-validated in practice.
- **Why GPs:** They provide a predictive mean and **uncertainty** σ(x) natively (required by EI/UCB), are data-efficient for small n (10–50 points), and scale to the target dimensions (2D–8D).
- **Ensemble acquisition:** If EI, PI, and UCB argmaxes are close in L2 (below `AGREE_THRESHOLD`), SEBO uses EI; otherwise the centroid — avoids over-committing to one strategy when acquisitions disagree.
- **Output warping:** When y spans many orders of magnitude or is heavily skewed, the GP is fitted on `log` or `Box-Cox` transformed targets (HEBO-inspired). The surrogate and acquisition operate in warped space; raw y is stored and reported. Implemented in `src/utils/warping.py`.

---

## Key Papers

| Source | Idea / technique | Application in SEBO |
|--------|------------------|---------------------------------|
| **Rasmussen & Williams**, *Gaussian Processes for Machine Learning* | GP regression; kernel choice (RBF, Matérn); hyperparameters via log-marginal likelihood | Justifies surrogate choice and kernel selection by LML |
| **Jones et al.** (1998) — Expected Improvement | EI as a principled acquisition balancing exploration and exploitation | Primary acquisition function; used in ensemble |
| **NeurIPS 2020 BBO Challenge** | Suggest–observe API; space-filling candidates (Sobol); avoidance of re-querying | Aligns implementation with a validated benchmark; motivates duplicate-avoidance and Sobol candidates |
| **HEBO** (Cowen-Rivers et al., 2022) | Output warping (log/Box-Cox of y before GP fit) | `OUTPUT_WARPING` in all notebooks; default `"log"` for skewed objectives |
| **TuRBO** (Eriksson et al., 2019) | Trust-region BO for high-dimensional spaces | Baseline in `sebo_benchmark.ipynb` via `src/optimizers/wrappers/turbo_solver.py` |

---

## Library Choices

| Library | Role | Why chosen |
|---------|------|------------------------------|
| **scikit-learn** (`GaussianProcessRegressor`) | Core GP surrogate | Stable API, built-in LML kernel optimisation, good for small n. GPyTorch would scale better but adds unnecessary complexity for this budget. |
| **scikit-optimize (skopt)** | Compute EI/PI/UCB; generate Sobol/LHS candidates | Widely used in BO; Sobol gives low-discrepancy coverage; used in NeurIPS 2020 starter kit. |
| **SciPy** (`differential_evolution`) | DE-GP-EI baseline: maximise GP-EI continuously on [0,1]^d | Continuous acquisition maximisation for the DE-GP-EI comparison solver. |
| **BoTorch / TuRBO** | TuRBO baseline | Trust-region BO for the benchmark comparison. |
| **Optuna** | TPE baseline | Standard open-source HPO solver for comparison. |
| **NumPy, Matplotlib** | Numerical computation and visualisation | Standard, stable stack. |

---

## Where to Find It

| File | Contents |
|------|----------|
| `src/optimizers/optimizer.py` | `BayesianOptimizer` — stateful `suggest / observe` API |
| `src/optimizers/my_bayesian/my_gp_skopt.py` | GP surrogate + ensemble acquisition (EI/PI/UCB) |
| `src/optimizers/wrappers/` | Optuna, TuRBO, DE-GP-EI, Hyperopt solver wrappers |
| `src/utils/warping.py` | Log / Box-Cox output warping |
| `src/utils/sampling_utils.py` | Sobol / LHS candidate generation |
| `notebooks/sebo_benchmark.ipynb` | Head-to-head benchmark with adaptive stopping |
| `notebooks/demo_sklearn_hpo.ipynb` | Self-contained HPO demo on sklearn Digits |
| `docs/model_card.md` | Architecture, performance, limitations |
