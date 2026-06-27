# Model Card — SEBO: Sample-Efficient Bayesian Optimizer

**Author:** Nikolas Karefyllidis, PhD
**Version:** 1.4
**Last updated:** June 2026

---

## Overview

**Name:** SEBO — GP Surrogate with Ensemble Acquisition
**Type:** Sequential, model-based black-box optimiser (maximisation)
**Install:** `pip install git+https://github.com/karefyllidis/SEBO.git`
**Framework:** scikit-learn `GaussianProcessRegressor`, scikit-optimize acquisition functions, Sobol/LHS candidate sampling
**Code:** `src/optimizers/optimizer.py`, `src/optimizers/my_bayesian/`, `src/utils/`

---

## Intended Use

**Suitable for:**
- Single-objective, noiseless or mildly noisy black-box maximisation over a bounded, continuous domain [0, 1]^d.
- Low-to-moderate dimensionality (d = 2–8), small evaluation budgets (10–80 observations).
- Problems where the objective is smooth enough to be modelled by a stationary GP kernel (RBF or Matérn).

**Not suitable for:**
- High-dimensional spaces (d >> 10) where GP covariance matrices become computationally expensive and sparse data leads to prior-dominated predictions.
- Strongly non-stationary functions (discontinuities, phase transitions, highly variable length scales across the domain).
- Multi-objective, constrained, or integer/mixed-variable optimisation without modification.

---

## Model Description

**Input:** A set of prior observations {(xᵢ, yᵢ)} where xᵢ ∈ [0, 1]^d and yᵢ ∈ ℝ. At each round, the model receives one new (x, y) pair from the oracle and produces a single recommended next query point x* ∈ [0, 1]^d.

**Output:** A single d-dimensional vector x* in [0, 1]^d.

**Architecture.** The pipeline has four stages:

1. **Surrogate fitting.** Three GP kernels are fitted to all prior observations:
   - RBF (squared exponential): smooth, infinitely differentiable.
   - Matérn ν=1.5: once-differentiable; more robust to rougher surfaces.
   - RBF + WhiteKernel: RBF with an explicit noise term; used when the oracle is stochastic.
   Kernel hyperparameters (length scale, output scale, noise level) are optimised by maximising log-marginal likelihood (LML) with 5–25 random restarts depending on dimensionality. The kernel with the highest LML is selected automatically.

2. **Output warping.** For skewed or multi-scale y, the GP is fitted on a transformed target (`log` or `Box-Cox`). Acquisition and incumbent tracking operate in warped space; raw y is stored and reported.

3. **Acquisition maximisation.** Three acquisition functions are computed over a Sobol/LHS candidate set of 2^15–2^18 points: **EI** (Expected Improvement), **PI** (Probability of Improvement), **UCB** (Upper Confidence Bound). Candidates within `MIN_DIST_THRESHOLD` (L2 distance) of any prior observation are masked to prevent re-querying.

4. **Ensemble decision.** Let **d_max** be the maximum pairwise L2 distance among the EI, PI, and UCB argmaxes. If **d_max < AGREE_THRESHOLD**, the next query is the **EI** argmax. If **d_max ≥ AGREE_THRESHOLD**, the next query is the **centroid** of all three — a soft blend when acquisitions disagree.

**Benchmark baselines.** SEBO is compared head-to-head against Optuna-TPE, TuRBO, DE-GP-EI, and Random Search in `notebooks/sebo_benchmark.ipynb`.

---

## NeurIPS 2020 BBO Challenge Results

Applied to 8 unknown oracle functions (2D–8D), one evaluation per function per round, 13 rounds:

| Function | Initial best y | Final best y | Improvement | Notes |
|----------|---------------|--------------|-------------|-------|
| F1 | ~0.0 | 0.6704 | Large | Narrow high region found in round 10 |
| F2 | ~0.19 | 0.7248 | Large | Best at x₁ ≈ 0.70 |
| F3 | ~−0.44 | −0.0032 | Large | Output always negative; improvement = less negative |
| F4 | ~0.04 | 0.2987 | Moderate | High variance; many local optima |
| F5 | ~1700 | 7493.9 | Very large | Near-boundary region [0.99, …] confirmed |
| F6 | ~−1.3 | −0.1402 | Large | Noisy oracle; repeated x returned different y |
| F7 | ~0.003 | 2.7968 | Large | Steady improvement via UCB exploration |
| F8 | ~5.6 | 9.9619 | Large | Consistent upward trend across all rounds |

**Diagnostics used:** Cumulative-best plots, GP mean/uncertainty surface plots (2D contours; pairwise projection slices for d ≥ 3), acquisition surface overlays, solver comparison plots (SEBO vs Optuna/TuRBO/DE-GP-EI) in each notebook's Section 6.

---

## Strategy Evolution Across Rounds

| Rounds | Key changes |
|--------|-------------|
| 1–3 | EI only, single RBF kernel, Sobol candidates. Exploration-heavy (ξ = 0.05–0.1). |
| 4–6 | LML kernel selection introduced. Matérn and RBF+WhiteKernel added. Ensemble acquisition (EI+PI+UCB). Output warping for skewed objectives. |
| 7–8 | `N_RESTARTS_KERNEL` scaled with dimensionality. Transition from exploration to exploitation for functions with stable incumbents. |
| 9–13 | Mostly exploitation. F6 identified as noisy oracle; duplicate rows retained, `GP_ALPHA` raised to 1e-3 for stability. |

---

## Assumptions and Limitations

**Key assumptions:**

1. **Stationarity.** The GP kernel assumes constant length scale and smoothness across the domain.
2. **Smoothness.** RBF/Matérn kernels assume at least once-differentiable functions.
3. **Fixed budget.** ξ, κ, and ensemble `AGREE_THRESHOLD` are set manually, not on an automated schedule.
4. **Determinism.** Default `GP_ALPHA = 1e-6` assumes noiseless oracle. Raise to `1e-3` for noisy oracles (as with F6).

**Known limitations:**

- **Curse of dimensionality.** With 20–50 observations in 5D–8D, the GP extrapolates widely in unvisited regions.
- **Exploitation bias.** Query distribution concentrates near early high-value regions; large portions of the domain may remain unexplored.
- **Isotropic kernel.** Single scalar length scale across all dimensions. ARD (Automatic Relevance Determination) would allow per-dimension length scales.
- **No restart mechanism.** Once the surrogate becomes confident in a local region, acquisition rarely proposes globally different candidates.

---

## Transparency and Reproducibility

Every query and oracle response is logged in local `observations.csv` files (gitignored — oracle trails are not published). Each notebook documents the kernel chosen, its LML-optimised hyperparameters, the acquisition function settings, and the reasoning in markdown cells. With matching observation files, Python environment (`requirements.txt`), and the same random seeds, any round's suggested query can be reproduced deterministically.
