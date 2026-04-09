# Model Card — GP-based Bayesian Optimisation Pipeline

**Author:** Nikolas Karefyllidis, PhD
**Version:** 1.0 (Round 10 final)
**Last updated:** April 2026

---

## Overview

**Name:** MyBO — GP Surrogate with Ensemble Acquisition
**Type:** Sequential, model-based black-box optimiser (maximisation)
**Framework:** scikit-learn `GaussianProcessRegressor`, scikit-optimize acquisition functions, Sobol/LHS candidate sampling
**Repository:** `notebooks/function_N_*.ipynb`, `src/optimizers/my_bayesian/`, `src/utils/`

---

## Intended Use

**Suitable for:**
- Single-objective, noiseless or mildly noisy black-box maximisation over a bounded, continuous domain [0, 1]^d.
- Low-to-moderate dimensionality (d = 2–8), small evaluation budgets (10–50 observations).
- Problems where the objective is smooth enough to be modelled by a stationary GP kernel (RBF or Matérn).

**Not suitable for:**
- High-dimensional spaces (d >> 10) where GP covariance matrices become computationally expensive and sparse data leads to prior-dominated predictions.
- Strongly non-stationary functions (discontinuities, phase transitions, highly variable length scales across the domain).
- Multi-objective, constrained, or integer/mixed-variable optimisation without modification.
- Time-sensitive or real-time applications — the pipeline runs interactively in Jupyter notebooks and is not designed for automated, high-throughput deployment.

---

## Model Description

**Input:** A set of prior observations {(xᵢ, yᵢ)} where xᵢ ∈ [0, 1]^d and yᵢ ∈ ℝ. At each round, the model receives one new (x, y) pair from the oracle and produces a single recommended next query point x* ∈ [0, 1]^d.

**Output:** A single d-dimensional vector x* in [0, 1]^d, formatted as a hyphen-separated string to six decimal places for portal submission (e.g. `0.433599-0.417313`).

**Architecture.** The pipeline has four stages:

1. **Surrogate fitting.** Three GP kernels are fitted to all prior observations:
   - RBF (squared exponential): smooth, infinitely differentiable.
   - Matérn ν=1.5: once-differentiable; more robust to rougher surfaces.
   - RBF + WhiteKernel: RBF with an explicit noise term; used when the oracle is stochastic.
   Kernel hyperparameters (length scale, output scale, noise level) are optimised by maximising log-marginal likelihood (LML) with 5–25 random restarts depending on dimensionality. The kernel with the highest LML is selected automatically (`GP_KERNEL = None`).

2. **Output warping.** For skewed or multi-scale y, the GP is fitted on a transformed target (`log` or `Box-Cox`). Functions using log warping by default: F1, F5, F7. Acquisition and incumbent tracking operate in warped space; raw y is stored and reported.

3. **Acquisition maximisation.** Three acquisition functions are computed over a Sobol/LHS candidate set of size 2^15–2^18:
   - **EI** (Expected Improvement, ξ = 0.01 by default)
   - **PI** (Probability of Improvement, ξ = 0.01)
   - **UCB** (Upper Confidence Bound, κ = 2.0)
   Candidates within `MIN_DIST_THRESHOLD` (L2 distance) of any prior observation are masked to prevent re-querying. Boundary masking (`BOUNDARY_MARGIN`) is applied for d ≤ 3 only.

4. **Ensemble decision.** If the maximum pairwise L2 distance between the three acquisition argmaxes exceeds `AGREE_THRESHOLD = 0.15`, the next query is the centroid of the three candidates (exploration signal). Otherwise, the EI argmax is used (exploitation signal). A solo mode (`USE_ENSEMBLE = False`) is available for direct comparison.

**External baselines (Section 6 of each notebook).** For comparison, the pipeline also queries Optuna-TPE, Optuna-GP, TuRBO, and DE-GP-EI and overlays their suggestions on the same plots.

---

## Strategy Evolution Across Ten Rounds

| Rounds | Key changes |
|--------|-------------|
| 1–3 | EI only, single RBF kernel, Sobol candidates. Exploration-heavy (ξ = 0.05–0.1). Initial warm-start data only. |
| 4–6 | LML kernel selection introduced. Matérn and RBF+WhiteKernel added. Ensemble acquisition (EI+PI+UCB). Per-function `MIN_DIST_THRESHOLD` and `BOUNDARY_MARGIN`. Output warping for F1, F5, F7. |
| 7–8 | `N_RESTARTS_KERNEL` scaled with dimensionality (20 for 5D–6D). Transition from exploration to exploitation for functions with stable incumbents (F5, F8). UCB preferred for sparse high-D functions. |
| 9–10 | Mostly exploitation. F6 identified as potentially noisy (same x returned different y across rounds 8–10); duplicate rows retained and `GP_ALPHA` raised to 1e-3 for numerical stability. F5 confirmed near-boundary region as global optimum. |

---

## Performance

**Metric.** Best observed y after all rounds (incumbent value). No ground truth available; performance is relative and compared across rounds via cumulative-best plots in each notebook.

| Function | Initial best y (10 pts) | Best y after Round 10 | Improvement | Notes |
|----------|------------------------|----------------------|-------------|-------|
| F1 | ~0.0 | 0.6704 | Large | Narrow high region found in round 10 |
| F2 | ~0.19 | 0.7248 | Large | Best at x₁ ≈ 0.70 |
| F3 | ~−0.44 | −0.0189 | Moderate | Output is always negative; improvement = less negative |
| F4 | ~0.04 | 0.2987 | Moderate | High variance; many local optima |
| F5 | ~1700 | 7493.9 | Very large | Near-boundary region [0.99, 0.99, …] |
| F6 | ~−1.3 | −0.1883 | Moderate | Noisy oracle; surrogate stuck in one region from round 8 |
| F7 | ~0.003 | 2.7968 | Large | Steady improvement via UCB exploration |
| F8 | ~5.6 | 9.9619 | Large | Consistent upward trend across all rounds |

**Diagnostics used:** Cumulative-best plots, GP mean/uncertainty surface plots (2D contours for d = 2; pairwise projection slices for d ≥ 3), acquisition surface overlays, candidate-pool scatter plots, and solver comparison plots (MyBO vs Optuna/TuRBO/DE-GP-EI) in each notebook's Section 6.

---

## Assumptions and Limitations

**Key assumptions:**

1. **Stationarity.** The GP kernel assumes constant length scale and smoothness across the domain. Violated in F4 (extreme outliers at y < −30) and possibly F6 (noise variance may vary across the space). When violated, the surrogate can be overconfident in smooth regions while missing sharp peaks or discontinuities.

2. **Smoothness.** The RBF/Matérn kernels assume the function is at least once-differentiable. Step functions or discontinuous objectives would require a different kernel family.

3. **Fixed budget awareness.** The pipeline does not dynamically adjust exploration rate based on remaining budget. A cooling schedule (decreasing ξ or κ over rounds) is applied manually per function rather than automatically.

4. **Determinism.** The pipeline assumes oracle evaluations are deterministic by default (`GP_ALPHA = 1e-6`). For F6, evidence of oracle noise led to retaining duplicate observations and raising `GP_ALPHA` to `1e-3`.

**Known limitations:**

- **Curse of dimensionality.** With 29–48 observations in 5D–8D, the GP extrapolates widely in unvisited regions. Predictions there are prior-dominated and may mislead the acquisition function.
- **Exploitation bias.** The query distribution is heavily concentrated near early high-value regions. Large portions of the domain remain unexplored, particularly for F5 (clustered at near-boundary [0.99, …]) and F6 (same point queried three consecutive rounds).
- **Isotropic kernel.** The RBF and Matérn kernels use a single scalar length scale (isotropic). An ARD (Automatic Relevance Determination) kernel would allow different length scales per dimension, which may be important for F7 and F8 where some dimensions are likely more sensitive than others.
- **No restart mechanism.** Once the surrogate becomes confident in a local region, the acquisition rarely proposes globally different candidates. A restart or space-filling diversity injection would help escape premature convergence (observed most clearly in F6).

---

## Ethical Considerations

**Transparency.** Every query and its oracle response is logged verbatim in version-controlled CSVs. Each notebook documents the kernel chosen, its hyperparameters after LML optimisation, the acquisition function and its settings, and the reasoning behind per-function configuration choices in markdown cells. A researcher with access to the repository, a matching Python environment (`requirements.txt`), and the same random seeds (set at the top of each notebook) can reproduce any round's suggested query deterministically.

**Reproducibility.** The `append_results/append_weekN_results.py` scripts record portal-returned values exactly as received, making the data trail unbroken from Week 1. External solver comparisons (Optuna, TuRBO, DE-GP-EI) are wrapped in consistent interfaces (`src/optimizers/wrappers/`) and logged alongside MyBO suggestions in each notebook.

**Real-world adaptation.** The transparency design of this pipeline directly supports adaptation to real-world ML/AI contexts:
- The datasheet and model card provide the documentation artefacts needed for model governance and audit trails.
- The explicit logging of acquisition choices and their reasoning mimics the kind of decision documentation required when deploying automated tuning systems in production.
- The identified limitations (exploitation bias, isotropic kernel, no restart) are precisely the failure modes a practitioner must monitor when applying GP-BO to expensive real-world objectives (drug discovery, hyperparameter tuning, A/B testing).

**Limitations of transparency.** The manual choice of which acquisition function to emphasise per function (e.g. switching F8 from EI to UCB in round 7) is documented in notebook markdown but not programmatically enforced or logged in a structured decision log. This is the main gap between the current transparency level and full reproducibility for an independent auditor.
