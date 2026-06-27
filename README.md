# SEBO — Sample-Efficient Bayesian Optimizer

**Author:** [Nikolas Karefyllidis, PhD](https://www.linkedin.com/in/karefyllidis/)

[![PyPI](https://img.shields.io/pypi/v/sebo?color=blue)](https://pypi.org/project/sebo/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karefyllidis/SEBO/blob/main/notebooks/demo_sklearn_hpo.ipynb)

GP-based Bayesian optimizer built from scratch — competitive with Optuna-TPE and TuRBO on sample efficiency. Drop-in for any black-box maximization problem where evaluations are expensive.

---

## Install

```bash
pip install sebo
```

```bash
pip install "sebo[benchmark]"   # + Optuna, TuRBO for comparisons
```

---

## Usage

```python
from sebo import BayesianOptimizer

optimizer = BayesianOptimizer(
    bounds=[(0.0, 1.0)] * 4,   # search space — any dimension
    output_warping="log",        # for skewed objectives (log or boxcox)
    use_ensemble=True,           # EI + PI + UCB with centroid fallback
)
optimizer.fit(X_init, y_init)   # warm-start with existing observations

for _ in range(n_rounds):
    x_next = optimizer.suggest()       # GP surrogate + ensemble acquisition
    y_next = oracle(x_next)           # your expensive function here
    optimizer.observe(x_next, y_next) # update the surrogate

best_x, best_y = optimizer.best
```

See [notebooks/demo_sklearn_hpo.ipynb](notebooks/demo_sklearn_hpo.ipynb) for a complete HPO example — no external data needed, runs on Colab.

---

## Benchmark

SEBO benchmarked against common open-source solvers on 6 synthetic black-box functions spanning four orders of magnitude in output scale. Adaptive stopping: all curves halt as soon as any solver first reaches ≥99% of the true maximum (cap: 80 evaluations).

![SEBO Benchmark Convergence](docs/sebo_benchmark_convergence.png)

*Incumbent best-y convergence. Green band = LHS warm-start. Dashed black line = true maximum. Dash-dot line = stopping point.*

Reproduce: `notebooks/sebo_benchmark.ipynb`

---

## Why SEBO

The BO loop in SEBO is the same engine used by **Optuna, SMAC, and Ax** internally — built from scratch so every design decision is explicit and auditable.

- **Hyperparameter optimisation** — replaces grid/random search; finds better configs in fewer model-training calls
- **Drug discovery & materials science** — sample-efficient search over molecular property spaces where each measurement is costly
- **Simulation optimisation** — engineering or physics simulations where one run takes minutes to hours
- **Sequential experiment design** — A/B tests, clinical dose-finding, adaptive sampling

---

## How It Works

```
fit GP → maximise acquisition → evaluate f(x*) → append (x*, y*) → repeat
```

**Kernel selection** — RBF, Matérn ν=1.5, and RBF+WhiteKernel compete at each round; the winner is chosen by log-marginal likelihood (LML) with L-BFGS-B restarts.

**Ensemble acquisition** — EI, PI, and UCB run simultaneously. If their argmaxes agree (L2 < threshold), SEBO uses EI. If they disagree, it queries the centroid — avoiding over-commitment to one strategy.

**Output warping** — for objectives spanning orders of magnitude, targets are log-transformed before GP fitting. The surrogate and acquisition operate in warped space; raw y is stored and reported.

---

## Background — NeurIPS 2020 BBO Challenge

SEBO was originally developed for the **[NeurIPS 2020 Black-Box Optimisation Challenge](https://neurips.cc/virtual/2020/protected/e_competitions.html)** format: 8 unknown objective functions (2D–8D), one evaluation per function per round, 13 rounds.

**Best observed y after 13 rounds:**

| Function | Analogy | Dim | Initial best | Final best | Improvement |
|----------|---------|-----|-------------|------------|-------------|
| F1 | Radiation detection | 2D | ~0.0 | 0.6704 | Large — narrow peak found in round 10 |
| F2 | Unknown ML model | 2D | ~0.19 | 0.7248 | Large |
| F3 | Drug discovery | 3D | ~−0.44 | −0.0032 | Large (less negative = better) |
| F4 | Warehouse logistics | 4D | ~0.04 | 0.2987 | Moderate |
| F5 | Chemical process yield | 4D | ~1700 | 7493.9 | Very large |
| F6 | Recipe formulation | 5D | ~−1.3 | −0.1402 | Large |
| F7 | Hyperparameter tuning | 6D | ~0.003 | 2.7968 | Large |
| F8 | High-dim ML model | 8D | ~5.6 | 9.9619 | Large |

### GP Surrogate Evolution — Function 3 (Drug Discovery, 3D)

![GP surrogate evolution](docs/gp_surrogate_function3_evolution.gif)

*Pairwise IDW projections of observed y across 13 rounds. Red dots = evaluations. Colour scale fixed across frames.*

---

## Project Structure

```
sebo/
├── sebo/               # pip package — public API
│   └── __init__.py     # exposes BayesianOptimizer
├── src/
│   ├── optimizers/
│   │   ├── optimizer.py                # BayesianOptimizer — suggest/observe API
│   │   ├── my_bayesian/my_gp_skopt.py # GP + ensemble acquisition (EI/PI/UCB)
│   │   └── wrappers/                   # optuna, turbo, de_gp_ei solver wrappers
│   └── utils/
│       ├── warping.py                  # log / Box-Cox output warping
│       ├── sampling_utils.py           # Sobol / LHS candidate generation
│       └── plot_utilities.py           # shared plot helpers
├── notebooks/
│   ├── demo_sklearn_hpo.ipynb          # HPO demo — start here
│   ├── sebo_benchmark.ipynb            # head-to-head benchmark
│   └── function_{1..8}_*.ipynb        # NeurIPS BBO per-function pipelines
├── docs/
│   ├── model_card.md
│   ├── datasheet.md
│   └── TECHNICAL_FOUNDATIONS.md
└── pyproject.toml
```

> Raw evaluation CSVs (`data/`, `initial_data/`) are gitignored. The demo and benchmark notebooks run without them.

---

## Stack

**Python 3.10+** · NumPy · SciPy · scikit-learn · scikit-optimize · Matplotlib

Optional: Optuna · BoTorch/TuRBO (`pip install "sebo[benchmark]"`)

---

## References

- Turner et al., PMLR 133 — [*Bayesian Optimization is Superior to Random Search for ML Hyperparameter Tuning*](https://proceedings.mlr.press/v133/turner21a.html)
- [NeurIPS 2020 Black-Box Optimisation Challenge](https://neurips.cc/virtual/2020/protected/e_competitions.html)

---

## Licence

MIT. Initial warm-start data provided by Imperial College London for educational use; redistribution permitted for non-commercial, academic purposes.
