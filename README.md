# Black-Box Optimisation with Bayesian Optimisation
### NeurIPS 2020 Black-Box Optimisation Competition — ICL PCMLAI Capstone

**Author:** [Nikolas Karefyllidis, PhD](https://www.linkedin.com/in/karefyllidis/)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Last Updated](https://img.shields.io/badge/Last%20Updated-April%202026-lightgrey)
![Rounds](https://img.shields.io/badge/Rounds-10-orange)
![Functions](https://img.shields.io/badge/Functions-8-purple)

This project follows the format of the **[NeurIPS 2020 Black-Box Optimisation Challenge](https://github.com/rdturnermtl/bbo_challenge_starter_kit)**: sequential optimisation of 8 unknown objective functions under a strict per-round evaluation budget, using the same suggest–observe API and Bayesmark evaluation framework as the competition. One query per function per week across 10 rounds.

**Bayesian Optimisation (BO)** was chosen as the core strategy because it is specifically designed for exactly this setting — expensive black-box objectives with no access to gradients or closed-form expressions, where every evaluation counts. BO builds a probabilistic surrogate (GP) over the unknown function and uses it to decide where to query next, trading off exploration of uncertain regions against exploitation of known good ones. This makes it far more sample-efficient than random search or grid search, which is critical when the evaluation budget is as small as 10–20 points per function. The pipeline features automatic kernel selection via log-marginal likelihood (RBF, Matérn, RBF+WhiteKernel), MLE-based kernel hyperparameter optimisation, and an ensemble acquisition strategy (EI + PI + UCB) that switches between exploitation and exploration based on inter-acquisition agreement.

**[Datasheet](docs/datasheet.md)** · **[Model Card](docs/model_card.md)**

---

## The 8 Functions

| # | Dim | Analogy | Description |
|---|-----|---------|-------------|
| 1 | 2D | Radiation detection | Sparse signal; output is near-zero across most of the domain with a narrow high-value region. |
| 2 | 2D | Mystery ML model | Noisy surface with multiple local peaks; requires careful balance of exploration and exploitation. |
| 3 | 3D | Drug discovery | Smooth but always-negative output; optimisation is finding the least-negative value. |
| 4 | 4D | Warehouse logistics | Many local optima and occasional extreme outliers; highly non-convex landscape. |
| 5 | 4D | Chemical process yield | Unimodal; output spans several orders of magnitude, concentrated near domain boundaries. |
| 6 | 5D | Recipe optimisation | Negative outputs throughout; mild oracle noise observed (same input returned different y across rounds). |
| 7 | 6D | Hyperparameter tuning | Simulates tuning a model (learning rate, regularisation, layers); smooth but sparse in 6D. |
| 8 | 8D | High-dimensional ML model | Highest-dimensional function; steady monotonic improvement observed across all 10 rounds. |

Domain: **[0, 1]^d** for all functions. Higher y is always better; F3 and F6 outputs are negative (e.g. −0.02 > −0.44).

### Example: GP Surrogate — Function 3 (Drug Discovery, 3D)

![GP surrogate pairwise projections for Function 3](docs/gp_surrogate_function3.png)

*Pairwise IDW-interpolated projections of the GP surrogate after 25 observations. Red dots are evaluated query points (numbered by round). The warm (light) regions indicate higher predicted y; the surrogate identifies a promising cluster near x₁ ≈ 0.15–0.20 in the x₁–x₂ plane.*

---

## Background: Bayesian Optimisation

**Bayesian optimisation (BO)** is a sample-efficient, sequential strategy for expensive black-box objectives. There is no formula for f(x) — only point evaluations. BO maintains a **Gaussian Process (GP) surrogate** that provides:

- **μ(x)** — predictive mean (exploitation signal)
- **σ(x)** — predictive uncertainty (exploration signal)

An **acquisition function** (EI, UCB, PI) combines these to select the next query. The loop is:

```
fit GP → maximise acquisition → evaluate f(x*) → append (x*, y*) → repeat
```

GPs are data-efficient and naturally quantify uncertainty, making them well-suited to low-budget, moderate-dimensional problems (d = 2–8 here). Three kernels are compared at each round — **RBF**, **Matérn (ν=1.5)**, and **RBF+WhiteKernel** — with the best selected automatically by log-marginal likelihood (LML).

---

## Pipeline

Each notebook follows a fixed 8-section structure:

1. **Setup** — imports, load observations from `data/problems/function_N/observations.csv`
2. **Parameters** — kernel, acquisition coefficients (ξ, κ), warping, candidate sampling, ensemble mode
3. **Visualise** — GP surrogate surfaces; 2D contour (d=2), pairwise projections (d≥3)
4. **Acquisition** — EI/PI/UCB over a Sobol/LHS candidate set; LML kernel selection; duplicate masking
5. **Select query** — ensemble centroid (if EI/PI/UCB disagree) or EI argmax; proximity guard
6. **MyBO vs open source** — compare with Optuna-TPE, Optuna-GP, TuRBO, DE-GP-EI
7. **Append feedback** — after portal returns (x, y), set `IF_APPEND_DATA = True`
8. **Save submission** — write chosen vector to `data/submissions/function_N/`

**Key configuration flags** (cell 2 of each notebook):

| Flag | Default | Effect |
|------|---------|--------|
| `GP_KERNEL` | `None` (LML auto) | Force kernel: `'RBF'`, `'Matern'`, `'RBF + WhiteKernel'` |
| `OUTPUT_WARPING` | `None` | `'log'` for F1/F5/F7 (skewed y); `'boxcox'` optional |
| `USE_ENSEMBLE` | `True` | Centroid of EI/PI/UCB when they disagree by > 0.15 L2 |
| `MIN_DIST_THRESHOLD` | `0.05` | Mask candidates within this L2 distance of any prior obs |
| `BOUNDARY_MARGIN` | `0.05` (d≤3), `0` (d≥4) | Mask near-boundary candidates (curse of dimensionality) |
| `NEXT_QUERY_SOLUTION` | `'MyBO'` | Which solver's vector to save: MyBO / Optuna-TPE / TuRBO / … |

---

## Quick Start

```bash
pip install -r requirements.txt
```

**Run the full pipeline** (append latest results + execute all notebooks + print portal strings):
```bash
python run_pipeline.py
```

Options:
- `--skip-notebooks` — print previously saved portal strings without re-running notebooks
- `--skip-scripts` — skip `append_results/*.py`

**Benchmark solvers** against the accumulated observations:
```bash
pip install -r requirements-benchmark.txt
python append_results/run_optimizers_on_data.py --solvers my_bo optuna turbo de_gp_ei
```

**Append a new week's portal results** (idempotent, version-controlled):
```bash
python append_results/append_week10_results.py
```

---

## Project Structure

```
black-box-optimization/
│
├── initial_data/                      # Read-only warm-start data (DO NOT MODIFY)
│   └── function_{1..8}/               # initial_inputs.npy, initial_outputs.npy
│
├── data/
│   ├── problems/function_{1..8}/      # observations.csv — all (x, y) pairs appended each round
│   ├── submissions/function_{1..8}/   # next_input.npy, next_input_portal.txt
│   └── results/                       # Exported plots (when IF_EXPORT_PLOT = True)
│
├── notebooks/
│   ├── function_1_Radiation-Detection.ipynb
│   ├── function_2_Mystery-ML-Model.ipynb
│   ├── function_3_Drug-Discovery.ipynb
│   ├── function_4_Warehouse-Logistics.ipynb
│   ├── function_5_Chemical-Process-Yield.ipynb
│   ├── function_6_Recipe-Optimization.ipynb
│   ├── function_7_Hyperparameter-Tuning.ipynb
│   └── function_8_High-dimensional-ML-Model.ipynb
│
├── src/
│   ├── optimizers/
│   │   ├── my_bayesian/my_gp_skopt.py          # GP+skopt BO (EI/PI/UCB/Thompson/Entropy Search)
│   │   └── wrappers/                            # optuna_solver, turbo_solver, de_gp_ei_solver, hyperopt_solver
│   └── utils/
│       ├── load_challenge_data.py               # load_function_data(N); read-only guard
│       ├── warping.py                            # apply_output_warping / inverse (log, boxcox)
│       ├── plot_utilities.py                     # Shared plot styling and export helpers
│       └── sampling_utils.py                     # sample_candidates() wrapper
│
├── append_results/
│   ├── append_week{1..10}_results.py            # Append portal (x, y) to observations.csv
│   └── run_optimizers_on_data.py                # Benchmark solvers on accumulated data
│
├── configs/
│   ├── optuna_optimizer.yaml
│   ├── turbo_optimizer.yaml
│   ├── de_gp_ei_optimizer.yaml
│   └── hyperopt_optimizer.yaml
│
├── docs/
│   ├── datasheet.md                             # Dataset documentation (composition, collection, uses)
│   ├── model_card.md                            # Model documentation (architecture, performance, limitations)
│   ├── TECHNICAL_FOUNDATIONS.md                 # Key papers, library choices, BO theory
│   ├── project_roadmap.md                       # Notebook workflow, planned components, future work
│   └── Capstone_Project_FAQs.md
│
├── run_pipeline.py
├── requirements.txt
├── requirements-benchmark.txt
└── README.md
```

---

## Documentation

| File | Contents |
|------|----------|
| [docs/datasheet.md](docs/datasheet.md) | Dataset datasheet — motivation, composition, collection, preprocessing, uses, licence |
| [docs/model_card.md](docs/model_card.md) | Model card — architecture, 10-round performance summary, assumptions, limitations, ethical considerations |
| [docs/TECHNICAL_FOUNDATIONS.md](docs/TECHNICAL_FOUNDATIONS.md) | BO theory, kernel choices, acquisition functions, key papers |
| [docs/project_roadmap.md](docs/project_roadmap.md) | Notebook workflow, `run_pipeline.py` usage, future work |
| [docs/Capstone_Project_FAQs.md](docs/Capstone_Project_FAQs.md) | Portal submission format, data loading, allowed methods |

---

## References

- **NeurIPS 2020 BBO Challenge** — [starter kit](https://github.com/rdturnermtl/bbo_challenge_starter_kit); top entries: HEBO (Huawei/Noah's Ark), ensemble BO (Nvidia), GP+SVM (JetBrains)
- Rasmussen & Williams, *Gaussian Processes for Machine Learning* (2006)
- Jones et al., *Efficient Global Optimization of Expensive Black-Box Functions*, J. Global Optim. (1998)
- See `docs/TECHNICAL_FOUNDATIONS.md` for full bibliography and library justifications
