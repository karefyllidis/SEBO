# Project roadmap: planned structure

Components listed here are in use or planned. Add folders back when you need them.

## Current project structure (simplified)

```
black-box-optimization/
├── initial_data/                 # Raw challenge data (DO NOT MODIFY)
│   ├── function_1/ … function_8/
│
├── src/
│   ├── optimizers/
│   │   ├── my_bayesian/          # acquisition_functions.py (UCB, EI, PI, Thompson, Entropy Search)
│   │   └── wrappers/             # optuna_solver.py (TPE / GPSampler), turbo_solver.py, de_gp_ei_solver.py (DE-GP-EI) — suggest(X,y,bounds) for Section 6
│   └── utils/
│       ├── load_challenge_data.py # load_function_data(N), assert_not_under_initial_data (blocks writes under initial_data only)
│       ├── plot_utilities.py     # style_axis, add_colorbar, style_legend, prepare_surface_for_plot, style_axis_3d; plot_2d_bo_state, plot_2d_function, plot_convergence, plot_gp_1d, plot_acquisition_1d, plot_bo_iteration_1d, plot_parallel_coordinates; DEFAULT_FONT_SIZE_*, DEFAULT_EXPORT_*
│       ├── warping.py            # apply_output_warping(y, mode=None|"log"|"boxcox"); inverse_output_warping — HEBO-inspired y transform for GP
│       └── sampling_utils.py    # sample_candidates() wrapper (F1 uses this; F2/F3+ use skopt.sampler directly)
│
├── data/
│   ├── problems/                 # Local appended data: only observations.csv per function (no .npy under data/)
│   ├── submissions/              # Next input to submit (function_1/next_input.npy, next_input_portal.txt)
│   (data/results/function_N/)    # Exported plots per function (observations+contour, 3D surface, GP kernels, Section 6, …)
│
├── notebooks/
│   ├── function_1_Radiation-Detection.ipynb      # F1 (2D): full options; Section 6 MyBO vs Optuna-TPE / Optuna-GP / TuRBO / DE-GP-EI, best obs blue "+"
│   ├── function_2_Mystery-ML-Model.ipynb         # F2 (2D): d=2 template — 3 kernels, ensemble; Section 6 solver comparison
│   ├── function_3_Drug-Discovery.ipynb           # F3 (3D): pairwise projections, GP slices; Section 6 solver comparison
│   ├── function_4_Warehouse-Logistics.ipynb      # F4 (4D): 6 pairwise plots; Section 6 solver comparison, Section 7 append feedback
│   ├── function_5_Chemical-Process-Yield.ipynb   # F5 (4D): same as F4
│   ├── function_6_Recipe-Optimization.ipynb      # F6 (5D): Section 6 solver comparison, Section 7 append
│   ├── function_7_Hyperparameter-Tuning.ipynb    # F7 (6D): Section 6 solver comparison, Section 7 append
│   └── function_8_High-dimensional-ML-Model.ipynb # F8 (8D): Section 6 solver comparison, Section 7 append
│
├── run_pipeline.py                   # Runs append_results/*.py + all 8 notebooks, prints portal strings; --skip-notebooks / --skip-scripts
├── append_results/               # append_week{N}_results.py (portal → observations.csv); run_optimizers_on_data.py (bench)
├── configs/
│   ├── optuna_optimizer.yaml     # Per-function Optuna defaults (notebook Section 6 may pass explicit sampler/seed)
│   ├── de_gp_ei_optimizer.yaml   # DE-GP-EI (scipy DE on GP-EI)
│   ├── hyperopt_optimizer.yaml   # Hyperopt TPE
│   ├── turbo_optimizer.yaml      # TuRBO (BoTorch)
│   └── problems/                 # (optional; see docs_private/private_notes.md)
│
├── docs/
│   ├── project_roadmap.md        # (this file)
│   ├── Capstone_Project_FAQs.md
│   ├── TECHNICAL_FOUNDATIONS.md  # Justification, key papers, library choices
│   └── …
│
├── docs_private/                 # Private notes (gitignored; structure not listed in open repo)
├── requirements.txt
├── requirements-benchmark.txt    # Optuna (bench / notebook Section 6); optional hyperopt, botorch
├── .gitignore
└── README.md
```

**Configs:** Optional `configs/bayesian_optimizer.yaml` supplies per-function MyBO settings via `my_gp_skopt.load_mybo_config()`. If the file is absent, `suggest()` uses code defaults (see README).

**Removed for now (add back when needed):**
- `configs/algorithms/`, `configs/experiments/` — algorithm/experiment configs
- `src/optimizers/genetic/` — removed (was a dead stub); use `wrappers/de_gp_ei_solver.py` (DE-GP-EI) for differential evolution on GP-EI
- `tests/test_objectives/` — we have no src/objective
- `notebooks/weekly_review/` — weekly notes
- `src/objective/`, `src/experiments/` — see private notes (e.g. in docs_private/)

## Notebook workflow (F2/F4 template — all notebooks adapted)

1. **Setup and load data** — Imports (GP, skopt acquisition/sampler), repo root, load from local CSV or `initial_data`, flags.
2. **Parameters** — Kernel choice (`GP_KERNEL = None` → LML auto-select, or manual), `OPTIMIZE_KERNEL`, kernel bounds (constant scale, length scale, white noise `(1e-12, 1e1)`), acquisition coefficients (`XI_EI_PI`, `KAPPA_UCB`) and **`AGREE_THRESHOLD`** (ensemble agree / centroid rule), candidate sampling (`n_cand` as power of 2), ensemble vs solo mode (`USE_ENSEMBLE`, `SOLO_STRATEGY`), `MIN_DIST_THRESHOLD` (min L2 distance from any observation; masks acquisition and drives proximity check/fallback), `BOUNDARY_MARGIN` (optional; mask candidates near edges [0,margin] or [1−margin,1]; 0.05 for low-d F1–F3, 0 for F4–F8). Late-round exploit-leaning ξ/κ and updated thresholds are documented in the root **README** and **docs/model_card.md** (tables mirror each notebook’s Parameters cell).
3. **Visualize** — Observations scatter, IDW contour, convergence plot. d=2: 2D contour + 3D surface. d≥3: 2D pairwise projections + IDW with per-row colorbars; uses coarser `n_grid_viz` for fast rendering.
4. **GP surrogate** — Fit 3 kernels (RBF, Matérn, RBF+WhiteKernel) with configurable bounds; select best by LML. 3×2 grid (mean + std). d≥3: 2D slices at median of held-out dimensions.
5. **Acquisition** — EI/PI/UCB computed for all kernels via `skopt.acquisition` on `n_cand` Sobol/LHS candidates. This cell computes `next_x_high_dist` (farthest candidate from observations) and uses it as fallback when the acquisition argmax lies in the masked set. Candidates within `MIN_DIST_THRESHOLD` of any observation are masked (EI/PI → −∞, LCB → +∞); when `BOUNDARY_MARGIN` > 0, candidates with any coordinate in [0, margin] or [1−margin, 1] are also masked (low-d only; F4–F8 use 0). If `BOUNDARY_MARGIN` is undefined (e.g. parameters cell not run), the cell sets it to 0. Ensemble logic (agree → EI argmax, disagree → centroid) or solo. Baselines: exploit + explore (no high-distance in F2–F8).
6. **Select & illustrate** — Final plot: d=2: 1×2 (mean + std); d≥3: pairwise GP slices with acquisition markers; `tight_layout(rect=[0,0,1,0.96])` + `suptitle(..., y=0.98)` avoids title overlap.
7. **Section 6: MyBO vs Open Source** — Compare this notebook’s next_x with Optuna, TuRBO, and DE-GP-EI (wrappers in `src/optimizers/wrappers/`; implementation `de_gp_ei_solver`). Observations and solver suggestions are plotted; the best observation is overlaid as a blue “+” on all panels.
8. **Export** — Append new observation and/or save next_x (cells after Section 6).

**F1** retains the original full-options layout (all acquisition functions, high-distance baseline, Thompson/Entropy). F1 uses `MIN_DIST_THRESHOLD = 0.01` and replaces the proposed query with the high-distance fallback only for true duplicates (dist &lt; 1e-3), so proposals can refine near the best point. All F1 plot titles show `warping: {WARP_LABEL}`; IDW contour uses symlog only when warping is set. **F1 visualization:** Observation scatter colour scale is built from the **observation** y range (not the IDW grid); left-panel points have grey edges. **Section 6 (all notebooks):** MyBO vs **Optuna-TPE**, **Optuna-GP**, TuRBO, DE-GP-EI; F1 left = observations by y, right = IDW contour; F3–F8 use pairwise panels; best observation is a blue “+”; solver suggestions overlaid with name-keyed markers (`_SOLVER_STYLE`). All F3–F8 notebooks are fully adapted with dimension-specific pair counts, per-row colorbars, and optimised rendering.

Notebook workflow above is the canonical adaptation checklist (F2 / F4 templates); see also `docs_private/40_notes_and_references/README.md`.

**run_pipeline.py** — Run from project root. Runs any `append_results/*.py` (sorted; includes `append_week*_results.py` after each portal round), executes all 8 notebooks when `nbconvert` and `ipykernel` are installed (see `requirements.txt`; generates submissions), then prints full portal strings for functions 1–8 and file paths. Use `--skip-notebooks` to skip notebook execution (show saved summary only); `--skip-scripts` to skip `append_results/*.py`.

Write safety: `assert_not_under_initial_data(path, project_root)` only forbids writes under `project_root/initial_data/`; `data/results/`, `data/submissions/`, `data/problems/` are allowed.

## Planned components (add as you go)

### `src/optimizers/my_bayesian/`

- **`acquisition_functions.py`** — skopt alternative (EI, PI, UCB, …); **notebooks use skopt** for acquisition; used by `de_gp_ei_solver` for EI.
- **`my_gp_skopt.py`** — MyBO `suggest()`; optional `configs/bayesian_optimizer.yaml`.

### `src/utils/`

- **`load_challenge_data.py`** — CSV / `initial_data` loads; write guard.
- **`plot_utilities.py`**, **`warping.py`**, **`sampling_utils.py`** — plots, y-warping, F1 sampling helpers (see README / notebooks).

### `configs/problems/`

- Optional YAML registry — no loader yet.

### `tests/`

- Not in repo; add when you automate checks.

### `docs/` and `docs_private/`

- **docs/** — roadmap, FAQs, foundations (see README table).
- **docs_private/** — log, TODO, guides; whitelisted: `20_notebooks_for_devel/`, `unused_or_removable_inventory.md` (short cleanup bullets). Rest ignored.

## Future work and later considerations

Ideas that are not required for the current capstone runs but worth revisiting if budget, time, or evaluation rigor increases.

- **Space-filling warm-up:** When you control the first queries (beyond fixed challenge `initial_data`), spend the first **k** evaluations on **LHS**, **Sobol**, or a small **stratified grid** (e.g. k ≈ 2d–5d or a fixed k like 5 in 2D) before leaning on BO/TPE. Improves global coverage when **n** is small and the surrogate is sensitive to early geometry.
- **Stronger baseline evaluation:** Average **multi-seed** suggestions or **best-so-far vs evaluation index** curves for Section 6 methods (`append_results/run_optimizers_on_data.py --seeds …`); single-seed snapshots are high-variance.
- **Fairer wrapper vs notebook BO:** Optionally **restrict** or **snap** continuous acquisition optimizers to the same **Sobol/LHS candidate set** (or grid) the notebook uses, so comparisons isolate the surrogate rule rather than the search space parameterization.
- **TuRBO beyond one-shot `suggest()`:** A **persistent** trust-region state (expand/shrink length) over a self-managed loop matches the usual TuRBO-1 narrative; the current wrapper is intentionally stateless per call.
- **Shared Section 6 code:** Function 1 imports `src.utils.compare_solvers`; other notebooks still duplicate the Section 6 helper block—could migrate them to the same helper when convenient.
- **Batch vs notebook warping:** If you add or change `OUTPUT_WARPING` on more functions, extend `src.utils.compare_solvers._OUTPUT_WARPING_BY_FUNCTION_ID` so CLI benchmarks stay aligned.
