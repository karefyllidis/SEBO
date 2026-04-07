# Cleanup notes (short)

- **Committed from `docs_private/`:** this file + `20_notebooks_for_devel/` (see root `.gitignore`).
- **Never treat as cleanup:** `append_results/append_week*_results.py` (portal → `observations.csv`).
- **Already done:** removed `genetic/`, empty `bayesian_optimizer.yaml` (optional YAML still supported), fake `tests/` trees in docs, `utils` barrel exports, `hebo_solver`, `ray_tune_solver`; slim devel paths; deleted local `data/optimizer_comparison/results.csv` if regenerated; renamed `bayesian/` → `my_bayesian/`; renamed `scripts/` → **`append_results/`** (`run_pipeline.py --skip-scripts` unchanged); `ga_optimizer.yaml` → **`de_gp_ei_optimizer.yaml`**; **`docs_private/`** layout uses **`00_` / `10_` / `20_` / `30_` / `40_` prefixes** (e.g. **`30_similar_projects/`**, **`10_canvas_submissions_archive/`**) — update links if you remove or rename a folder.
- **Bench extras:** `requirements-benchmark.txt` adds Optuna (>=3.6). `hyperopt` / `botorch` are optional installs for `run_optimizers_on_data.py` (`--solvers hyperopt` / `turbo`).
- **Maybe later:** trim unused helpers in `plot_utilities.py` (search `20_notebooks_for_devel/*.ipynb` first); drop duplicate `evolutionary_methods_guide.html` if `.md` is canonical.
- **Regenerable / ignorable:** `data/optimizer_comparison/results.csv`; `initial_data/`, `data/problems/`, etc. are gitignored by design.
- Before deleting anything: `grep -r <basename>` from repo root.
