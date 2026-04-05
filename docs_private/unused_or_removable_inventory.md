# Cleanup notes (short)

- **Committed from `docs_private/`:** this file + `20_notebooks_for_devel/` (see root `.gitignore`).
- **Never treat as cleanup:** `append_results/append_week*_results.py` (portal → `observations.csv`).
- **Already done:** removed `genetic/`, empty `bayesian_optimizer.yaml` (optional YAML still supported), fake `tests/` trees in docs, `utils` barrel exports, `hebo_solver`, `ray_tune_solver`; slim devel paths; deleted local `data/optimizer_comparison/results.csv` if regenerated; renamed `bayesian/` → `my_bayesian/`; renamed `scripts/` → **`append_results/`** (`run_pipeline.py --skip-scripts` unchanged).
- **Bench extras:** `requirements-benchmark.txt` adds Optuna (>=3.6). `hyperopt` / `botorch` are optional installs for `run_optimizers_on_data.py` (`--solvers hyperopt` / `turbo`).
- **Maybe later:** trim unused helpers in `plot_utilities.py` (search `20_notebooks_for_devel/*.ipynb` first); align `similar_projects/` vs `30_similar_projects/` in links; drop duplicate `evolutionary_methods_guide.html` if `.md` is canonical.
- **Regenerable / ignorable:** `data/optimizer_comparison/results.csv`; `initial_data/`, `data/problems/`, etc. are gitignored by design.
- Before deleting anything: `grep -r <basename>` from repo root.
