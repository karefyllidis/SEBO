# Inventory: unused, redundant, or removable items

Living note for this repo. **Scan date:** based on static analysis of the workspace (imports, `glob`, grep). Re-check before deleting anything.

**Git:** The repo’s `.gitignore` whitelists this path with `!docs_private/unused_or_removable_inventory.md` so it can be committed while the rest of `docs_private/` stays ignored by default.

---

## 1. Safe structural cleanup (low risk if you accept trade-offs)

| Item | Why it looks unused | If you remove it |
|------|---------------------|------------------|
| **`src/optimizers/genetic/`** | `__init__.py` is only a docstring plus a **fully commented-out** block that imports non-existent `genetic/acquisition_functions.py`. Nothing imports `src.optimizers.genetic`. | Delete the folder, or replace with a one-line README pointing to `ga_solver` (DE-GP-EI). Removing is clean-up only. |
| **`configs/bayesian_optimizer.yaml`** | File contains **comments only**—no `default:` or `function_N:` keys. `my_gp_skopt.suggest()` still **opens** this path; behaviour falls back to code defaults. | Either **add real YAML** (hyperparameters per function) or **delete the file** and adjust `my_gp_skopt.load_*` to treat missing YAML as `{}` without expecting the file to exist (verify loader already tolerates missing file). |
| **`tests/` tree in README / roadmap** | No `tests/**/*.py` exists in the project layout—suite is **not implemented** yet. | Remove the `tests/` lines from README `project_roadmap` trees **or** add a minimal `tests/test_load_challenge_data.py`. Keeping the folder in docs as “planned” is fine but misleading if the directory does not exist. |
| **`src/utils/__init__.py` barrel re-exports** | No `from src.utils import …` usage found in `.py` or tracked notebooks; callers import **`src.utils.load_challenge_data`**, **`warping`**, **`sampling_utils`**, **`plot_utilities`** directly. | You could slim `__init__.py` to an empty docstring or delete re-exports; optional style/refactor. |

---

## 2. Large modules mostly unused by current function notebooks

| Item | Usage | Note |
|------|--------|------|
| **`src/utils/plot_utilities.py`** | Tracked **function_1** / **function_2** notebooks import **`prepare_surface_for_plot`** only (plus internal use inside the same module). Helpers like `plot_gp_1d`, `plot_bo_iteration_1d`, `plot_2d_bo_state`, `plot_convergence`, `plot_parallel_coordinates` appear **unused** outside this file. | May still be used by **`function_0_devel.ipynb`** (partially tracked) or **`docs_private/20_notebooks_for_devel/*.ipynb`**. Before deleting large chunks, search those notebooks. |
| **`src/optimizers/bayesian/acquisition_functions.py`** | Notebooks use **skopt** (`gaussian_ei`, etc.). This module is used by **`ga_solver`** (EI) and **`bayesian/__init__.py`** exports—not redundant. | Keep unless you inlined EI only inside `ga_solver`. |

---

## 3. Code used only by the benchmark CLI (not Section 6 notebooks)

These are **used** by `scripts/run_optimizers_on_data.py` but **not** referenced inside **function_1–8** notebooks:

- **`src/optimizers/bayesian/my_gp_skopt.py`** (`--solvers my_bo`)
- **`hebo_solver.py`**, **`hyperopt_solver.py`**, **`ray_tune_solver.py`** (optional extras)

If you **only** care about the weekly notebook → portal workflow, you could drop benchmark-only pieces—but that shrinks comparison tooling. **`requirements-benchmark.txt`** lists only `optuna`; HEBO/hyperopt/ray are “install if you use that solver.”

---

## 4. Scripts: historical append files (optional archiving)

- **`scripts/append_week1_results.py` … `append_week9_results.py`** — Each week’s portal row. All are **idempotent**; `run_all.py` runs **every** `scripts/*.py` (except post-notebook allowlist).  
- **Not “unused”** for reproducibility. **Could** be archived (e.g. `scripts/archive/weeks/`) after the course ends to reduce clutter; then adjust `run_all.py` or move archived scripts out of `scripts/` so they are not run every time.

---

## 5. Redundant or generated artifacts

| Item | Note |
|------|------|
| **`data/optimizer_comparison/results.csv`** | Typical **output** of `run_optimizers_on_data.py --output …`. Repo `.gitignore` includes `*.csv`, so it often **won’t be committed**; safe to delete locally and regenerate. |
| **`docs_private/40_notes_and_references/10_evolutionary_methods/evolutionary_methods_guide.html`** | Likely a **duplicate export** of the `.md` beside it. If the `.md` is canonical, the `.html` can be removed to avoid drift—unless you rely on HTML for offline viewing. |

---

## 6. Docs / paths worth reconciling (not “unused,” but inconsistent)

| Topic | Detail |
|-------|--------|
| **`function_0_devel.ipynb` location** | `.gitignore` expects **`docs_private/notebooks/function_0_devel.ipynb`**, while other development notebooks live under **`docs_private/20_notebooks_for_devel/`**. Confirm one layout and update **README / Capstone FAQs** paths accordingly. |
| **Similar-project notes path** | This tree has used both `docs_private/similar_projects/` and **`docs_private/30_similar_projects/`** in docs vs disk. **Search** for `notes_from_bbo_starter_kit.md` and align **TODO / project_log** links to the folder that actually exists. |
| **`configs/problems/`** | Roadmap says removed; **no loader** in code—safe to ignore until you add a YAML-driven problem registry. |

---

## 7. `.gitignore`–only or external folders (cannot audit here)

- **`initial_data/`**, **`data/problems/`**, **`data/submissions/`**, **`data/results/`**, **`submission-template/`** — Ignored or absent in a bare clone; not “unused,” just not in version control.
- **`docs_private/*`** (except the one notebook exception) — Large private tree; this inventory cannot assert every file is read by the codebase.

---

## 8. Summary: highest-value tidy-ups

1. Remove or document **`src/optimizers/genetic/`** (dead stub).  
2. Either populate **`configs/bayesian_optimizer.yaml`** or stop requiring an empty file.  
3. Confirm whether **`plot_utilities`** 1D/2D helpers are needed; trim or move to a `notebooks` util if only `prepare_surface_for_plot` stays.  
4. Align **devel notebook paths** and **gitignore** with the real `docs_private` folder names.  
5. Optionally archive old **`append_week*.py`** after the challenge, or exclude them from `run_all.py`'s script phase.

When in doubt, **`grep -r` from repo root** for the basename before deleting.
