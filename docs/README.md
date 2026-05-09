# Documentation index

| Path | Role |
|------|------|
| [README.md](../README.md) | Main entry: workflow, structure, methods, references; **challenge complete** (final data through Week 13) |
| [model_card.md](model_card.md) | Model architecture, **per-function acquisition hyperparameters**, performance |
| [project_roadmap.md](project_roadmap.md) | Layout, notebook steps, planned pieces |
| [Capstone_Project_FAQs.md](Capstone_Project_FAQs.md) | Capstone-specific data, submission, Section 6 |
| [TECHNICAL_FOUNDATIONS.md](TECHNICAL_FOUNDATIONS.md) | Papers, library choices, “where to read more” |

**Dependencies:** root `requirements.txt` (notebooks + `run_pipeline.py`); `requirements-benchmark.txt` for Optuna and optional HPO extras.

**Optimizer configs (per-function YAML):** `configs/optuna_optimizer.yaml`, `de_gp_ei_optimizer.yaml`, `turbo_optimizer.yaml`, `hyperopt_optimizer.yaml`; optional `bayesian_optimizer.yaml` for MyBO (`my_gp_skopt`).

Private notes live under `docs_private/` (mostly gitignored). Short cleanup list: `docs_private/unused_or_removable_inventory.md`.
