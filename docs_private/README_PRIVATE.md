# Private docs (`docs_private/`)

This folder is **gitignored by default** (see root `.gitignore`). Only `20_notebooks_for_devel/` and `unused_or_removable_inventory.md` are whitelisted for optional sharing.

**Keep local (do not publish on a public GitHub remote):**

- Oracle evaluation trails: `data/problems/function_*/observations.csv` and portal submissions under `data/submissions/` — also gitignored at repo root.
- Warm-start arrays: `initial_data/` — gitignored; provided by the course, read-only locally.
- Course reflections, discussion drafts, and similar notes under paths such as `40_notes_and_references/`.

**Public repo** carries code, configs, append scripts, documentation (`docs/`), and **aggregated** transparency artefacts (datasheet, model card, README). Summary statistics in the model card do not expose row-level oracle data.
