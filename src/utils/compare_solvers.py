"""
Shared helpers for comparing MyBO with external wrappers (Optuna, TuRBO, DE-GP-EI).

Used by ``append_results/run_optimizers_on_data.py`` and by
``notebooks/function_1_Radiation-Detection.ipynb`` (Section 6).
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from src.utils.warping import apply_output_warping

# Output warping for ``run_optimizers_on_data.py`` only (same transform as the GP sees in-notebook).
# MyBO hyperparameters live in each ``function_*.ipynb`` (``OUTPUT_WARPING``); keep this dict in sync
# when you change warping there, or batch vs notebook comparisons will disagree.
_OUTPUT_WARPING_BY_FUNCTION_ID: dict[int, str | None] = {
    1: "log",  # function_1_Radiation-Detection.ipynb OUTPUT_WARPING
}

# Plot style: marker, matplotlib color (Section 6 summary)
SOLVER_STYLE: dict[str, tuple[str, str]] = {
    "MyBO": ("s", "red"),
    "Optuna-TPE": ("o", "blue"),
    "Optuna-GP": ("o", "deepskyblue"),
    "TuRBO": ("^", "green"),
    "DE-GP-EI": ("D", "magenta"),
}
FALLBACK_STYLES: list[tuple[str, str]] = [
    ("P", "orange"),
    ("*", "cyan"),
    ("h", "lime"),
]

_DEFAULT_EXTERNAL_SEEDS = {
    "optuna_tpe": 42,
    "optuna_gp": 46,
    "turbo": 43,
    "de": 44,
}


def output_warping_mode(function_id: int | None) -> str | None:
    """Warping for batch scripts; values mirror each notebook's ``OUTPUT_WARPING``."""
    if function_id is None:
        return None
    mode = _OUTPUT_WARPING_BY_FUNCTION_ID.get(function_id)
    if mode is None or str(mode).strip().lower() in ("none", ""):
        return None
    return str(mode).strip()


def prepare_y_for_surrogates(
    y_raw: np.ndarray,
    function_id: int,
) -> tuple[np.ndarray, str | None]:
    """
    Transform ``y`` like the corresponding notebook before GP fitting.

    Pass the returned vector to all ``suggest(X, y, ...)`` wrappers in one run,
    and call MyBO with ``output_warping='none'`` so warping is not applied twice.
    """
    y_raw = np.asarray(y_raw, dtype=np.float64).ravel()
    mode = output_warping_mode(function_id)
    if not mode:
        return y_raw.copy(), None
    y_fit, _, _ = apply_output_warping(y_raw, mode=mode)
    return y_fit, mode


def collect_external_solver_suggestions(
    X: np.ndarray,
    y: np.ndarray,
    bounds: list[tuple[float, float]],
    function_id: int,
    *,
    seed_overrides: dict[str, int] | None = None,
    quiet: bool = False,
) -> list[tuple[str, np.ndarray]]:
    """
    Call Optuna (TPE + GP), TuRBO, and DE-GP-EI; return successful (name, x_next) pairs.

    ``y`` must already be in surrogate space (e.g. warped like Section 4/6 in the notebook).
    """
    seeds = {**_DEFAULT_EXTERNAL_SEEDS, **(seed_overrides or {})}
    specs: list[tuple[str, str, dict[str, Any]]] = [
        (
            "Optuna-TPE",
            "src.optimizers.wrappers.optuna_solver",
            {"bounds": bounds, "function_id": function_id, "seed": seeds["optuna_tpe"]},
        ),
        (
            "Optuna-GP",
            "src.optimizers.wrappers.optuna_solver",
            {
                "bounds": bounds,
                "function_id": function_id,
                "seed": seeds["optuna_gp"],
                "sampler": "gp",
            },
        ),
        (
            "TuRBO",
            "src.optimizers.wrappers.turbo_solver",
            {"bounds": bounds, "function_id": function_id, "seed": seeds["turbo"]},
        ),
        (
            "DE-GP-EI",
            "src.optimizers.wrappers.de_gp_ei_solver",
            {"bounds": bounds, "function_id": function_id, "seed": seeds["de"]},
        ),
    ]
    out: list[tuple[str, np.ndarray]] = []
    for name, mod_path, kwargs in specs:
        try:
            mod = importlib.import_module(mod_path)
            x_pt = np.asarray(mod.suggest(X, y, **kwargs)).ravel()
            out.append((name, x_pt))
        except Exception as e:
            if not quiet:
                print(f"{name}: {e}")
    return out


def try_suggest_import(
    name: str,
    mod_path: str,
    X: np.ndarray,
    y: np.ndarray,
    kwargs: dict[str, Any],
) -> tuple[str, np.ndarray] | None:
    """Import ``mod_path`` and call ``suggest(X, y, **kwargs)``; print and return None on failure."""
    try:
        mod = importlib.import_module(mod_path)
        x_pt = np.asarray(mod.suggest(X, y, **kwargs)).ravel()
        return (name, x_pt)
    except Exception as e:
        print(f"{name}: {e}")
        return None
