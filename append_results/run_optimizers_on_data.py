#!/usr/bin/env python3
"""
Run multiple BO solvers (MyBO, Optuna, etc.) on your challenge data and compare next suggestions.

Loads data from data/problems/function_N/observations.csv (or initial_data if no CSV).
For each function 1..8 and each solver, computes the suggested next point and prints/writes results.

``y`` is transformed to match each notebook's ``OUTPUT_WARPING`` via
``src.utils.compare_solvers._OUTPUT_WARPING_BY_FUNCTION_ID`` before being passed to every
surrogate; MyBO is called with ``output_warping='none'`` so warping is not applied twice.

Usage (from project root):
  python append_results/run_optimizers_on_data.py
  python append_results/run_optimizers_on_data.py --output data/optimizer_comparison/results.csv
  python append_results/run_optimizers_on_data.py --functions 1 2 8
  python append_results/run_optimizers_on_data.py --seeds 42 43 44
  python append_results/run_optimizers_on_data.py --optuna-sampler gp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.compare_solvers import prepare_y_for_surrogates
from src.utils.load_challenge_data import load_function_data, load_problem_data_csv


def load_problem(fid: int) -> tuple[np.ndarray, np.ndarray]:
    """Load (X, y) for function fid. CSV if present, else initial_data."""
    csv_path = ROOT / "data" / "problems" / f"function_{fid}" / "observations.csv"
    if csv_path.exists():
        return load_problem_data_csv(csv_path)
    return load_function_data(fid)


def _load_notebook_suggestion(fid: int) -> np.ndarray | None:
    """Read the notebook's saved next_input_portal.txt for function *fid*."""
    txt = ROOT / "data" / "submissions" / f"function_{fid}" / "next_input_portal.txt"
    if not txt.exists():
        return None
    parts = txt.read_text().strip().split("-")
    return np.array([float(v) for v in parts])


def run_solvers(
    functions: list[int] | None = None,
    solvers: list[str] | None = None,
    output_path: Path | None = None,
    seed: int = 42,
    seeds: list[int] | None = None,
    optuna_sampler: str | None = None,
) -> dict[int, dict[str, np.ndarray | None | dict[int, np.ndarray]]]:
    """
    Run each solver on each function.

    If ``seeds`` has one entry, each results[fid][name] is a 1d array or None.
    If ``seeds`` has multiple entries, results[fid][name] is dict[seed, array] or None,
    except meta keys n_obs, best_y, dim, y_warp_mode.
    """
    if functions is None:
        functions = list(range(1, 9))
    if solvers is None:
        solvers = ["notebook", "optuna"]
    seed_list = seeds if seeds is not None else [seed]

    results: dict[int, dict] = {}
    for fid in functions:
        X, y_raw = load_problem(fid)
        d = X.shape[1]
        bounds = [(0.0, 1.0)] * d
        n_obs = len(y_raw)
        best_y = float(np.max(y_raw))
        y_fit, y_warp_mode = prepare_y_for_surrogates(y_raw, fid)
        results[fid] = {
            "n_obs": n_obs,
            "best_y": best_y,
            "dim": d,
            "y_warp_mode": y_warp_mode,
        }

        multi = len(seed_list) > 1
        for run_seed in seed_list:
            for name in solvers:
                key = name
                try:
                    if name == "notebook":
                        x_next = _load_notebook_suggestion(fid)
                        if x_next is None:
                            raise FileNotFoundError(f"no next_input_portal.txt for function_{fid}")
                    elif name == "my_bo":
                        from src.optimizers.my_bayesian.my_gp_skopt import suggest as my_suggest

                        x_next = my_suggest(
                            X,
                            y_fit,
                            bounds=bounds,
                            function_id=fid,
                            seed=run_seed,
                            output_warping="none",
                        )
                    elif name == "optuna":
                        from src.optimizers.wrappers.optuna_solver import suggest as optuna_suggest

                        o_kw: dict = {
                            "bounds": bounds,
                            "function_id": fid,
                            "seed": run_seed,
                        }
                        if optuna_sampler is not None:
                            o_kw["sampler"] = optuna_sampler
                        x_next = optuna_suggest(X, y_fit, **o_kw)
                    elif name == "hyperopt":
                        from src.optimizers.wrappers.hyperopt_solver import suggest as hyperopt_suggest

                        x_next = hyperopt_suggest(X, y_fit, bounds=bounds, function_id=fid, seed=run_seed)
                    elif name == "turbo":
                        from src.optimizers.wrappers.turbo_solver import suggest as turbo_suggest

                        x_next = turbo_suggest(X, y_fit, bounds=bounds, function_id=fid, seed=run_seed)
                    elif name in ("ga", "de_gp_ei"):
                        from src.optimizers.wrappers.de_gp_ei_solver import suggest as de_gp_ei_suggest

                        x_next = de_gp_ei_suggest(X, y_fit, bounds=bounds, function_id=fid, seed=run_seed)
                    else:
                        raise ValueError(f"Unknown solver: {name}")
                    arr = np.asarray(x_next).ravel()
                    if multi:
                        if key not in results[fid]:
                            results[fid][key] = {}
                        results[fid][key][run_seed] = arr  # type: ignore[index]
                    else:
                        results[fid][key] = arr
                except Exception as e:
                    if multi:
                        if key not in results[fid]:
                            results[fid][key] = {}
                        results[fid][key][run_seed] = None  # type: ignore[index]
                        results[fid][f"{key}_error_{run_seed}"] = str(e)[:120]
                    else:
                        results[fid][key] = None
                        results[fid][f"{key}_error"] = str(e)[:120]

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(results, output_path, solvers, seed_list)

    return results


def _write_csv(
    results: dict,
    path: Path,
    solvers: list[str],
    seed_list: list[int],
) -> None:
    """Write CSV: function_id, solver, seed (if multi), x1, ..."""
    rows = []
    multi = len(seed_list) > 1
    for fid in sorted(results.keys()):
        r = results[fid]
        d = r["dim"]
        for name in solvers:
            if multi:
                for s in seed_list:
                    x = None
                    slot = r.get(name)
                    if isinstance(slot, dict):
                        x = slot.get(s)
                    if x is None:
                        continue
                    row = [fid, name, s] + [f"{x[j]:.6f}" for j in range(d)]
                    rows.append(row)
            else:
                x = r.get(name)
                if x is None or isinstance(x, dict):
                    continue
                row = [fid, name, ""] + [f"{x[j]:.6f}" for j in range(d)]
                rows.append(row)
    if not rows:
        return
    max_d = max(r["dim"] for r in results.values())
    header = ["function_id", "solver", "seed"] + [f"x{j+1}" for j in range(max_d)]
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            while len(row) < len(header):
                row.append("")
            f.write(",".join(str(v) for v in row[: len(header)]) + "\n")
    print(f"Wrote {path}")


def main():
    p = argparse.ArgumentParser(description="Run BO solvers on challenge data and compare suggestions.")
    p.add_argument("--functions", type=int, nargs="+", default=None, help="Function IDs (default: 1..8)")
    p.add_argument(
        "--solvers",
        type=str,
        nargs="+",
        default=["notebook", "optuna"],
        help="Solvers: notebook, my_bo, optuna, hyperopt, turbo, ga|de_gp_ei (DE-GP-EI)",
    )
    p.add_argument("--output", "-o", type=Path, default=None, help="Output CSV path")
    p.add_argument("--seed", type=int, default=42, help="Random seed when --seeds is not set")
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run each solver with these seeds (e.g. 42 43 44); CSV includes a seed column.",
    )
    p.add_argument(
        "--optuna-sampler",
        type=str,
        default=None,
        help="Override Optuna sampler: tpe, gp, cmaes, random (default: YAML / TPE)",
    )
    args = p.parse_args()

    results = run_solvers(
        functions=args.functions,
        solvers=args.solvers,
        output_path=args.output,
        seed=args.seed,
        seeds=args.seeds,
        optuna_sampler=args.optuna_sampler,
    )

    seed_list = args.seeds if args.seeds is not None else [args.seed]
    multi = len(seed_list) > 1
    solvers = args.solvers or ["notebook", "optuna"]
    print("\n" + "=" * 70)
    print("Suggested next point per function and solver (portal format: 6 decimals, hyphen-separated)")
    print("=" * 70)
    for fid in sorted(results.keys()):
        r = results[fid]
        n_obs, best_y, d = r["n_obs"], r["best_y"], r["dim"]
        ywm = r.get("y_warp_mode")
        wm = f", y_warp={ywm!r}" if ywm else ""
        print(f"\nFunction {fid} (d={d}, n={n_obs}, best_y(raw)={best_y:.4g}{wm})")
        for name in solvers:
            if multi:
                slot = r.get(name)
                if not isinstance(slot, dict):
                    err = r.get(f"{name}_error_{seed_list[0]}", r.get(f"{name}_error", "?"))
                    print(f"  {name}: ERROR — {err}")
                    continue
                for s in seed_list:
                    x = slot.get(s)
                    if x is None:
                        err = r.get(f"{name}_error_{s}", "?")
                        print(f"  {name} (seed={s}): ERROR — {err}")
                    else:
                        portal_str = "-".join(f"{float(x[j]):.6f}" for j in range(len(x)))
                        print(f"  {name} (seed={s}): {portal_str}")
            else:
                x = r.get(name)
                if isinstance(x, dict):
                    x = None
                if x is None:
                    err = r.get(f"{name}_error", "?")
                    print(f"  {name}: ERROR — {err}")
                else:
                    portal_str = "-".join(f"{float(x[j]):.6f}" for j in range(len(x)))
                    print(f"  {name}: {portal_str}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
