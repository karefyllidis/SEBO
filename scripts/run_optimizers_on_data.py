#!/usr/bin/env python3
"""
Run multiple BO solvers (MyBO, Optuna, etc.) on your challenge data and compare next suggestions.

Loads data from data/problems/function_N/observations.csv (or initial_data if no CSV).
For each function 1..8 and each solver, computes the suggested next point and prints/writes results.

Usage (from project root):
  python scripts/run_optimizers_on_data.py
  python scripts/run_optimizers_on_data.py --output data/optimizer_comparison/results.csv
  python scripts/run_optimizers_on_data.py --functions 1 2 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.load_challenge_data import load_function_data, load_problem_data_csv


def load_problem(fid: int) -> tuple[np.ndarray, np.ndarray]:
    """Load (X, y) for function fid. CSV if present, else initial_data."""
    csv_path = ROOT / "data" / "problems" / f"function_{fid}" / "observations.csv"
    if csv_path.exists():
        return load_problem_data_csv(csv_path)
    return load_function_data(fid)


def run_solvers(
    functions: list[int] | None = None,
    solvers: list[str] | None = None,
    output_path: Path | None = None,
    seed: int = 42,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Run each solver on each function. Return results[function_id][solver_name] = x_next (1d array).
    """
    if functions is None:
        functions = list(range(1, 9))
    if solvers is None:
        solvers = ["my_bo", "optuna"]

    results = {}
    for fid in functions:
        X, y = load_problem(fid)
        d = X.shape[1]
        bounds = [(0.0, 1.0)] * d
        n_obs = len(y)
        best_y = float(np.max(y))
        results[fid] = {"n_obs": n_obs, "best_y": best_y, "dim": d}

        for name in solvers:
            try:
                if name == "my_bo":
                    from src.optimizers.bayesian.my_gp_skopt import suggest as my_suggest
                    x_next = my_suggest(X, y, bounds=bounds, function_id=fid, seed=seed)
                elif name == "optuna":
                    from src.optimizers.wrappers.optuna_solver import suggest as optuna_suggest
                    x_next = optuna_suggest(X, y, bounds=bounds, function_id=fid, seed=seed)
                elif name == "hebo":
                    from src.optimizers.wrappers.hebo_solver import suggest as hebo_suggest
                    x_next = hebo_suggest(X, y, bounds=bounds, function_id=fid, seed=seed)
                elif name == "hyperopt":
                    from src.optimizers.wrappers.hyperopt_solver import suggest as hyperopt_suggest
                    x_next = hyperopt_suggest(X, y, bounds=bounds, function_id=fid, seed=seed)
                elif name == "turbo":
                    from src.optimizers.wrappers.turbo_solver import suggest as turbo_suggest
                    x_next = turbo_suggest(X, y, bounds=bounds, function_id=fid, seed=seed)
                elif name == "ray_tune":
                    from src.optimizers.wrappers.ray_tune_solver import suggest as ray_tune_suggest
                    x_next = ray_tune_suggest(X, y, bounds=bounds, function_id=fid, seed=seed)
                else:
                    raise ValueError(f"Unknown solver: {name}")
                results[fid][name] = np.asarray(x_next).ravel()
            except Exception as e:
                results[fid][name] = None
                results[fid][f"{name}_error"] = str(e)[:80]

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(results, output_path, solvers)

    return results


def _write_csv(results: dict, path: Path, solvers: list[str]) -> None:
    """Write a CSV with one row per (function_id, solver): function_id, solver, x1, x2, ..."""
    rows = []
    for fid in sorted(results.keys()):
        r = results[fid]
        d = r["dim"]
        for name in solvers:
            x = r.get(name)
            if x is None:
                continue
            row = [fid, name] + [f"{x[j]:.6f}" for j in range(d)]
            rows.append(row)
    if not rows:
        return
    max_d = max(r["dim"] for r in results.values())
    header = ["function_id", "solver"] + [f"x{j+1}" for j in range(max_d)]
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            # Pad row if needed
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
        default=["my_bo", "optuna"],
        help="Solvers: my_bo, optuna, hebo, hyperopt, turbo, ray_tune",
    )
    p.add_argument("--output", "-o", type=Path, default=None, help="Output CSV path")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    results = run_solvers(
        functions=args.functions,
        solvers=args.solvers,
        output_path=args.output,
        seed=args.seed,
    )

    # Print table
    solvers = args.solvers or ["my_bo", "optuna"]
    print("\n" + "=" * 70)
    print("Suggested next point per function and solver (portal format: 6 decimals, hyphen-separated)")
    print("=" * 70)
    for fid in sorted(results.keys()):
        r = results[fid]
        n_obs, best_y, d = r["n_obs"], r["best_y"], r["dim"]
        print(f"\nFunction {fid} (d={d}, n={n_obs}, best_y={best_y:.4g})")
        for name in solvers:
            x = r.get(name)
            if x is None:
                err = r.get(f"{name}_error", "?")
                print(f"  {name}: ERROR — {err}")
            else:
                portal_str = "-".join(f"{float(x[j]):.6f}" for j in range(len(x)))
                print(f"  {name}: {portal_str}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
