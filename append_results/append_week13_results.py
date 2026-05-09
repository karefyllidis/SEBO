#!/usr/bin/env python3
"""
Append Week 13 portal results to local datasets (data/problems/function_N/).
Run from project root.

Idempotent: skips if Week 13 input row exists with same y; updates y if same x but new y.
If multiple CSV rows share the same x (within atol), collapses them to one row with Week 13 y.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if (ROOT / "src").exists():
    sys.path.insert(0, str(ROOT))
from src.utils.load_challenge_data import (
    load_function_data,
    load_problem_data_csv,
    save_problem_data_csv,
)

# Week 13 results from portal email (2026)
WEEK13 = {
    1: (np.array([0.130519, 0.662247]), -4.591286787849761e-100),
    2: (np.array([0.713518, 0.142892]), 0.5608464405084902),
    3: (np.array([0.434652, 0.592156, 0.480918]), -0.005776744059530596),
    4: (np.array([0.428007, 0.371857, 0.367980, 0.332301]), 0.26094073261010964),
    5: (np.array([0.905317, 0.974625, 0.950240, 0.982036]), 5676.2961141179185),
    6: (np.array([0.481165, 0.388661, 0.645839, 0.784362, 0.189561]), -0.14017687280075747),
    7: (
        np.array([0.166609, 0.018875, 0.241927, 0.175944, 0.251584, 0.617741]),
        2.349640609855117,
    ),
    8: (
        np.array(
            [
                0.106108,
                0.221344,
                0.004089,
                0.334132,
                0.633680,
                0.380726,
                0.246926,
                0.301113,
            ]
        ),
        9.8719751840441,
    ),
}

ATOL = 1e-9
Y_RTOL = 1e-12

CSV_NAME = "observations.csv"


def main():
    problems_dir = ROOT / "data" / "problems"
    problems_dir.mkdir(parents=True, exist_ok=True)
    for fid in range(1, 9):
        x_new, y_new = WEEK13[fid]
        x_new = np.asarray(x_new, dtype=np.float64).reshape(1, -1)
        x_row = x_new.ravel()
        out_dir = problems_dir / f"function_{fid}"
        csv_path = out_dir / CSV_NAME

        current = None
        if csv_path.exists():
            current = load_problem_data_csv(csv_path)
        if current is not None:
            X_cur, y_cur = current
            assert x_new.shape[1] == X_cur.shape[1], f"Function {fid}: dimension mismatch"
            m = np.all(np.isclose(X_cur, x_row, atol=ATOL), axis=1)
            n_match = int(np.sum(m))
            if n_match == 0:
                X_updated = np.vstack([X_cur, x_new])
                y_updated = np.append(y_cur, y_new)
            elif n_match == 1:
                j = int(np.argmax(m))
                if np.isclose(y_cur[j], y_new, rtol=Y_RTOL, atol=ATOL):
                    print(
                        f"Function {fid}: Week 13 row already in dataset (same x,y), skip. Points: {len(y_cur)}"
                    )
                    continue
                y_cur = y_cur.copy()
                y_cur[j] = y_new
                X_updated, y_updated = X_cur, y_cur
                print(
                    f"Function {fid}: updated row {j + 1} y -> Week 13 value | {len(y_updated)} points -> {csv_path.name}"
                )
                save_problem_data_csv(csv_path, X_updated, y_updated)
                continue
            else:
                X_updated = np.vstack([X_cur[~m], x_new])
                y_updated = np.append(y_cur[~m], y_new)
                print(
                    f"Function {fid}: deduped {n_match} rows with same x; Week 13 y | {len(y_updated)} points -> {csv_path.name}"
                )
                save_problem_data_csv(csv_path, X_updated, y_updated)
                continue
        else:
            X_init, y_init = load_function_data(fid)
            assert x_new.shape[1] == X_init.shape[1], f"Function {fid}: dimension mismatch"
            X_updated = np.vstack([X_init, x_new])
            y_updated = np.append(y_init, y_new)

        out_dir.mkdir(parents=True, exist_ok=True)
        save_problem_data_csv(csv_path, X_updated, y_updated)
        n = len(y_updated)
        print(f"Function {fid}: {n} points -> {csv_path.name}")
    print("Done. data/problems/function_1..8 updated (CSV). Re-run notebooks for the next query.")


if __name__ == "__main__":
    main()
