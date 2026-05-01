#!/usr/bin/env python3
"""
Append Week 12 portal results to local datasets (data/problems/function_N/).
Run from project root.

Idempotent: skips if Week 12 input row exists with same y; updates y if same x but new y.
If multiple CSV rows share the same x (within atol), collapses them to one row with Week 12 y.
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

# Week 12 results from portal email (2026)
WEEK12 = {
    1: (np.array([0.412326, 0.450431]), 0.0854400127915847),
    2: (np.array([0.728372, 0.882371]), 0.4327213060438788),
    3: (np.array([0.420202, 0.504677, 0.451972]), -0.0031775339721649763),
    4: (np.array([0.428007, 0.371857, 0.367980, 0.332301]), 0.26094073261010964),
    5: (np.array([0.817213, 0.991318, 0.999221, 0.985301]), 5795.365145157042),
    6: (np.array([0.481165, 0.388661, 0.645839, 0.784362, 0.189561]), -0.2689509767125587),
    7: (
        np.array([0.077416, 0.102168, 0.368352, 0.259725, 0.404457, 0.561304]),
        2.4356358974795804,
    ),
    8: (
        np.array(
            [
                0.019514,
                0.295609,
                0.060409,
                0.058391,
                0.610349,
                0.840275,
                0.071832,
                0.360149,
            ]
        ),
        9.7705433468094,
    ),
}

ATOL = 1e-9
Y_RTOL = 1e-12

CSV_NAME = "observations.csv"


def main():
    problems_dir = ROOT / "data" / "problems"
    problems_dir.mkdir(parents=True, exist_ok=True)
    for fid in range(1, 9):
        x_new, y_new = WEEK12[fid]
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
                        f"Function {fid}: Week 12 row already in dataset (same x,y), skip. Points: {len(y_cur)}"
                    )
                    continue
                y_cur = y_cur.copy()
                y_cur[j] = y_new
                X_updated, y_updated = X_cur, y_cur
                print(
                    f"Function {fid}: updated row {j + 1} y -> Week 12 value | {len(y_updated)} points -> {csv_path.name}"
                )
                save_problem_data_csv(csv_path, X_updated, y_updated)
                continue
            else:
                X_updated = np.vstack([X_cur[~m], x_new])
                y_updated = np.append(y_cur[~m], y_new)
                print(
                    f"Function {fid}: deduped {n_match} rows with same x; Week 12 y | {len(y_updated)} points -> {csv_path.name}"
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
