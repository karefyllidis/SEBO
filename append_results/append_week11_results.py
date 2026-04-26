#!/usr/bin/env python3
"""
Append Week 11 portal results to local datasets (data/problems/function_N/).
Run from project root.

Idempotent: skips if Week 11 input row exists with same y; updates y if same x but new y.
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

# Week 11 results from portal email (2026)
WEEK11 = {
    1: (np.array([0.624272, 0.128322]), -8.698804499956185e-90),
    2: (np.array([0.715142, 0.266021]), 0.5977002156291722),
    3: (np.array([0.324369, 0.377106, 0.478210]), -0.015862332750060177),
    4: (np.array([0.428007, 0.371857, 0.367980, 0.332301]), 0.26094073261010964),
    5: (np.array([0.973102, 0.999937, 0.921008, 0.968818]), 6394.8011302310115),
    6: (np.array([0.481165, 0.388661, 0.645839, 0.784362, 0.189561]), -0.12335688995008082),
    7: (
        np.array([0.102964, 0.203430, 0.183624, 0.388057, 0.303266, 0.554585]),
        2.2368932340501178,
    ),
    8: (
        np.array(
            [
                0.000913,
                0.146301,
                0.026885,
                0.203975,
                0.870160,
                0.547260,
                0.349740,
                0.705928,
            ]
        ),
        9.8948775102426,
    ),
}

ATOL = 1e-9
Y_RTOL = 1e-12

CSV_NAME = "observations.csv"


def _row_index(X_saved: np.ndarray, x_new: np.ndarray) -> int | None:
    if X_saved is None or len(X_saved) == 0:
        return None
    m = np.all(np.isclose(X_saved, x_new.ravel(), atol=ATOL), axis=1)
    if not np.any(m):
        return None
    return int(np.argmax(m))


def main():
    problems_dir = ROOT / "data" / "problems"
    problems_dir.mkdir(parents=True, exist_ok=True)
    for fid in range(1, 9):
        x_new, y_new = WEEK11[fid]
        x_new = np.asarray(x_new, dtype=np.float64).reshape(1, -1)
        out_dir = problems_dir / f"function_{fid}"
        csv_path = out_dir / CSV_NAME

        current = None
        if csv_path.exists():
            current = load_problem_data_csv(csv_path)
        if current is not None:
            X_cur, y_cur = current
            assert x_new.shape[1] == X_cur.shape[1], f"Function {fid}: dimension mismatch"
            j = _row_index(X_cur, x_new)
            if j is not None:
                if np.isclose(y_cur[j], y_new, rtol=Y_RTOL, atol=ATOL):
                    print(
                        f"Function {fid}: Week 11 row already in dataset (same x,y), skip. Points: {len(y_cur)}"
                    )
                    continue
                y_cur = y_cur.copy()
                y_cur[j] = y_new
                X_updated, y_updated = X_cur, y_cur
                save_problem_data_csv(csv_path, X_updated, y_updated)
                print(
                    f"Function {fid}: updated row {j+1} y -> Week 11 value | {len(y_updated)} points -> {csv_path.name}"
                )
                continue
            X_updated = np.vstack([X_cur, x_new])
            y_updated = np.append(y_cur, y_new)
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
