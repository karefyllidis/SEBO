#!/usr/bin/env python3
"""
Append Week 10 portal results to local datasets (data/problems/function_N/).
Under data/ we use only CSV: observations.csv. No .npy in data/problems/.
Run from project root.

Idempotent for *new* points: skips append if this Week 10 input row already exists with the same y.
If the same input row exists but y differs (e.g. corrected / duplicate week), updates y to this Week 10 value.
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

# Week 10 results from portal (input list, output scalar per function)
WEEK10 = {
    1: (np.array([0.433599, 0.417313]), 0.6703674428760071),
    2: (np.array([0.746742, 0.050126]), 0.2593462299483941),
    3: (np.array([0.437681, 0.398010, 0.444610]), -0.018861538878924992),
    4: (np.array([0.428007, 0.371857, 0.367980, 0.332301]), 0.26094073261010964),
    5: (np.array([0.998976, 0.999253, 0.817947, 0.940257]), 5294.946217718119),
    6: (np.array([0.481165, 0.388661, 0.645839, 0.784362, 0.189561]), -0.160460160640041),
    7: (np.array([0.127119, 0.041886, 0.225997, 0.326884, 0.295103, 0.638188]), 2.7205611715260853),
    8: (
        np.array([0.060041, 0.052933, 0.140975, 0.130168, 0.573728, 0.584904, 0.019586, 0.569622]),
        9.8886309907616,
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


def _load_current(out_dir: Path, fid: int):
    csv_path = out_dir / CSV_NAME
    if csv_path.exists():
        return load_problem_data_csv(csv_path)
    return None


def main():
    problems_dir = ROOT / "data" / "problems"
    problems_dir.mkdir(parents=True, exist_ok=True)
    for fid in range(1, 9):
        x_new, y_new = WEEK10[fid]
        x_new = np.asarray(x_new, dtype=np.float64).reshape(1, -1)
        out_dir = problems_dir / f"function_{fid}"
        csv_path = out_dir / CSV_NAME

        current = _load_current(out_dir, fid)
        if current is not None:
            X_cur, y_cur = current
            assert x_new.shape[1] == X_cur.shape[1], f"Function {fid}: dimension mismatch"
            j = _row_index(X_cur, x_new)
            if j is not None:
                if np.isclose(y_cur[j], y_new, rtol=Y_RTOL, atol=ATOL):
                    print(f"Function {fid}: Week 10 row already in dataset (same x,y), skip. Points: {len(y_cur)}")
                    continue
                y_cur = y_cur.copy()
                y_cur[j] = y_new
                X_updated, y_updated = X_cur, y_cur
                save_problem_data_csv(csv_path, X_updated, y_updated)
                print(f"Function {fid}: updated row {j+1} y -> Week 10 value | {len(y_updated)} points -> {csv_path.name}")
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
