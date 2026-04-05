#!/usr/bin/env python3
"""
Append Week 9 portal results to local datasets (data/problems/function_N/).
Under data/ we use only CSV: observations.csv. No .npy in data/problems/.
Run from project root.

Idempotent for *new* points: skips append if this Week 9 input row already exists with the same y.
If the same input row exists but y differs (e.g. corrected / duplicate week), updates y to this Week 9 value.
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

# Week 9 results from portal (input list, output scalar per function)
WEEK9 = {
    1: (np.array([0.474452, 0.419810]), -0.0057984983828449875),
    2: (np.array([0.651544, 0.050385]), 0.3216509530803791),
    3: (np.array([0.464456, 0.848362, 0.050182]), -0.06243729940358006),
    4: (np.array([0.383207, 0.399323, 0.355895, 0.338283]), 0.2986720728440919),
    5: (np.array([0.995970, 0.917847, 0.960052, 0.999812]), 6585.443172394044),
    6: (np.array([0.481165, 0.388661, 0.645839, 0.784362, 0.189561]), -0.18827120393119393),
    7: (np.array([0.127783, 0.186740, 0.300025, 0.279050, 0.374934, 0.595440]), 2.714099512784121),
    8: (
        np.array([0.055574, 0.117504, 0.049096, 0.170442, 0.742522, 0.577505, 0.168337, 0.415774]),
        9.9618845257074,
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
        x_new, y_new = WEEK9[fid]
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
                    print(f"Function {fid}: Week 9 row already in dataset (same x,y), skip. Points: {len(y_cur)}")
                    continue
                y_cur = y_cur.copy()
                y_cur[j] = y_new
                X_updated, y_updated = X_cur, y_cur
                save_problem_data_csv(csv_path, X_updated, y_updated)
                print(f"Function {fid}: updated row {j+1} y -> Week 9 value | {len(y_updated)} points -> {csv_path.name}")
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
