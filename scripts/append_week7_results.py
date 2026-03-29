#!/usr/bin/env python3
"""
Append Week 7 portal results to local datasets (data/problems/function_N/).
Under data/ we use only CSV: observations.csv. No .npy in data/problems/.
Run from project root. Idempotent: skips if Week 7 point already in dataset.
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

# Week 7 results from portal (input list, output scalar per function)
WEEK7 = {
    1: (np.array([0.376332, 0.390541]), -0.0006340007160880252),
    2: (np.array([0.699700, 0.067846]), 0.7248315392864594),
    3: (np.array([0.183260, 0.050457, 0.613467]), -0.14562454432155156),
    4: (np.array([0.438749, 0.327912, 0.345519, 0.387965]), -0.07265191566424489),
    5: (np.array([0.991716, 0.989804, 0.990131, 0.922233]), 6881.242859801931),
    6: (np.array([0.400636, 0.360119, 0.888644, 0.880408, 0.007806]), -0.4163999800263345),
    7: (np.array([0.044092, 0.220143, 0.354274, 0.166747, 0.268236, 0.598000]), 2.394174234566946),
    8: (
        np.array([0.047361, 0.805176, 0.003143, 0.369563, 0.545514, 0.957356, 0.028878, 0.951107]),
        9.1562675109191,
    ),
}

ATOL = 1e-9

CSV_NAME = "observations.csv"


def _already_appended(X_saved: np.ndarray, x_new: np.ndarray) -> bool:
    if X_saved is None or len(X_saved) == 0:
        return False
    return np.any(np.all(np.isclose(X_saved, x_new.ravel(), atol=ATOL), axis=1))


def _load_current(out_dir: Path, fid: int):
    """Load current data from CSV only (under data/ we use only CSV). Returns (X, y) or None."""
    csv_path = out_dir / CSV_NAME
    if csv_path.exists():
        return load_problem_data_csv(csv_path)
    return None


def main():
    problems_dir = ROOT / "data" / "problems"
    problems_dir.mkdir(parents=True, exist_ok=True)
    for fid in range(1, 9):
        x_new, y_new = WEEK7[fid]
        x_new = np.asarray(x_new, dtype=np.float64).reshape(1, -1)
        out_dir = problems_dir / f"function_{fid}"
        csv_path = out_dir / CSV_NAME

        current = _load_current(out_dir, fid)
        if current is not None:
            X_cur, y_cur = current
            if _already_appended(X_cur, x_new):
                print(f"Function {fid}: already appended (Week 7 in dataset), skip. Points: {len(y_cur)}")
                continue
            assert x_new.shape[1] == X_cur.shape[1], f"Function {fid}: dimension mismatch"
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
    print("Done. data/problems/function_1..8 updated (CSV). Re-run notebooks for Week 8.")


if __name__ == "__main__":
    main()
