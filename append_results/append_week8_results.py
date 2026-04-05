#!/usr/bin/env python3
"""
Append Week 8 portal results to local datasets (data/problems/function_N/).
Under data/ we use only CSV: observations.csv. No .npy in data/problems/.
Run from project root. Idempotent: skips if Week 8 point already in dataset.
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

# Week 8 results from portal (input list, output scalar per function)
WEEK8 = {
    1: (np.array([0.444679, 0.420876]), 0.2525433426527988),
    2: (np.array([0.746609, 0.105042]), 0.4495610978451831),
    3: (np.array([0.368349, 0.436108, 0.530551]), -0.01896340547730605),
    4: (np.array([0.434599, 0.393585, 0.261779, 0.358912]), -1.3002280188067235),
    5: (np.array([0.931482, 0.994251, 0.987514, 0.989074]), 6996.271579906622),
    6: (np.array([0.481165, 0.388661, 0.645839, 0.784362, 0.189561]), -0.237216252018943),
    7: (np.array([0.094171, 0.197407, 0.284847, 0.249358, 0.346147, 0.639900]), 2.7968085377147034),
    8: (
        np.array([0.104715, 0.039158, 0.078624, 0.039779, 0.039491, 0.131645, 0.054677, 0.705337]),
        9.4993829270366,
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
        x_new, y_new = WEEK8[fid]
        x_new = np.asarray(x_new, dtype=np.float64).reshape(1, -1)
        out_dir = problems_dir / f"function_{fid}"
        csv_path = out_dir / CSV_NAME

        current = _load_current(out_dir, fid)
        if current is not None:
            X_cur, y_cur = current
            if _already_appended(X_cur, x_new):
                print(f"Function {fid}: already appended (Week 8 in dataset), skip. Points: {len(y_cur)}")
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
    print("Done. data/problems/function_1..8 updated (CSV). Re-run notebooks for Week 9.")


if __name__ == "__main__":
    main()
