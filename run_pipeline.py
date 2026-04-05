#!/usr/bin/env python3
"""BBO pipeline: run append_results/*.py, execute notebooks, print submission summary. Run from project root."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APPEND_RESULTS = ROOT / "append_results"

_POST_NOTEBOOK_SCRIPTS = {"run_optimizers_on_data.py"}


def run_scripts(skip=False, only_post=False):
    """Yield (name, ok, msg) for *.py in append_results/.

    If *only_post* is True, run only post-notebook scripts.
    Otherwise run everything except post-notebook scripts.
    """
    for p in sorted(APPEND_RESULTS.glob("*.py")):
        is_post = p.name in _POST_NOTEBOOK_SCRIPTS
        if only_post and not is_post:
            continue
        if not only_post and is_post:
            continue
        if skip:
            yield p.name, True, "skipped"
            continue
        r = subprocess.run([sys.executable, str(p)], cwd=ROOT, capture_output=True, text=True, timeout=120)
        msg = (r.stdout or r.stderr or "").strip() or ("ok" if r.returncode == 0 else f"exit {r.returncode}")
        yield p.name, r.returncode == 0, msg


def execute_notebooks():
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError:
        yield "notebooks", False, "pip install -r requirements.txt (nbconvert, ipykernel)"
        return
    nb_dir = ROOT / "notebooks"
    for i in range(1, 9):
        nbs = list(nb_dir.glob(f"function_{i}_*.ipynb"))
        if not nbs:
            yield f"function_{i}", False, "not found"
            continue
        nb_path = nbs[0]
        try:
            nb = nbformat.read(nb_path, as_version=4)
            ExecutePreprocessor(timeout=300).preprocess(nb, {"metadata": {"path": str(ROOT)}})
            nb_path.write_text(nbformat.writes(nb))
            yield nb_path.name, True, "ok"
        except Exception as e:
            yield nb_path.name, False, str(e)[:60]


def submission_summary():
    sub_dir = ROOT / "data" / "submissions"
    problems_dir = ROOT / "data" / "problems"
    portal = {}
    for n in range(1, 9):
        txt = sub_dir / f"function_{n}" / "next_input_portal.txt"
        npy = sub_dir / f"function_{n}" / "next_input.npy"
        if txt.exists():
            portal[n] = txt.read_text().strip()
        elif npy.exists():
            import numpy as np
            portal[n] = "-".join(f"{x:.6f}" for x in np.load(npy).ravel())
        else:
            portal[n] = "(not generated)"
    missing = []
    for n in range(1, 9):
        fn_dir = problems_dir / f"function_{n}"
        csv_path = fn_dir / "observations.csv"
        has_data = csv_path.exists()
        if not has_data:
            missing.append(n)
    if missing:
        print(
            "NOTE: data/problems/ has no appended data for function(s)",
            missing,
            "\n  → Notebooks will load initial_data only and may suggest the SAME point as last time.\n"
            "  → Run append_results/append_weekN_results.py after portal feedback, then re-run notebooks.",
        )
    return portal


def main():
    p = argparse.ArgumentParser(description="Run append_results/*.py, execute notebooks, print submission summary.")
    p.add_argument("--skip-notebooks", action="store_true", help="Skip notebook execution; only show saved summary")
    p.add_argument(
        "--skip-scripts",
        action="store_true",
        help="Skip append_results/*.py (portal append + optimizer bench); only show summary",
    )
    args = p.parse_args()

    print("run_pipeline.py", ROOT, "\n")

    # Phase 1: data-prep (append_week*.py etc.)
    if APPEND_RESULTS.exists():
        print("Append results:")
        for name, ok, msg in run_scripts(skip=args.skip_scripts, only_post=False):
            print(f"  [{'OK' if ok else 'FAIL'}] {name}: {msg}")
        print()

    # Phase 2: execute notebooks (compute + save submissions)
    if not args.skip_notebooks:
        print("Notebooks:")
        for name, ok, msg in execute_notebooks():
            print(f"  [{'OK' if ok else 'FAIL'}] {name}: {msg}")
        print()

    # Phase 3: post-notebook (run_optimizers_on_data.py — comparison)
    if APPEND_RESULTS.exists():
        post = list(run_scripts(skip=args.skip_scripts, only_post=True))
        if post:
            print("Post-notebook (append_results):")
            for name, ok, msg in post:
                print(f"  [{'OK' if ok else 'FAIL'}] {name}: {msg}")
            print()

    # Phase 4: portal summary from freshly written files
    portal = submission_summary()
    print("=" * 60)
    print("SUBMISSION — portal strings (copy-paste per function)")
    print("=" * 60)
    for n in range(1, 9):
        s = portal.get(n, "(missing)")
        print(f"  {n} | {s}")
    print("=" * 60)
    print("Files: data/submissions/function_N/next_input_portal.txt")
    print("       submission-template/")


if __name__ == "__main__":
    main()
