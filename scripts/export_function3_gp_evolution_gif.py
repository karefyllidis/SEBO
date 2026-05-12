#!/usr/bin/env python3
"""
Build an animated GIF of Function 3 observation evolution as the BO loop progresses.

Mirrors the **IDW pairwise** plot from Section 2 of
``notebooks/function_3_Drug-Discovery.ipynb`` (inverse-distance-weighted y surface,
magma colormap, red observations with white round numbers, "(interpolated)" titles,
shared y colourbar on the right). Frames step through the observation count, from
the warm-start (15 points) to the full dataset.

Requires a local ``data/problems/function_3/observations.csv`` (warm-start rows
first, then weekly appends). Run from the repository root:

    python scripts/export_function3_gp_evolution_gif.py

Outputs ``docs/gp_surrogate_function3_evolution.gif`` by default.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# Repo root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.load_challenge_data import load_function_data, load_problem_data_csv  # noqa: E402
from src.utils.plot_utilities import setup_matplotlib  # noqa: E402

# --- Aesthetic constants aligned with Section 2 of function_3_Drug-Discovery.ipynb ---
CMAP = "magma"
CONTOURF_LEVELS = 30
CONTOUR_LINE_LEVELS = 11
CONTOUR_LINEWIDTH = 1.0
CONTOUR_ALPHA = 0.5
SCATTER_EDGE_LW = 1.5
SCATTER_SIZE = 50
N_GRID = 60  # Section 2 uses n_grid_viz = 60
F3_FIG_1X3 = (16.0, 5.0)
PAIRS = [
    (0, 1, r"$x_1$", r"$x_2$"),
    (0, 2, r"$x_1$", r"$x_3$"),
    (1, 2, r"$x_2$", r"$x_3$"),
]


def _idw_grids(n_grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ug = np.linspace(0, 1, n_grid)
    Ug, Vg = np.meshgrid(ug, ug)
    return Ug, Vg, ug


def _idw_surfaces(X: np.ndarray, y: np.ndarray, Ug: np.ndarray, Vg: np.ndarray) -> list[np.ndarray]:
    """One IDW (1 / d^2) surface per pair, matching Section 2's algebra."""
    surfaces: list[np.ndarray] = []
    grid_2d = np.column_stack([Ug.ravel(), Vg.ravel()])
    for i, j, *_ in PAIRS:
        obs_2d = np.column_stack([X[:, i], X[:, j]])
        dist = np.sqrt(((grid_2d[:, None, :] - obs_2d[None, :, :]) ** 2).sum(axis=2)) + 1e-12
        w = 1.0 / (dist**2)
        Y_idw = (w * y[None, :]).sum(axis=1) / w.sum(axis=1)
        surfaces.append(Y_idw.reshape(Ug.shape))
    return surfaces


def _global_color_range(
    X_full: np.ndarray, y_full: np.ndarray, Ug: np.ndarray, Vg: np.ndarray, n_init: int
) -> tuple[float, float]:
    """Use the **final** IDW surface to fix colour limits — keeps frames comparable."""
    surfaces = _idw_surfaces(X_full, y_full, Ug, Vg)
    lo = float(min(np.nanmin(s) for s in surfaces))
    hi = float(max(np.nanmax(s) for s in surfaces))
    # Also fold the observed y range in case early frames push slightly outside.
    lo = min(lo, float(y_full[:n_init].min()), float(y_full.min()))
    hi = max(hi, float(y_full[:n_init].max()), float(y_full.max()))
    if hi <= lo:
        hi = lo + 1e-10
    return lo, hi


def _render_frame_png_bytes(
    X: np.ndarray,
    y: np.ndarray,
    *,
    Ug: np.ndarray,
    Vg: np.ndarray,
    levels: np.ndarray,
    vmin: float,
    vmax: float,
    n_obs: int,
    bo_rounds_done: int,
    dpi: float,
    figsize: tuple[float, float],
) -> bytes:
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    surfaces = _idw_surfaces(X, y, Ug, Vg)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    line_levels = levels[:: max(1, len(levels) // CONTOUR_LINE_LEVELS)]
    for ax, (i, j, li, lj), Y_idw in zip(axes, PAIRS, surfaces, strict=True):
        ax.contourf(Ug, Vg, Y_idw, levels=levels, cmap=CMAP, norm=norm)
        ax.contour(
            Ug,
            Vg,
            Y_idw,
            levels=line_levels,
            colors="white",
            linewidths=CONTOUR_LINEWIDTH,
            linestyles="-",
            alpha=CONTOUR_ALPHA,
        )
        ax.scatter(
            X[:, i],
            X[:, j],
            c="red",
            s=SCATTER_SIZE,
            edgecolors="k",
            linewidths=SCATTER_EDGE_LW,
            zorder=2,
        )
        for idx in range(len(y)):
            ax.text(
                float(X[idx, i]) + 0.02,
                float(X[idx, j]) + 0.02,
                str(idx + 1),
                fontsize=8,
                color="white",
                zorder=10,
            )
        ax.set_xlabel(li)
        ax.set_ylabel(lj)
        ax.set_title(f"{li} vs {lj} (interpolated)")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), location="right", shrink=0.8, label="y", pad=0.02)
    fig.suptitle(
        f"Function 3 — observations on pairwise projections (IDW)  |  "
        f"n = {n_obs} obs  |  BO rounds after warm-start: {bo_rounds_done}",
        fontsize=11,
        y=1.02,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "data" / "problems" / "function_3" / "observations.csv",
        help="Path to function_3 observations.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "gp_surrogate_function3_evolution.gif",
        help="Output GIF path",
    )
    parser.add_argument(
        "--duration-ms",
        type=int,
        default=600,
        help="Display duration per frame in milliseconds",
    )
    parser.add_argument(
        "--hold-last-ms",
        type=int,
        default=1800,
        help="Hold the final frame longer (milliseconds)",
    )
    parser.add_argument("--dpi", type=float, default=110.0, help="Rasterisation DPI")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=F3_FIG_1X3,
        metavar=("W", "H"),
        help="Figure size in inches (width height)",
    )
    parser.add_argument(
        "--no-verify-init",
        action="store_true",
        help="Skip checking that CSV rows match initial_data warm-start",
    )
    args = parser.parse_args()

    setup_matplotlib()

    if not args.csv.is_file():
        sys.stderr.write(
            f"Missing {args.csv}. Export needs local observations (gitignored). "
            "Restore CSV under data/problems/function_3/.\n"
        )
        sys.exit(1)

    X_full, y_full = load_problem_data_csv(args.csv)
    X_init, _ = load_function_data(3)
    n_init = len(X_init)
    if not args.no_verify_init:
        if X_full.shape[0] < n_init or X_full.shape[1] != X_init.shape[1]:
            sys.stderr.write(
                f"CSV has shape {X_full.shape}; expected at least {n_init} rows "
                f"and d={X_init.shape[1]}.\n"
            )
            sys.exit(1)
        if not np.allclose(X_full[:n_init], X_init, rtol=0.0, atol=1e-5):
            sys.stderr.write(
                "First rows of observations.csv do not match initial_data/function_3. "
                "Use --no-verify-init to override.\n"
            )
            sys.exit(1)

    Ug, Vg, _ = _idw_grids(N_GRID)
    vmin, vmax = _global_color_range(X_full, y_full, Ug, Vg, n_init)
    levels = np.linspace(vmin, vmax, CONTOURF_LEVELS)

    frames: list[Image.Image] = []
    n_total = len(X_full)
    for end in range(n_init, n_total + 1):
        png = _render_frame_png_bytes(
            X_full[:end],
            y_full[:end],
            Ug=Ug,
            Vg=Vg,
            levels=levels,
            vmin=vmin,
            vmax=vmax,
            n_obs=end,
            bo_rounds_done=max(0, end - n_init),
            dpi=args.dpi,
            figsize=(args.figsize[0], args.figsize[1]),
        )
        frames.append(Image.open(io.BytesIO(png)).convert("RGB"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = max(20, int(args.duration_ms))
    hold = max(base, int(args.hold_last_ms))
    durations = [base] * (len(frames) - 1) + [hold]
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {len(frames)} frames -> {args.output}")


if __name__ == "__main__":
    main()
