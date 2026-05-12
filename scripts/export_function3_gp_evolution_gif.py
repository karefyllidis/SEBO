#!/usr/bin/env python3
"""
Build an animated GIF of Function 3 GP surrogate mean slices as observations accumulate.

Mirrors ``notebooks/function_3_Drug-Discovery.ipynb`` GP kernels, LML selection,
pairwise slice geometry (median held-out coordinate), and contour styling for the
**posterior mean** row only (keeps file size reasonable).

Requires a local ``data/problems/function_3/observations.csv`` (warm-start rows first,
then weekly appends). Run from the repository root:

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
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import (  # noqa: E402
    ConstantKernel,
    Matern,
    RBF,
    WhiteKernel,
)

# Repo root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.load_challenge_data import load_function_data, load_problem_data_csv  # noqa: E402
from src.utils.plot_utilities import setup_matplotlib  # noqa: E402
from src.utils.warping import apply_output_warping  # noqa: E402

# --- Defaults aligned with ``function_3_Drug-Discovery.ipynb`` Parameters / GP cells ---
OUTPUT_WARPING = None
CONSTANT_KERNEL_SCALE = 1.0
LENGTH_SCALE = 1.0
GP_ALPHA = 1e-6
MATERN_NU = 1.5
WHITE_NOISE_LEVEL = 1e-5
GP_KERNEL = None  # None → LML pick among three kernels
OPTIMIZE_KERNEL = True
N_RESTARTS_KERNEL = 15
CONSTANT_SCALE_BOUNDS = (1e-3, 1e3)
LENGTH_SCALE_BOUNDS = (1e-2, 100)
WHITE_NOISE_BOUNDS = (1e-12, 1e1)

CONTOURF_LEVELS = 30
CONTOUR_LINE_LEVELS = 11
CONTOUR_LINEWIDTH = 1.0
CONTOUR_LINESTYLE = "-"
CONTOUR_ALPHA = 0.5

CANONICAL_NAMES = ["RBF", "Matérn (ν=1.5)", "RBF + WhiteKernel"]
KERNEL_ALIASES = {
    "rbf": "RBF",
    "matern": "Matérn (ν=1.5)",
    "matérn (ν=1.5)": "Matérn (ν=1.5)",
    "matern (ν=1.5)": "Matérn (ν=1.5)",
    "rbf + whitekernel": "RBF + WhiteKernel",
    "rbf+whitekernel": "RBF + WhiteKernel",
    "rbf + white": "RBF + WhiteKernel",
    "white": "RBF + WhiteKernel",
}


def _normalize_kernel_input(val: object) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    if s.lower() in ("none", ""):
        return None
    return s


def _resolve_kernel_name(user_input: str | None) -> str | None:
    if user_input is None:
        return None
    key = user_input.lower().strip()
    if key in KERNEL_ALIASES:
        return KERNEL_ALIASES[key]
    if user_input in CANONICAL_NAMES:
        return user_input
    for c in CANONICAL_NAMES:
        if c.lower() == key or key in c.lower():
            return c
    return None


def fit_best_gp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    gp_kernel: str | None,
    optimize_kernel: bool,
    n_restarts_kernel: int,
) -> tuple[GaussianProcessRegressor, str]:
    """Fit three kernels, pick best by LML (or forced kernel). Matches notebook logic."""
    _gp_optimizer = None if not optimize_kernel else "fmin_l_bfgs_b"
    _gp_n_restarts = n_restarts_kernel if optimize_kernel else 0

    kernel_RBF = (
        ConstantKernel(CONSTANT_KERNEL_SCALE, constant_value_bounds=CONSTANT_SCALE_BOUNDS)
        * RBF(length_scale=LENGTH_SCALE, length_scale_bounds=LENGTH_SCALE_BOUNDS)
    )
    kernel_Matern = (
        ConstantKernel(CONSTANT_KERNEL_SCALE, constant_value_bounds=CONSTANT_SCALE_BOUNDS)
        * Matern(
            length_scale=LENGTH_SCALE,
            nu=MATERN_NU,
            length_scale_bounds=LENGTH_SCALE_BOUNDS,
        )
    )
    kernel_RBF_noise = (
        ConstantKernel(CONSTANT_KERNEL_SCALE, constant_value_bounds=CONSTANT_SCALE_BOUNDS)
        * RBF(length_scale=LENGTH_SCALE, length_scale_bounds=LENGTH_SCALE_BOUNDS)
        + WhiteKernel(noise_level=WHITE_NOISE_LEVEL, noise_level_bounds=WHITE_NOISE_BOUNDS)
    )

    gp_RBF = GaussianProcessRegressor(
        kernel=kernel_RBF,
        alpha=GP_ALPHA,
        optimizer=_gp_optimizer,
        n_restarts_optimizer=_gp_n_restarts,
        normalize_y=True,
    )
    gp_Matern = GaussianProcessRegressor(
        kernel=kernel_Matern,
        alpha=GP_ALPHA,
        optimizer=_gp_optimizer,
        n_restarts_optimizer=_gp_n_restarts,
        normalize_y=True,
    )
    gp_RBF_noise = GaussianProcessRegressor(
        kernel=kernel_RBF_noise,
        alpha=GP_ALPHA,
        optimizer=_gp_optimizer,
        n_restarts_optimizer=_gp_n_restarts,
        normalize_y=True,
    )

    gp_RBF.fit(X, y)
    gp_Matern.fit(X, y)
    gp_RBF_noise.fit(X, y)

    gps = [
        (gp_RBF, "RBF"),
        (gp_Matern, "Matérn (ν=1.5)"),
        (gp_RBF_noise, "RBF + WhiteKernel"),
    ]

    use_lml = _normalize_kernel_input(gp_kernel) is None
    if use_lml:
        lml_scores = {name: gp.log_marginal_likelihood_value_ for gp, name in gps}
        best_name = max(lml_scores, key=lml_scores.get)
    else:
        resolved = _resolve_kernel_name(_normalize_kernel_input(gp_kernel))
        if resolved is None:
            lml_scores = {name: gp.log_marginal_likelihood_value_ for gp, name in gps}
            best_name = max(lml_scores, key=lml_scores.get)
        else:
            best_name = resolved

    best_gp = next(g for g, n in gps if n == best_name)
    return best_gp, best_name


def _slice_grids(n_slice: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ug = np.linspace(0, 1, n_slice)
    Ug, Vg = np.meshgrid(ug, ug)
    return Ug, Vg, ug


def build_slices_info(X: np.ndarray, Ug: np.ndarray, Vg: np.ndarray):
    med = [float(np.median(X[:, k])) for k in range(3)]
    return [
        (
            np.column_stack([Ug.ravel(), Vg.ravel(), np.full(Ug.size, med[2])]),
            Ug,
            Vg,
            0,
            1,
            r"$x_1$",
            r"$x_2$",
            rf"$x_3$ = {med[2]:.2f}",
        ),
        (
            np.column_stack([Ug.ravel(), np.full(Ug.size, med[1]), Vg.ravel()]),
            Ug,
            Vg,
            0,
            2,
            r"$x_1$",
            r"$x_3$",
            rf"$x_2$ = {med[1]:.2f}",
        ),
        (
            np.column_stack([np.full(Ug.size, med[0]), Ug.ravel(), Vg.ravel()]),
            Ug,
            Vg,
            1,
            2,
            r"$x_2$",
            r"$x_3$",
            rf"$x_1$ = {med[0]:.2f}",
        ),
    ]


def mu_ranges_for_prefix(
    X: np.ndarray,
    y: np.ndarray,
    *,
    Ug: np.ndarray,
    Vg: np.ndarray,
    gp_kernel: str | None,
    optimize_kernel: bool,
    n_restarts_kernel: int,
) -> list[tuple[float, float]]:
    """Min/max of μ over each slice row for stable colour limits across frames."""
    best_gp, _ = fit_best_gp(
        X,
        y,
        gp_kernel=gp_kernel,
        optimize_kernel=optimize_kernel,
        n_restarts_kernel=n_restarts_kernel,
    )
    slices_info = build_slices_info(X, Ug, Vg)
    ranges: list[tuple[float, float]] = []
    for slice_pts, *_rest in slices_info:
        mu_slice, _ = best_gp.predict(slice_pts, return_std=True)
        mu_slice = mu_slice.reshape(Ug.shape)
        lo, hi = float(np.nanmin(mu_slice)), float(np.nanmax(mu_slice))
        if hi <= lo:
            hi = lo + 1e-10
        ranges.append((lo, hi))
    return ranges


def render_mean_row_png_bytes(
    best_gp: GaussianProcessRegressor,
    best_name: str,
    X: np.ndarray,
    *,
    Ug: np.ndarray,
    Vg: np.ndarray,
    mu_ranges: list[tuple[float, float]],
    warp_label: str,
    n_obs: int,
    bo_rounds_done: int,
    dpi: float,
    figsize: tuple[float, float],
) -> bytes:
    slices_info = build_slices_info(X, Ug, Vg)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (slice_pts, Ag, Bg, ia, ib, la, lb, slice_label), (mu_lo, mu_hi) in zip(
        axes, slices_info, mu_ranges, strict=True
    ):
        mu_slice, _ = best_gp.predict(slice_pts, return_std=True)
        mu_slice = mu_slice.reshape(Ug.shape)
        levels_mu = np.linspace(mu_lo, mu_hi, CONTOURF_LEVELS)
        cf = ax.contourf(Ag, Bg, mu_slice, levels=levels_mu, cmap="viridis")
        ax.contour(
            Ag,
            Bg,
            mu_slice,
            levels=np.linspace(mu_lo, mu_hi, CONTOUR_LINE_LEVELS),
            colors="k",
            linewidths=CONTOUR_LINEWIDTH,
            alpha=CONTOUR_ALPHA,
            linestyles=CONTOUR_LINESTYLE,
        )
        plt.colorbar(cf, ax=ax, shrink=0.82).set_label(r"$\mu(\mathbf{x})$")
        ax.scatter(
            X[:, ia],
            X[:, ib],
            c="red",
            s=55,
            edgecolors="k",
            zorder=4,
            linewidths=1.2,
        )
        for j in range(len(X)):
            ax.annotate(
                str(j + 1),
                (float(X[j, ia]), float(X[j, ib])),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=7,
                color="white",
                zorder=5,
            )
        ax.set_xlabel(la)
        ax.set_ylabel(lb)
        ax.set_title(f"{best_name} — mean ({slice_label})", fontsize=10)
        ax.set_aspect("equal")

    fig.suptitle(
        f"Function 3 — GP posterior mean (pairwise slices)  |  "
        f"n = {n_obs} obs  |  BO rounds after warm-start: {bo_rounds_done}  |  warping: {warp_label}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
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
        default=550,
        help="Display duration per frame in milliseconds",
    )
    parser.add_argument("--dpi", type=float, default=110.0, help="Rasterisation DPI")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(13.0, 4.2),
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

    X_full, y_orig = load_problem_data_csv(args.csv)
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

    y_warped, warp_params, _msg = apply_output_warping(y_orig, mode=OUTPUT_WARPING)
    warp_label = warp_params[0] if warp_params else "None"

    Ug, Vg, _ = _slice_grids(50)

    # Global μ colour limits per slice row (stable across animation)
    mu_ranges_global = mu_ranges_for_prefix(
        X_full,
        y_warped,
        Ug=Ug,
        Vg=Vg,
        gp_kernel=GP_KERNEL,
        optimize_kernel=OPTIMIZE_KERNEL,
        n_restarts_kernel=N_RESTARTS_KERNEL,
    )

    frames: list[Image.Image] = []
    for end in range(n_init, len(X_full) + 1):
        X = X_full[:end]
        y = y_warped[:end]
        best_gp, best_name = fit_best_gp(
            X,
            y,
            gp_kernel=GP_KERNEL,
            optimize_kernel=OPTIMIZE_KERNEL,
            n_restarts_kernel=N_RESTARTS_KERNEL,
        )
        bo_rounds = max(0, end - n_init)
        png = render_mean_row_png_bytes(
            best_gp,
            best_name,
            X,
            Ug=Ug,
            Vg=Vg,
            mu_ranges=mu_ranges_global,
            warp_label=warp_label,
            n_obs=end,
            bo_rounds_done=bo_rounds,
            dpi=args.dpi,
            figsize=(args.figsize[0], args.figsize[1]),
        )
        frames.append(Image.open(io.BytesIO(png)).convert("RGB"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(20, int(args.duration_ms))
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {len(frames)} frames -> {args.output}")


if __name__ == "__main__":
    main()
