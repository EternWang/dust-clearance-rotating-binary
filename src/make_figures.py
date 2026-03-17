"""Create figures and derived analysis products for the dust simulation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = REPO_ROOT / "outputs" / "sim_results.npz"
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = REPO_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def rotation_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def barycenter(star1_pos: np.ndarray, star2_pos: np.ndarray, M1: float, M2: float) -> np.ndarray:
    return (M1 * star1_pos + M2 * star2_pos) / (M1 + M2)


def to_rotating_frame(
    points: np.ndarray,
    star1_pos: np.ndarray,
    star2_pos: np.ndarray,
    M1: float,
    M2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate coordinates so the instantaneous binary axis lies on the x-axis."""
    com = barycenter(star1_pos, star2_pos, M1, M2)
    axis = star2_pos - star1_pos
    angle = float(np.arctan2(axis[1], axis[0]))
    rotation = rotation_matrix(-angle)
    points_rot = (rotation @ (points - com).T).T
    star1_rot = rotation @ (star1_pos - com)
    star2_rot = rotation @ (star2_pos - com)
    return points_rot, star1_rot, star2_rot


def radial_diagnostics(
    r0: np.ndarray,
    survived: np.ndarray,
    escaped_at_step: np.ndarray,
    dt: float,
    n_bins: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return binned survival and escape-time diagnostics."""
    bins = np.linspace(r0.min(), r0.max(), n_bins)
    centers = 0.5 * (bins[:-1] + bins[1:])
    idx = np.digitize(r0, bins)

    survival_fraction = np.full(len(centers), np.nan)
    escaped_count = np.zeros(len(centers), dtype=int)
    median_escape_time = np.full(len(centers), np.nan)
    lower_escape_time = np.full(len(centers), np.nan)
    upper_escape_time = np.full(len(centers), np.nan)

    escape_time = np.where(escaped_at_step >= 0, escaped_at_step * dt, np.nan)

    for b in range(1, len(bins)):
        mask = idx == b
        if not np.any(mask):
            continue

        survival_fraction[b - 1] = survived[mask].mean()

        escaped_times = escape_time[mask & ~survived]
        escaped_times = escaped_times[np.isfinite(escaped_times)]
        escaped_count[b - 1] = len(escaped_times)
        if len(escaped_times) == 0:
            continue

        median_escape_time[b - 1] = float(np.median(escaped_times))
        lower_escape_time[b - 1] = float(np.percentile(escaped_times, 25))
        upper_escape_time[b - 1] = float(np.percentile(escaped_times, 75))

    return (
        centers,
        survival_fraction,
        escaped_count,
        median_escape_time,
        np.column_stack([lower_escape_time, upper_escape_time]),
    )


def write_analysis_products(
    centers: np.ndarray,
    survival_fraction: np.ndarray,
    escaped_count: np.ndarray,
    median_escape_time: np.ndarray,
    escape_band: np.ndarray,
    survived: np.ndarray,
    escaped_at_step: np.ndarray,
    dt: float,
    M1: float,
    M2: float,
    binary_separation: np.ndarray,
) -> None:
    """Write CSV and JSON summaries for downstream use in README/reporting."""
    csv_path = OUTPUT_DIR / "radial_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "initial_radius_center",
                "survival_fraction",
                "escaped_count",
                "median_escape_time",
                "escape_time_p25",
                "escape_time_p75",
            ]
        )
        for idx, center in enumerate(centers):
            writer.writerow(
                [
                    f"{center:.6f}",
                    (
                        f"{survival_fraction[idx]:.6f}"
                        if np.isfinite(survival_fraction[idx])
                        else ""
                    ),
                    int(escaped_count[idx]),
                    (
                        f"{median_escape_time[idx]:.6f}"
                        if np.isfinite(median_escape_time[idx])
                        else ""
                    ),
                    (
                        f"{escape_band[idx, 0]:.6f}"
                        if np.isfinite(escape_band[idx, 0])
                        else ""
                    ),
                    (
                        f"{escape_band[idx, 1]:.6f}"
                        if np.isfinite(escape_band[idx, 1])
                        else ""
                    ),
                ]
            )

    finite_survival = np.isfinite(survival_fraction)
    min_index = int(np.nanargmin(survival_fraction)) if np.any(finite_survival) else None

    escaped_times = escaped_at_step[escaped_at_step >= 0] * dt
    summary = {
        "mass_ratio_M1_over_M2": float(M1 / M2),
        "overall_survival_fraction": float(survived.mean()),
        "escaped_fraction": float((~survived).mean()),
        "escaped_particles": int((~survived).sum()),
        "surviving_particles": int(survived.sum()),
        "mean_binary_separation": float(np.mean(binary_separation)),
        "binary_separation_std": float(np.std(binary_separation)),
        "median_escape_time": (
            float(np.median(escaped_times)) if len(escaped_times) else None
        ),
        "most_unstable_radius_center": (
            float(centers[min_index]) if min_index is not None else None
        ),
        "minimum_survival_fraction": (
            float(survival_fraction[min_index]) if min_index is not None else None
        ),
    }

    json_path = OUTPUT_DIR / "analysis_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"{NPZ_PATH} not found. Run: python src/simulate_dust.py"
        )

    dat = np.load(NPZ_PATH)
    r0 = dat["r0"]
    final_pos = dat["final_pos"]
    survived = dat["survived"]
    escaped_at_step = dat["escaped_at_step"]
    history_star1_pos = dat["history_star1_pos"]
    history_star2_pos = dat["history_star2_pos"]
    final_star1_pos = dat["final_star1_pos"]
    final_star2_pos = dat["final_star2_pos"]
    t_end = float(dat["t_end"])
    dt = float(dat["dt"])

    M1 = float(dat["M1"])
    M2 = float(dat["M2"])
    max_radius = float(dat["max_radius"])
    centers, survival_fraction, escaped_count, median_escape_time, escape_band = radial_diagnostics(
        r0, survived, escaped_at_step, dt
    )
    binary_separation = np.linalg.norm(history_star2_pos - history_star1_pos, axis=1)

    write_analysis_products(
        centers,
        survival_fraction,
        escaped_count,
        median_escape_time,
        escape_band,
        survived,
        escaped_at_step,
        dt,
        M1,
        M2,
        binary_separation,
    )

    # Figure 1: original survival curve, kept for continuity with the report.
    fig1 = plt.figure(figsize=(8, 4.8))
    ax1 = fig1.add_subplot(111)
    ax1.plot(centers, survival_fraction, marker="o", linewidth=1.8, markersize=4)
    ax1.set_title("Survival fraction vs initial orbital radius")
    ax1.set_xlabel("Initial radius r0 (nondimensional)")
    ax1.set_ylabel("Fraction surviving at t_end")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(FIG_DIR / "survival_fraction_vs_initial_radius.png", dpi=200)
    plt.close(fig1)

    # Figure 2: combined diagnostics for a portfolio-style repository overview.
    fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(8, 7.2), sharex=True)
    ax2.plot(centers, survival_fraction, color="#1f77b4", linewidth=2.0)
    ax2.scatter(centers, survival_fraction, color="#1f77b4", s=16)
    ax2.set_ylabel("Survival fraction")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, alpha=0.25)
    ax2.set_title("Radial stability diagnostics")

    valid_escape = np.isfinite(median_escape_time)
    ax3.plot(centers[valid_escape], median_escape_time[valid_escape], color="#d62728", linewidth=2.0)
    ax3.fill_between(
        centers[valid_escape],
        escape_band[valid_escape, 0],
        escape_band[valid_escape, 1],
        color="#d62728",
        alpha=0.2,
        label="25th-75th percentile",
    )
    ax3.scatter(centers[valid_escape], median_escape_time[valid_escape], color="#d62728", s=16)
    ax3.set_xlabel("Initial radius r0 (nondimensional)")
    ax3.set_ylabel("Median escape time")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper right")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "instability_diagnostics.png", dpi=200)
    plt.close(fig2)

    # Figure 3: final positions in the co-rotating frame.
    final_rot, star1_rot, star2_rot = to_rotating_frame(
        final_pos, final_star1_pos, final_star2_pos, M1, M2
    )

    fig3 = plt.figure(figsize=(6.6, 6.6))
    ax4 = fig3.add_subplot(111)
    ax4.scatter(
        final_rot[~survived, 0],
        final_rot[~survived, 1],
        s=2,
        alpha=0.10,
        color="#d62728",
        label="escaped by t_end",
    )
    ax4.scatter(
        final_rot[survived, 0],
        final_rot[survived, 1],
        s=3,
        alpha=0.55,
        color="#1f77b4",
        label="survivors",
    )
    ax4.scatter(
        [star1_rot[0], star2_rot[0]],
        [star1_rot[1], star2_rot[1]],
        marker="x",
        s=[140, 90],
        color="black",
        label="primaries",
    )
    ax4.set_xlim(-max_radius, max_radius)
    ax4.set_ylim(-max_radius, max_radius)
    ax4.set_aspect("equal", adjustable="box")
    ax4.set_title("Final dust positions in the instantaneous rotating frame")
    ax4.set_xlabel("x (rotating frame)")
    ax4.set_ylabel("y (rotating frame)")
    ax4.grid(True, alpha=0.25)
    ax4.legend(loc="upper right")
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / "final_positions_rotating_frame.png", dpi=200)
    plt.close(fig3)


if __name__ == "__main__":
    main()
