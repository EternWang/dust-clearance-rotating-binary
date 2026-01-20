"""Generate figures from dust-clearance simulation output.

Input
-----
- outputs/sim_results.npz

Output
------
- figures/survival_fraction_vs_initial_radius.png
- figures/final_positions_rotating_frame.png

Usage
-----
python src/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = REPO_ROOT / "outputs" / "sim_results.npz"
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def rotation_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def omega(G: float, M1: float, M2: float, D: float) -> float:
    return float(np.sqrt(G * (M1 + M2) / D ** 3))


def main() -> None:
    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"{NPZ_PATH} not found. Run: python src/simulate_dust.py"
        )

    dat = np.load(NPZ_PATH)
    r0 = dat["r0"]
    final_pos = dat["final_pos"]
    survived = dat["survived"]
    t_end = float(dat["t_end"])

    G = float(dat["G"])
    M1 = float(dat["M1"])
    M2 = float(dat["M2"])
    D = float(dat["D"])

    w = omega(G, M1, M2, D)

    # --- Figure 1: survival fraction vs initial radius ---
    bins = np.linspace(r0.min(), r0.max(), 40)
    idx = np.digitize(r0, bins)
    centers = 0.5 * (bins[:-1] + bins[1:])

    frac = np.empty(len(centers))
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() == 0:
            frac[b - 1] = np.nan
        else:
            frac[b - 1] = survived[m].mean()

    fig1 = plt.figure(figsize=(8, 4.8))
    ax1 = fig1.add_subplot(111)
    ax1.plot(centers, frac, marker="o", linestyle="-")
    ax1.set_title("Survival fraction vs initial orbital radius")
    ax1.set_xlabel("Initial radius r₀ (nondimensional)")
    ax1.set_ylabel("Fraction surviving (r < max_radius at t_end)")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(FIG_DIR / "survival_fraction_vs_initial_radius.png", dpi=200)
    plt.close(fig1)

    # --- Figure 2: final positions in rotating frame (co-rotating with the binary) ---
    # Rotate by -w t_end so the stars are approximately fixed on the x-axis.
    R = rotation_matrix(-w * t_end)
    final_rot = (R @ final_pos.T).T

    # star locations at t=0 (in rotating frame)
    r1 = D * M2 / (M1 + M2)
    r2 = D * M1 / (M1 + M2)
    s1 = np.array([-r1, 0.0])
    s2 = np.array([+r2, 0.0])

    fig2 = plt.figure(figsize=(6.4, 6.4))
    ax2 = fig2.add_subplot(111)
    ax2.scatter(final_rot[survived, 0], final_rot[survived, 1], s=2, alpha=0.6, label="survivors")
    ax2.scatter([s1[0], s2[0]], [s1[1], s2[1]], marker="x", s=80, label="primaries")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title("Final dust positions in the rotating frame")
    ax2.set_xlabel("x (rotating frame)")
    ax2.set_ylabel("y (rotating frame)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "final_positions_rotating_frame.png", dpi=200)
    plt.close(fig2)


if __name__ == "__main__":
    main()
