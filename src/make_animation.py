"""Create a GitHub-friendly GIF from the simulated dust trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


REPO_ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = REPO_ROOT / "outputs" / "sim_results.npz"
GIF_PATH = REPO_ROOT / "figures" / "orbit_evolution.gif"


def rotation_matrix(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def omega(G: float, M1: float, M2: float, D: float) -> float:
    return float(np.sqrt(G * (M1 + M2) / D**3))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_particles", type=int, default=800)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"{NPZ_PATH} not found. Run: python src/simulate_dust.py"
        )

    dat = np.load(NPZ_PATH)
    if "history_pos" not in dat:
        raise KeyError(
            "Simulation output does not contain trajectory history. "
            "Re-run python src/simulate_dust.py with a recent version of the script."
        )

    history_pos = dat["history_pos"]
    history_times = dat["history_times"]
    history_steps = dat["history_steps"]
    escaped_at_step = dat["escaped_at_step"]

    G = float(dat["G"])
    M1 = float(dat["M1"])
    M2 = float(dat["M2"])
    D = float(dat["D"])
    max_radius = float(dat["max_radius"])

    n_particles = history_pos.shape[1]
    if n_particles > args.max_particles:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(n_particles, size=args.max_particles, replace=False))
        history_pos = history_pos[:, keep, :]
        escaped_at_step = escaped_at_step[keep]

    w = omega(G, M1, M2, D)
    history_rot = np.empty_like(history_pos)
    for idx, time_value in enumerate(history_times):
        rotation = rotation_matrix(-w * float(time_value))
        history_rot[idx] = (rotation @ history_pos[idx].T).T

    r1 = D * M2 / (M1 + M2)
    r2 = D * M1 / (M1 + M2)

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    dust = ax.scatter([], [], s=4, alpha=0.45, color="#1f77b4", edgecolors="none")
    primaries = ax.scatter([-r1, r2], [0.0, 0.0], marker="x", s=90, color="black")
    _ = primaries

    time_text = ax.text(
        0.03,
        0.97,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Dust clearing in the co-rotating frame")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)

    def init() -> tuple[object, object]:
        dust.set_offsets(np.empty((0, 2)))
        time_text.set_text("")
        return dust, time_text

    def update(frame_idx: int) -> tuple[object, object]:
        active = (escaped_at_step < 0) | (escaped_at_step > history_steps[frame_idx])
        points = history_rot[frame_idx][active]
        if len(points) == 0:
            dust.set_offsets(np.empty((0, 2)))
        else:
            dust.set_offsets(points)
        time_text.set_text(f"t = {history_times[frame_idx]:.1f}")
        return dust, time_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(history_rot),
        init_func=init,
        interval=1000 / args.fps,
        blit=True,
    )

    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    animation.save(GIF_PATH, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
