"""Create a GitHub-friendly GIF from the simulated dust trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = REPO_ROOT / "outputs" / "sim_results.npz"
GIF_PATH = REPO_ROOT / "figures" / "orbit_evolution.gif"


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
    history_star1_pos = dat["history_star1_pos"]
    history_star2_pos = dat["history_star2_pos"]
    history_times = dat["history_times"]
    history_steps = dat["history_steps"]
    escaped_at_step = dat["escaped_at_step"]

    M1 = float(dat["M1"])
    M2 = float(dat["M2"])
    max_radius = float(dat["max_radius"])

    n_particles = history_pos.shape[1]
    if n_particles > args.max_particles:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(n_particles, size=args.max_particles, replace=False))
        history_pos = history_pos[:, keep, :]
        escaped_at_step = escaped_at_step[keep]

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    dust = ax.scatter([], [], s=4, alpha=0.45, color="#1f77b4", edgecolors="none")
    star1 = ax.scatter([], [], s=160, color="#111111", label="M1 = 3")
    star2 = ax.scatter([], [], s=90, color="#666666", label="M2 = 1")

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
    ax.set_title("Dust clearing with self-consistent RK4 binary dynamics")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")

    def init() -> tuple[object, object, object, object]:
        dust.set_offsets(np.empty((0, 2)))
        star1.set_offsets(np.empty((0, 2)))
        star2.set_offsets(np.empty((0, 2)))
        time_text.set_text("")
        return dust, star1, star2, time_text

    def update(frame_idx: int) -> tuple[object, object, object, object]:
        active = (escaped_at_step < 0) | (escaped_at_step > history_steps[frame_idx])
        points = history_pos[frame_idx][active]
        if len(points) == 0:
            dust.set_offsets(np.empty((0, 2)))
        else:
            dust.set_offsets(points)

        star1.set_offsets(history_star1_pos[frame_idx][None, :])
        star2.set_offsets(history_star2_pos[frame_idx][None, :])
        time_text.set_text(f"t = {history_times[frame_idx]:.1f}")
        return dust, star1, star2, time_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(history_pos),
        init_func=init,
        interval=1000 / args.fps,
        blit=True,
    )

    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    animation.save(GIF_PATH, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
