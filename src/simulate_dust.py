"""Dust clearance simulation in a rotating binary (Sun–Jupiter analogue).

We simulate test particles moving under the gravitational field of two primaries
(M1=1, M2=3) on a fixed circular orbit of separation D=1, in nondimensional
units with G=1.

The goal is to visualize *resonance-driven clearing*: certain initial orbital
radii are unstable and become depleted over time.

Outputs
-------
- outputs/sim_results.npz : compressed numpy archive containing initial radii,
  final positions, survival mask, and run parameters.

Usage
-----
python src/simulate_dust.py --n_particles 5000 --steps 4000 --dt 0.02 --seed 42

The defaults are chosen to run quickly on a laptop while still showing
qualitative resonance gaps.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Params:
    G: float = 1.0
    M1: float = 1.0
    M2: float = 3.0
    D: float = 1.0

    # numerical settings
    dt: float = 0.02
    steps: int = 2000
    n_particles: int = 2000
    seed: int = 42

    # initial condition distribution
    r_min: float = 0.5
    r_max: float = 2.0
    v_noise: float = 0.05

    # escape/collision thresholds
    max_radius: float = 3.0
    softening: float = 0.05


def omega(p: Params) -> float:
    return float(np.sqrt(p.G * (p.M1 + p.M2) / p.D ** 3))


def star_positions(t: float, p: Params) -> tuple[np.ndarray, np.ndarray]:
    """Return (r1, r2) at time t in the inertial frame."""
    w = omega(p)
    # distances from COM
    r1 = p.D * p.M2 / (p.M1 + p.M2)  # 0.75
    r2 = p.D * p.M1 / (p.M1 + p.M2)  # 0.25

    c = np.cos(w * t)
    s = np.sin(w * t)

    # Choose counter-clockwise circular motion
    pos1 = np.array([-r1 * c, -r1 * s])
    pos2 = np.array([+r2 * c, +r2 * s])
    return pos1, pos2


def acceleration(pos: np.ndarray, t: float, p: Params) -> np.ndarray:
    """Compute acceleration for all particles at time t.

    pos: (N,2)
    returns: (N,2)
    """
    s1, s2 = star_positions(t, p)

    r1 = s1[None, :] - pos
    r2 = s2[None, :] - pos

    d1 = np.linalg.norm(r1, axis=1)
    d2 = np.linalg.norm(r2, axis=1)

    # softening to avoid singularities
    d1 = np.maximum(d1, p.softening)
    d2 = np.maximum(d2, p.softening)

    a = p.G * (p.M1 * r1 / d1[:, None] ** 3 + p.M2 * r2 / d2[:, None] ** 3)
    return a


def rk4_step(pos: np.ndarray, vel: np.ndarray, t: float, p: Params) -> tuple[np.ndarray, np.ndarray]:
    dt = p.dt

    k1v = acceleration(pos, t, p)
    k1x = vel

    k2v = acceleration(pos + 0.5 * dt * k1x, t + 0.5 * dt, p)
    k2x = vel + 0.5 * dt * k1v

    k3v = acceleration(pos + 0.5 * dt * k2x, t + 0.5 * dt, p)
    k3x = vel + 0.5 * dt * k2v

    k4v = acceleration(pos + dt * k3x, t + dt, p)
    k4x = vel + dt * k3v

    pos_next = pos + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    vel_next = vel + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)

    return pos_next, vel_next


def init_particles(p: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(p.seed)

    r0 = rng.uniform(p.r_min, p.r_max, p.n_particles)
    theta = rng.uniform(0.0, 2 * np.pi, p.n_particles)

    x = r0 * np.cos(theta)
    y = r0 * np.sin(theta)
    pos = np.column_stack([x, y])

    # Keplerian speed about total mass at radius r0 (about COM)
    v_kep = np.sqrt(p.G * (p.M1 + p.M2) / r0)
    vx = -v_kep * np.sin(theta) + rng.normal(0.0, p.v_noise, p.n_particles)
    vy = +v_kep * np.cos(theta) + rng.normal(0.0, p.v_noise, p.n_particles)
    vel = np.column_stack([vx, vy])

    return r0, pos, vel


def run(p: Params, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    r0, pos, vel = init_particles(p)

    survived = np.ones(p.n_particles, dtype=bool)

    t = 0.0
    for _ in range(p.steps):
        pos_next, vel_next = rk4_step(pos, vel, t, p)

        # mark escape
        radii = np.linalg.norm(pos_next, axis=1)
        escaped = radii > p.max_radius
        survived &= ~escaped

        # keep integrating everyone for simplicity (masking is optional)
        pos, vel = pos_next, vel_next
        t += p.dt

    np.savez_compressed(
        out_path,
        r0=r0,
        final_pos=pos,
        final_vel=vel,
        survived=survived,
        t_end=t,
        dt=p.dt,
        steps=p.steps,
        n_particles=p.n_particles,
        seed=p.seed,
        M1=p.M1,
        M2=p.M2,
        D=p.D,
        G=p.G,
        max_radius=p.max_radius,
    )


def parse_args() -> Params:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_particles", type=int, default=Params.n_particles)
    ap.add_argument("--steps", type=int, default=Params.steps)
    ap.add_argument("--dt", type=float, default=Params.dt)
    ap.add_argument("--seed", type=int, default=Params.seed)
    ap.add_argument("--max_radius", type=float, default=Params.max_radius)

    args = ap.parse_args()

    return Params(
        dt=args.dt,
        steps=args.steps,
        n_particles=args.n_particles,
        seed=args.seed,
        max_radius=args.max_radius,
    )


if __name__ == "__main__":
    params = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run(params, repo_root / "outputs" / "sim_results.npz")
