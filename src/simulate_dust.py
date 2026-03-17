"""Dust clearance simulation in a rotating binary.

This script evolves a swarm of massless test particles in the gravitational
field of two primaries on a fixed circular orbit. In addition to the final
particle state, it stores downsampled trajectory snapshots and first-escape
times so the repository can generate both static diagnostics and a GitHub-
friendly animation.
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

    # Numerical settings.
    dt: float = 0.02
    steps: int = 2000
    n_particles: int = 2000
    seed: int = 42
    save_every: int = 20

    # Initial condition distribution.
    r_min: float = 0.5
    r_max: float = 2.0
    v_noise: float = 0.05

    # Escape and collision thresholds.
    max_radius: float = 3.0
    softening: float = 0.05


def omega(p: Params) -> float:
    return float(np.sqrt(p.G * (p.M1 + p.M2) / p.D**3))


def star_positions(t: float, p: Params) -> tuple[np.ndarray, np.ndarray]:
    """Return the primary positions at time t in the inertial frame."""
    w = omega(p)
    r1 = p.D * p.M2 / (p.M1 + p.M2)
    r2 = p.D * p.M1 / (p.M1 + p.M2)

    c = np.cos(w * t)
    s = np.sin(w * t)

    pos1 = np.array([-r1 * c, -r1 * s])
    pos2 = np.array([+r2 * c, +r2 * s])
    return pos1, pos2


def acceleration(pos: np.ndarray, t: float, p: Params) -> np.ndarray:
    """Compute accelerations for all particles at time t."""
    s1, s2 = star_positions(t, p)

    r1 = s1[None, :] - pos
    r2 = s2[None, :] - pos

    d1 = np.maximum(np.linalg.norm(r1, axis=1), p.softening)
    d2 = np.maximum(np.linalg.norm(r2, axis=1), p.softening)

    return p.G * (p.M1 * r1 / d1[:, None] ** 3 + p.M2 * r2 / d2[:, None] ** 3)


def rk4_step(
    pos: np.ndarray, vel: np.ndarray, t: float, p: Params
) -> tuple[np.ndarray, np.ndarray]:
    """Advance every particle by one RK4 step."""
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
    """Sample initial radii, phases, and nearly circular orbital speeds."""
    rng = np.random.default_rng(p.seed)

    r0 = rng.uniform(p.r_min, p.r_max, p.n_particles)
    theta = rng.uniform(0.0, 2 * np.pi, p.n_particles)

    pos = np.column_stack([r0 * np.cos(theta), r0 * np.sin(theta)])

    v_kep = np.sqrt(p.G * (p.M1 + p.M2) / r0)
    vx = -v_kep * np.sin(theta) + rng.normal(0.0, p.v_noise, p.n_particles)
    vy = +v_kep * np.cos(theta) + rng.normal(0.0, p.v_noise, p.n_particles)
    vel = np.column_stack([vx, vy])

    return r0, pos, vel


def run(p: Params, out_path: Path) -> None:
    """Run the simulation and write results to a compressed numpy archive."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    r0, pos, vel = init_particles(p)
    survived = np.ones(p.n_particles, dtype=bool)
    escaped_at_step = np.full(p.n_particles, -1, dtype=int)

    history_pos = [pos.copy()]
    history_times = [0.0]
    history_steps = [0]

    t = 0.0
    for step in range(1, p.steps + 1):
        pos, vel = rk4_step(pos, vel, t, p)
        t = step * p.dt

        radii = np.linalg.norm(pos, axis=1)
        newly_escaped = (escaped_at_step < 0) & (radii > p.max_radius)
        escaped_at_step[newly_escaped] = step
        survived[newly_escaped] = False

        should_save = p.save_every > 0 and (
            step % p.save_every == 0 or step == p.steps
        )
        if should_save:
            history_pos.append(pos.copy())
            history_times.append(t)
            history_steps.append(step)

    np.savez_compressed(
        out_path,
        r0=r0,
        final_pos=pos,
        final_vel=vel,
        survived=survived,
        escaped_at_step=escaped_at_step,
        history_pos=np.stack(history_pos),
        history_times=np.array(history_times),
        history_steps=np.array(history_steps),
        t_end=t,
        dt=p.dt,
        steps=p.steps,
        n_particles=p.n_particles,
        seed=p.seed,
        save_every=p.save_every,
        M1=p.M1,
        M2=p.M2,
        D=p.D,
        G=p.G,
        max_radius=p.max_radius,
    )


def parse_args() -> Params:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_particles", type=int, default=Params.n_particles)
    parser.add_argument("--steps", type=int, default=Params.steps)
    parser.add_argument("--dt", type=float, default=Params.dt)
    parser.add_argument("--seed", type=int, default=Params.seed)
    parser.add_argument("--max_radius", type=float, default=Params.max_radius)
    parser.add_argument("--save_every", type=int, default=Params.save_every)
    args = parser.parse_args()

    return Params(
        dt=args.dt,
        steps=args.steps,
        n_particles=args.n_particles,
        seed=args.seed,
        max_radius=args.max_radius,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    params = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    run(params, repo_root / "outputs" / "sim_results.npz")
