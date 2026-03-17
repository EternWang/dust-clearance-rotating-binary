"""Dust clearance simulation in a rotating binary.

This version integrates the full binary-plus-dust state with RK4:

- the two primaries move under their mutual gravitational attraction
- the dust particles are massless tracers that feel both primaries
- the dust does not back-react on the primaries

The default mass ratio is M1:M2 = 3:1.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Params:
    G: float = 1.0
    M1: float = 3.0
    M2: float = 1.0
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
    """Angular speed of the corresponding circular two-body solution."""
    return float(np.sqrt(p.G * (p.M1 + p.M2) / p.D**3))


def init_primaries(p: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize the two primaries on a circular orbit about the barycenter."""
    r1 = p.D * p.M2 / (p.M1 + p.M2)
    r2 = p.D * p.M1 / (p.M1 + p.M2)
    w = omega(p)

    star1_pos = np.array([-r1, 0.0])
    star2_pos = np.array([+r2, 0.0])

    # Tangential velocities for a circular two-body orbit.
    star1_vel = np.array([0.0, -w * r1])
    star2_vel = np.array([0.0, +w * r2])
    return star1_pos, star1_vel, star2_pos, star2_vel


def init_particles(p: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample initial dust radii, phases, and nearly circular speeds."""
    rng = np.random.default_rng(p.seed)

    r0 = rng.uniform(p.r_min, p.r_max, p.n_particles)
    theta = rng.uniform(0.0, 2 * np.pi, p.n_particles)

    pos = np.column_stack([r0 * np.cos(theta), r0 * np.sin(theta)])

    # Approximate barycentric circular speed plus a small perturbation.
    v_kep = np.sqrt(p.G * (p.M1 + p.M2) / r0)
    vx = -v_kep * np.sin(theta) + rng.normal(0.0, p.v_noise, p.n_particles)
    vy = +v_kep * np.cos(theta) + rng.normal(0.0, p.v_noise, p.n_particles)
    vel = np.column_stack([vx, vy])

    return r0, pos, vel


def primary_accelerations(
    star1_pos: np.ndarray, star2_pos: np.ndarray, p: Params
) -> tuple[np.ndarray, np.ndarray]:
    """Accelerations of the two primaries due only to each other."""
    delta = star2_pos - star1_pos
    dist = max(float(np.linalg.norm(delta)), p.softening)

    star1_acc = p.G * p.M2 * delta / dist**3
    star2_acc = -p.G * p.M1 * delta / dist**3
    return star1_acc, star2_acc


def dust_acceleration(
    dust_pos: np.ndarray, star1_pos: np.ndarray, star2_pos: np.ndarray, p: Params
) -> np.ndarray:
    """Acceleration of each dust particle from the two moving primaries."""
    rel1 = star1_pos[None, :] - dust_pos
    rel2 = star2_pos[None, :] - dust_pos

    d1 = np.maximum(np.linalg.norm(rel1, axis=1), p.softening)
    d2 = np.maximum(np.linalg.norm(rel2, axis=1), p.softening)

    return p.G * (p.M1 * rel1 / d1[:, None] ** 3 + p.M2 * rel2 / d2[:, None] ** 3)


def state_derivatives(
    star1_pos: np.ndarray,
    star1_vel: np.ndarray,
    star2_pos: np.ndarray,
    star2_vel: np.ndarray,
    dust_pos: np.ndarray,
    dust_vel: np.ndarray,
    p: Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return time derivatives for the full binary-plus-dust state."""
    star1_acc, star2_acc = primary_accelerations(star1_pos, star2_pos, p)
    dust_acc = dust_acceleration(dust_pos, star1_pos, star2_pos, p)
    return star1_vel, star1_acc, star2_vel, star2_acc, dust_vel, dust_acc


def rk4_step(
    star1_pos: np.ndarray,
    star1_vel: np.ndarray,
    star2_pos: np.ndarray,
    star2_vel: np.ndarray,
    dust_pos: np.ndarray,
    dust_vel: np.ndarray,
    p: Params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance the full state by one fourth-order Runge-Kutta step."""
    dt = p.dt

    k1 = state_derivatives(star1_pos, star1_vel, star2_pos, star2_vel, dust_pos, dust_vel, p)

    k2 = state_derivatives(
        star1_pos + 0.5 * dt * k1[0],
        star1_vel + 0.5 * dt * k1[1],
        star2_pos + 0.5 * dt * k1[2],
        star2_vel + 0.5 * dt * k1[3],
        dust_pos + 0.5 * dt * k1[4],
        dust_vel + 0.5 * dt * k1[5],
        p,
    )

    k3 = state_derivatives(
        star1_pos + 0.5 * dt * k2[0],
        star1_vel + 0.5 * dt * k2[1],
        star2_pos + 0.5 * dt * k2[2],
        star2_vel + 0.5 * dt * k2[3],
        dust_pos + 0.5 * dt * k2[4],
        dust_vel + 0.5 * dt * k2[5],
        p,
    )

    k4 = state_derivatives(
        star1_pos + dt * k3[0],
        star1_vel + dt * k3[1],
        star2_pos + dt * k3[2],
        star2_vel + dt * k3[3],
        dust_pos + dt * k3[4],
        dust_vel + dt * k3[5],
        p,
    )

    star1_pos_next = star1_pos + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    star1_vel_next = star1_vel + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    star2_pos_next = star2_pos + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    star2_vel_next = star2_vel + (dt / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
    dust_pos_next = dust_pos + (dt / 6.0) * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])
    dust_vel_next = dust_vel + (dt / 6.0) * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5])

    return (
        star1_pos_next,
        star1_vel_next,
        star2_pos_next,
        star2_vel_next,
        dust_pos_next,
        dust_vel_next,
    )


def barycenter(star1_pos: np.ndarray, star2_pos: np.ndarray, p: Params) -> np.ndarray:
    """Instantaneous center of mass of the binary."""
    return (p.M1 * star1_pos + p.M2 * star2_pos) / (p.M1 + p.M2)


def run(p: Params, out_path: Path) -> None:
    """Run the simulation and save the trajectory and summary state."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    r0, dust_pos, dust_vel = init_particles(p)
    star1_pos, star1_vel, star2_pos, star2_vel = init_primaries(p)

    survived = np.ones(p.n_particles, dtype=bool)
    escaped_at_step = np.full(p.n_particles, -1, dtype=int)

    history_dust_pos = [dust_pos.copy()]
    history_star1_pos = [star1_pos.copy()]
    history_star2_pos = [star2_pos.copy()]
    history_times = [0.0]
    history_steps = [0]

    t = 0.0
    for step in range(1, p.steps + 1):
        (
            star1_pos,
            star1_vel,
            star2_pos,
            star2_vel,
            dust_pos,
            dust_vel,
        ) = rk4_step(star1_pos, star1_vel, star2_pos, star2_vel, dust_pos, dust_vel, p)

        t = step * p.dt
        com = barycenter(star1_pos, star2_pos, p)
        radii = np.linalg.norm(dust_pos - com[None, :], axis=1)
        newly_escaped = (escaped_at_step < 0) & (radii > p.max_radius)
        escaped_at_step[newly_escaped] = step
        survived[newly_escaped] = False

        if p.save_every > 0 and (step % p.save_every == 0 or step == p.steps):
            history_dust_pos.append(dust_pos.copy())
            history_star1_pos.append(star1_pos.copy())
            history_star2_pos.append(star2_pos.copy())
            history_times.append(t)
            history_steps.append(step)

    np.savez_compressed(
        out_path,
        r0=r0,
        final_pos=dust_pos,
        final_vel=dust_vel,
        survived=survived,
        escaped_at_step=escaped_at_step,
        history_pos=np.stack(history_dust_pos),
        history_star1_pos=np.stack(history_star1_pos),
        history_star2_pos=np.stack(history_star2_pos),
        history_times=np.array(history_times),
        history_steps=np.array(history_steps),
        final_star1_pos=star1_pos,
        final_star1_vel=star1_vel,
        final_star2_pos=star2_pos,
        final_star2_vel=star2_vel,
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
