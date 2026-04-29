# Dust Clearance in a Rotating Binary

A reproducible Python project built from a UCSB physics final project on
**resonance-driven dust clearing** in a rotating binary system.

This repository is structured as a small simulation-and-analysis codebase rather
than a report dump: it integrates the binary and dust with RK4, generates the
raw simulation output, derives binned stability diagnostics, and exports both
static figures and a GitHub-friendly animation.
It is also organized as a compact computational replication package: simulation
parameters, generated arrays, derived CSV/JSON summaries, figures, and report artifacts
are linked through rerunnable code.

## What this repository demonstrates

- vectorized **NumPy** simulation of thousands of test particles
- a custom fixed-step **RK4** integrator for a coupled binary-plus-dust system
- derived **data products** (`CSV` and `JSON`) from simulation output
- scientific visualization with **matplotlib**
- animated result generation for a portfolio-ready **README GIF**
- a reproducible workflow that connects code, figures, and report artifacts
- lightweight smoke tests for the numerical setup and derived diagnostics

## Physical model

- Two primaries are included in the dynamical state and move under their
  **mutual gravitational attraction**
- Dust particles are treated as **massless tracers**
- Dust feels the gravity of both primaries, but the dust does **not** back-react
  on the binary
- Default mass ratio: **`M1:M2 = 3:1`**
- Initial binary separation: **`D = 1`** in nondimensional units

## Key results

For the default run (`n_particles=2000`, `steps=2000`, `dt=0.02`, `seed=42`):

| Quantity | Result |
|---|---:|
| Mass ratio | `M1/M2 = 3.0` |
| Particles simulated | `2000` |
| Surviving particles at `t_end` | `479` |
| Escaped particles at `t_end` | `1521` |
| Overall survival fraction | `0.2395` |
| Median escape time | `4.76` |
| Most unstable initial radius bin | `r0 ~= 1.711` |
| Minimum survival fraction | `0.0000` |
| Mean binary separation | `0.99999985` |
| Binary separation std. dev. | `7.33e-08` |

Interpretation:

- Stability depends strongly on the particle's **initial orbital radius**
- The deepest depletion in this run occurs near **`r0 ~= 1.71`**
- The system shows **radial "danger zones"** consistent with resonance-driven clearing
- The binary separation stays essentially fixed, which is what we expect for a
  correctly initialized circular two-body orbit integrated with a small RK4 step

## Why RK4 here?

This project uses a classic fourth-order Runge-Kutta step to update the
positions and velocities of:

- primary 1
- primary 2
- every dust particle

At each timestep, RK4 evaluates the derivatives four times:

```text
k1 = f(y_n)
k2 = f(y_n + 0.5 dt k1)
k3 = f(y_n + 0.5 dt k2)
k4 = f(y_n + dt k3)

y_{n+1} = y_n + (dt/6) (k1 + 2k2 + 2k3 + k4)
```

Why it helps here:

- it is much more accurate than a simple Euler update at the same timestep
- it handles the time-dependent gravitational field from the moving primaries well
- it is still easy to read and explain in a portfolio project

## Orbit evolution (README animation)

The animation below is generated from downsampled trajectory snapshots saved by
the RK4 integrator. Unlike the earlier fixed-orbit version of the project, both
primaries now move explicitly in the animation.

![Orbit evolution GIF](figures/orbit_evolution.gif)

## Static analysis outputs

### Instability diagnostics

This figure combines two analysis views:

- top: survival fraction vs initial radius
- bottom: median escape time of escaped particles vs initial radius

![Instability diagnostics](figures/instability_diagnostics.png)

### Final positions in the rotating frame

![Final positions](figures/final_positions_rotating_frame.png)

## Data products written by the pipeline

The repository now writes structured outputs in addition to figures:

- `outputs/sim_results.npz`
  downsampled trajectory history, final positions, survival mask, escape times,
  and run parameters
- `outputs/radial_diagnostics.csv`
  radius-binned survival and escape-time summary
- `outputs/analysis_summary.json`
  headline metrics suitable for reports or dashboards

## Reproduce the analysis

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/simulate_dust.py --n_particles 2000 --steps 2000 --dt 0.02 --seed 42 --save_every 20
python src/make_figures.py
python src/make_animation.py --max_particles 800 --fps 12 --dpi 110
```

If `make` is available:

```bash
make all
make animation
```

Smoke test:

```bash
python -m unittest discover -s tests
```

Generated assets are written to:

- `figures/survival_fraction_vs_initial_radius.png`
- `figures/instability_diagnostics.png`
- `figures/final_positions_rotating_frame.png`
- `figures/orbit_evolution.gif`

## Repository structure

```text
.
|-- src/
|   |-- simulate_dust.py
|   |-- make_figures.py
|   `-- make_animation.py
|-- outputs/
|   |-- sim_results.npz
|   |-- radial_diagnostics.csv
|   `-- analysis_summary.json
|-- figures/
|   |-- survival_fraction_vs_initial_radius.png
|   |-- instability_diagnostics.png
|   |-- final_positions_rotating_frame.png
|   `-- orbit_evolution.gif
|-- report/
|   |-- report.tex
|   `-- final_project_129_updated.pdf
`-- summary/
    |-- one_page_summary.tex
    `-- one_page_summary.pdf
```

## Project origin

This repository is adapted from a UCSB physics course final project. The goal of
the GitHub version is to present the work as a **reproducible numerical-analysis
project** with clear data products, visual diagnostics, and readable Python.

The PDF in `report/` is the original course artifact. The GitHub version of the
project extends that baseline by integrating the binary self-consistently with
RK4 and by adding analysis products and animation assets for portfolio use.

## Modeling notes

This is a simplified numerical experiment, not a full astrophysical disk model. Dust
particles are massless tracers; close encounters are softened; gas drag, collisions,
radiation pressure, and particle-size distributions are outside the scope of the
repository. The value of the project is the transparent simulation workflow and the
diagnostic treatment of where particles remain stable or clear out.

## Attribution

Author: Hongyu Wang.  
Course context: UCSB Physics final project; instructor David Berenstein.
