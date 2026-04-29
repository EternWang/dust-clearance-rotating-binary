from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from make_figures import radial_diagnostics, write_analysis_products  # noqa: E402
from simulate_dust import Params, init_primaries, run  # noqa: E402


class DustSimulationSmokeTest(unittest.TestCase):
    def test_small_run_produces_traceable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "sim_results.npz"
            params = Params(n_particles=120, steps=120, dt=0.02, seed=7, save_every=12)
            run(params, out_path)

            with np.load(out_path) as data:
                self.assertEqual(int(data["n_particles"]), 120)
                self.assertEqual(data["history_pos"].shape[0], 11)
                self.assertEqual(data["final_pos"].shape, (120, 2))
                self.assertGreaterEqual(float(data["survived"].mean()), 0.0)
                self.assertLessEqual(float(data["survived"].mean()), 1.0)

    def test_radial_diagnostics_and_summary_are_well_formed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            params = Params(n_particles=120, steps=120, dt=0.02, seed=11, save_every=12)
            out_path = Path(tmp) / "sim_results.npz"
            run(params, out_path)

            with np.load(out_path) as data:
                centers, survival_fraction, escaped_count, median_escape_time, escape_band = radial_diagnostics(
                    data["r0"],
                    data["survived"],
                    data["escaped_at_step"],
                    float(data["dt"]),
                    n_bins=12,
                )
                self.assertEqual(len(centers), 11)
                self.assertEqual(len(survival_fraction), 11)
                self.assertEqual(len(escaped_count), 11)

                import make_figures  # noqa: PLC0415

                original_output_dir = make_figures.OUTPUT_DIR
                try:
                    make_figures.OUTPUT_DIR = Path(tmp)
                    write_analysis_products(
                        centers,
                        survival_fraction,
                        escaped_count,
                        median_escape_time,
                        escape_band,
                        data["survived"],
                        data["escaped_at_step"],
                        float(data["dt"]),
                        float(data["M1"]),
                        float(data["M2"]),
                        np.linalg.norm(data["history_star2_pos"] - data["history_star1_pos"], axis=1),
                    )
                finally:
                    make_figures.OUTPUT_DIR = original_output_dir

            summary = json.loads((Path(tmp) / "analysis_summary.json").read_text(encoding="utf-8"))
            self.assertIn("overall_survival_fraction", summary)
            self.assertIn("mean_binary_separation", summary)

    def test_binary_initialization_preserves_barycenter(self) -> None:
        params = Params()
        star1_pos, _, star2_pos, _ = init_primaries(params)
        barycenter = (params.M1 * star1_pos + params.M2 * star2_pos) / (params.M1 + params.M2)
        np.testing.assert_allclose(barycenter, np.zeros(2), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
