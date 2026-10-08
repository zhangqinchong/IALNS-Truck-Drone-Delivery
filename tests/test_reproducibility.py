import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from generate_instances import exposure_rows, generate, read_excel
from run_ialns import build_problem, load_engine, solve
from run_experiments import jobs_for


class ReproducibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = json.loads((ROOT / "settings" / "parameters.json").read_text(encoding="utf-8"))
        cls.metadata = json.loads((ROOT / "settings" / "generation_metadata.json").read_text(encoding="utf-8"))

    def test_frozen_source_and_inputs(self):
        self.assertEqual(hashlib.sha256((ROOT / "src" / "ialns.py").read_bytes()).hexdigest(),
                         self.settings["algorithm_source_sha256"])
        self.assertEqual(len(self.settings["input_catalog"]), 48)
        for case in self.settings["input_catalog"]:
            path = ROOT / case["input_file"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), case["sha256"], path.name)

    def test_ten_solver_seeds(self):
        self.assertEqual(self.settings["solver_seeds_in_order"],
            [104729, 130363, 155921, 181081, 206369, 231779, 257053, 282427, 307831, 333287])

    def test_experiment_plan(self):
        for suite, expected in (("main", 240), ("small", 60), ("sensitivity", 810)):
            jobs = list(jobs_for(suite, self.settings))
            self.assertEqual(len(jobs), expected)
            self.assertEqual(len({j["id"] for j in jobs}), expected)
            self.assertTrue(all((ROOT / j["input_file"]).exists() for j in jobs))

    def test_102_generator(self):
        for case in self.metadata["base_102"]["instances"]:
            generated, seed = generate(case["family"], case["size"])
            self.assertEqual(seed, case["seed"])
            self.assertEqual(generated, read_excel(ROOT / "instances" / "base" / case["output_file"]))

    def test_nested_exposure_generator(self):
        for design in self.metadata["exposure_ratio"]["designs"]:
            for ratio in self.metadata["exposure_ratio"]["levels"]:
                name = f"{Path(design['base_instance']).stem}_exposure_ratio_{ratio:.1f}_tight.xlsx"
                self.assertEqual(exposure_rows(design, ratio), read_excel(ROOT / "instances" / "exposure_ratio" / name))

    def test_scale_once_and_full_defaults(self):
        module = load_engine()
        data = build_problem(module, ROOT / "instances" / "base" / "c101改10.xlsx")
        self.assertEqual(data.node_xy[0], (24.0, 30.0))
        self.assertEqual(data.node_xy[data.depot_end_id], (24.0, 30.0))
        self.assertEqual(data.depot_end_id, 11)
        self.assertTrue(data.cfg.allow_launch_during_truck_ready_wait)
        self.assertTrue(data.cfg.exposure_uses_service_time)
        self.assertEqual(len(module.ALNSOptimizer(data, seed=104729).operator_stats), 21)

    def test_small_full_iteration_regression(self):
        payload = solve(ROOT / "instances" / "base" / "c101改5.xlsx", seed=104729)
        self.assertTrue(payload["feasible"])
        self.assertEqual(len(payload["objective_history"]), 4500)
        self.assertTrue(math.isclose(payload["objective"], 216.1790306306536, rel_tol=0, abs_tol=1e-7))
        self.assertEqual(payload["truck_route"], [0, 5, 3, 4, 2, 1, 6])


if __name__ == "__main__":
    unittest.main()
