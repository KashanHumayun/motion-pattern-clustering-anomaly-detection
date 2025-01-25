from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from motion_pattern.pipeline import run_demo


class MotionPipelineSmokeTest(unittest.TestCase):
    def test_demo_pipeline_creates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            summary = run_demo(output_dir=output_dir, n_samples=120)
            self.assertIn("kmeans", summary)
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "motion_analysis.csv").exists())
            self.assertTrue((output_dir / "motion_patterns.png").exists())
            self.assertGreaterEqual(summary["anomaly_detection"]["anomaly_count"], 1)


if __name__ == "__main__":
    unittest.main()
