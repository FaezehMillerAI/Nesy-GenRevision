import tempfile
from pathlib import Path
import unittest

import pandas as pd

from nesy_gen.data.schema import RadiologyExample
from nesy_gen.evaluation.visualization import build_qualitative_html, save_qualitative_html, save_standard_plots


class VisualizationTest(unittest.TestCase):
    def test_build_qualitative_html(self):
        html = build_qualitative_html(
            examples=[
                RadiologyExample(
                    "s1",
                    "/tmp/image.png",
                    "",
                    "reference report",
                    "test",
                )
            ],
            predictions=pd.DataFrame([{"study_id": "s1", "prediction": "generated report"}]),
            run_name="toy",
            max_examples=5,
        )
        self.assertIn("generated report", html)
        self.assertIn("reference report", html)

    def test_save_qualitative_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = save_qualitative_html("<html></html>", Path(tmp) / "report.html")
            self.assertTrue(out.exists())

    def test_save_standard_plots_empty_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_standard_plots(output_dir=tmp, run_name="toy")
            self.assertEqual(paths, [])


if __name__ == "__main__":
    unittest.main()

