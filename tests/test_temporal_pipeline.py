import unittest

from scripts.run_demo import build_toy_pipeline


class TemporalPipelineTest(unittest.TestCase):
    def test_demo_pipeline_reasons(self):
        pipeline = build_toy_pipeline()
        links, audit = pipeline.reason("pneumonia", "right lower lobe consolidation")
        self.assertGreaterEqual(len(links), 2)
        self.assertGreater(audit.mean_satisfaction, 0.5)
        self.assertIn("P:consolidation", audit.valid_nodes)


if __name__ == "__main__":
    unittest.main()

