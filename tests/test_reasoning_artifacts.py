import tempfile
from pathlib import Path
import unittest

from nesy_gen.evaluation.reasoning import reasoning_coverage_frame, reasoning_score_frame, save_reasoning_artifacts


class ReasoningArtifactsTest(unittest.TestCase):
    def test_save_reasoning_artifacts(self):
        rows = [
            {
                "study_id": "s1",
                "image_path": "x.png",
                "split": "test",
                "num_links": 1,
                "clause_scores": {
                    "bio_temporal": 1.0,
                    "finding_to_diagnosis": 1.0,
                    "located_in_type": 1.0,
                    "mean": 1.0,
                },
                "linked_entities": [
                    {
                        "node_name": "opacity",
                        "node_id": "1",
                        "node_type": "effect/phenotype",
                        "negated": False,
                    }
                ],
                "metadata": {},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_reasoning_artifacts(rows, tmp, prefix="toy")
            self.assertTrue(Path(paths["reasoning"]).exists())
            self.assertEqual(reasoning_score_frame(rows).iloc[0]["mean"], 1.0)
            self.assertEqual(reasoning_coverage_frame(rows).iloc[0]["num_positive"], 1)


if __name__ == "__main__":
    unittest.main()

