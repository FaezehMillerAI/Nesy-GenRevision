import tempfile
from pathlib import Path
import unittest

import pandas as pd

from nesy_gen.data.schema import RadiologyExample
from nesy_gen.evaluation.entity_validation import write_entity_validation_bundle


class EntityValidationTest(unittest.TestCase):
    def test_write_entity_validation_bundle(self):
        examples = [
            RadiologyExample(
                study_id="s1",
                image_path=None,
                indication="",
                report="No pleural effusion. Heart size is normal.",
                split="test",
            )
        ]
        vocab = pd.DataFrame(
            [
                {
                    "node_id": "1",
                    "node_name": "pleural effusion",
                    "node_type": "effect/phenotype",
                    "alias": "pleural effusion",
                },
                {
                    "node_id": "2",
                    "node_name": "normal",
                    "node_type": "effect/phenotype",
                    "alias": "normal",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_entity_validation_bundle(
                examples,
                vocab,
                tmp,
                prefix="toy",
                audit_sample_size=10,
            )

            self.assertTrue(Path(paths["summary"]).exists())
            coverage = pd.read_csv(paths["coverage"])
            ablation = pd.read_csv(paths["linker_ablation"])
            self.assertEqual(coverage.iloc[0]["filtered_links"], 1)
            self.assertEqual(coverage.iloc[0]["raw_links"], 2)
            self.assertIn("normal", set(ablation["node_name"]))


if __name__ == "__main__":
    unittest.main()
