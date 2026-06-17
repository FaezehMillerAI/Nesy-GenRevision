import unittest

import pandas as pd

from nesy_gen.evaluation.clinical_metrics import (
    chexpert_lite_frame,
    chexpert_lite_labels,
    radgraph_lite_frame,
)
from nesy_gen.kg.entity_linking import LexicalEntityLinker


class ClinicalMetricsTest(unittest.TestCase):
    def test_chexpert_lite_detects_positive_and_negated_findings(self):
        labels = chexpert_lite_labels("No pleural effusion. There is cardiomegaly.")

        self.assertEqual(labels["pleural_effusion"], 0)
        self.assertEqual(labels["cardiomegaly"], 1)

    def test_chexpert_lite_frame_reports_label_hallucination(self):
        frame = pd.DataFrame(
            [
                {
                    "study_id": "s1",
                    "prediction": "There is cardiomegaly.",
                    "reference": "The heart size is normal.",
                }
            ]
        )

        result = chexpert_lite_frame(frame)

        self.assertEqual(result.iloc[0]["positive_label_hallucination_rate"], 1.0)

    def test_radgraph_lite_frame(self):
        vocab = pd.DataFrame(
            [
                {
                    "node_id": "1",
                    "node_name": "pleural effusion",
                    "node_type": "effect/phenotype",
                    "alias": "pleural effusion",
                }
            ]
        )
        linker = LexicalEntityLinker(vocab)
        frame = pd.DataFrame(
            [
                {
                    "study_id": "s1",
                    "prediction": "no pleural effusion",
                    "reference": "no pleural effusion",
                }
            ]
        )

        result = radgraph_lite_frame(frame, linker)

        self.assertEqual(result.iloc[0]["radgraph_lite_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
