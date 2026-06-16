import tempfile
import json
from pathlib import Path
import unittest

import pandas as pd

from nesy_gen.evaluation.analysis import entity_frequency_frame, low_score_frame, score_bin_frame, write_analysis_bundle
from nesy_gen.evaluation.factuality import ReportPair, evaluate_report_pairs
from nesy_gen.kg.entity_linking import LexicalEntityLinker


class AnalysisFactualityTest(unittest.TestCase):
    def test_analysis_frames(self):
        rows = [
            {
                "study_id": "s1",
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
                        "node_id": "1",
                        "node_name": "opacity",
                        "node_type": "effect/phenotype",
                        "negated": False,
                    }
                ],
            }
        ]
        scores = pd.DataFrame(
            [
                {
                    "study_id": "s1",
                    "split": "test",
                    "num_links": 1,
                    "bio_temporal": 1.0,
                    "finding_to_diagnosis": 1.0,
                    "located_in_type": 1.0,
                    "mean": 1.0,
                }
            ]
        )
        self.assertEqual(entity_frequency_frame(rows).iloc[0]["count"], 1)
        self.assertEqual(low_score_frame(rows).iloc[0]["study_id"], "s1")
        self.assertFalse(score_bin_frame(scores).empty)

    def test_write_analysis_bundle(self):
        rows = [
            {
                "study_id": "s1",
                "split": "test",
                "num_links": 0,
                "clause_scores": {
                    "bio_temporal": 0.0,
                    "finding_to_diagnosis": 0.0,
                    "located_in_type": 0.0,
                    "mean": 0.0,
                },
                "linked_entities": [],
            }
        ]
        scores = pd.DataFrame(
            [
                {
                    "study_id": "s1",
                    "split": "test",
                    "num_links": 0,
                    "bio_temporal": 0.0,
                    "finding_to_diagnosis": 0.0,
                    "located_in_type": 0.0,
                    "mean": 0.0,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reasoning = root / "reasoning.json"
            reasoning.write_text(json.dumps(rows), encoding="utf-8")
            scores_path = root / "scores.csv"
            scores.to_csv(scores_path, index=False)
            paths = write_analysis_bundle(reasoning, scores_path, root, prefix="toy")
            self.assertTrue(Path(paths["summary"]).exists())

    def test_factuality_eval(self):
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
                    "node_name": "pneumonia",
                    "node_type": "disease",
                    "alias": "pneumonia",
                },
            ]
        )
        linker = LexicalEntityLinker(vocab)
        frame = evaluate_report_pairs(
            [ReportPair("s1", "pleural effusion", "no pleural effusion pneumonia")],
            linker,
        )
        self.assertEqual(frame.iloc[0]["unsupported_count"], 1.0)
        self.assertEqual(frame.iloc[0]["negation_mismatch_count"], 1)


if __name__ == "__main__":
    unittest.main()
