import tempfile
from pathlib import Path
import unittest

import pandas as pd

from nesy_gen.evaluation.official_metrics import (
    chexbert_label_metrics,
    official_radgraph_metrics,
    prepare_official_report_csv,
)


class OfficialMetricsTest(unittest.TestCase):
    def test_chexbert_label_metrics(self):
        pred = pd.DataFrame([{"study_id": "s1", "Cardiomegaly": 1, "Edema": 0}])
        ref = pd.DataFrame([{"study_id": "s1", "Cardiomegaly": 1, "Edema": 1}])

        scores = chexbert_label_metrics(pred, ref)

        self.assertEqual(scores.iloc[0]["precision"], 1.0)
        self.assertEqual(scores.iloc[0]["recall"], 0.5)
        self.assertEqual(scores.iloc[0]["label_mismatch_count"], 1)

    def test_official_radgraph_metrics(self):
        pred = {
            "s1": {
                "entities": {
                    "1": {
                        "label": "OBS",
                        "tokens": ["pleural", "effusion"],
                        "assertion": "definitely present",
                    }
                }
            }
        }
        ref = {
            "s1": {
                "entities": {
                    "1": {
                        "label": "OBS",
                        "tokens": ["pleural", "effusion"],
                        "assertion": "definitely present",
                    }
                }
            }
        }

        scores = official_radgraph_metrics(pred, ref)

        self.assertEqual(scores.iloc[0]["radgraph_f1"], 1.0)

    def test_prepare_official_report_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reports.csv"
            frame = pd.DataFrame([{"study_id": "s1", "prediction": "lungs clear"}])

            prepare_official_report_csv(frame, path, text_column="prediction")

            saved = pd.read_csv(path)
            self.assertEqual(saved.iloc[0]["Report Impression"], "lungs clear")


if __name__ == "__main__":
    unittest.main()
