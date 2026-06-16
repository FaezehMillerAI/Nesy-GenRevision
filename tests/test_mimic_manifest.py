import tempfile
from pathlib import Path
import unittest

import pandas as pd

from nesy_gen.data.manifests import build_mimic_aug_manifest, parse_list_cell
from nesy_gen.data.schema import load_jsonl


class MimicManifestTest(unittest.TestCase):
    def test_parse_list_cell(self):
        self.assertEqual(parse_list_cell("['a', 'b']"), ["a", "b"])
        self.assertEqual(parse_list_cell("[]"), [])
        self.assertEqual(parse_list_cell(float("nan")), [])

    def test_build_mimic_aug_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = root / "official_data_iccv_final" / "files" / "p10" / "p10000001" / "s50000001"
            files.mkdir(parents=True)
            image = files / "abc.jpg"
            image.write_text("not-an-image", encoding="utf-8")

            row = {
                "Unnamed: 0.1": 0,
                "Unnamed: 0": 0,
                "subject_id": 10000001,
                "image": "['files/p10/p10000001/s50000001/abc.jpg']",
                "view": "['PA']",
                "AP": "[]",
                "PA": "['files/p10/p10000001/s50000001/abc.jpg']",
                "Lateral": "[]",
                "text": "['Findings: No focal consolidation.']",
                "text_augment": "['Finds: No focus.']",
            }
            pd.DataFrame([row]).to_csv(root / "mimic_cxr_aug_train.csv", index=False)
            pd.DataFrame([row]).to_csv(root / "mimic_cxr_aug_validate.csv", index=False)

            out = root / "manifest.jsonl"
            examples = build_mimic_aug_manifest(root / "official_data_iccv_final", out)
            loaded = load_jsonl(out)
            self.assertEqual(len(examples), 2)
            self.assertEqual(len(loaded), 2)
            self.assertTrue(loaded[0].image_path.endswith("abc.jpg"))
            self.assertIn(loaded[1].split, {"val", "test"})


if __name__ == "__main__":
    unittest.main()

