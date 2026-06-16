import tempfile
from pathlib import Path
import unittest

import pandas as pd

from nesy_gen.data.manifests import build_iuxray_manifest
from nesy_gen.data.schema import load_jsonl


class ManifestBuildersTest(unittest.TestCase):
    def test_build_iuxray_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            (image_dir / "1.png").write_text("not-an-image", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "uid": 1,
                        "MeSH": "normal",
                        "Problems": "normal",
                        "image": "Chest",
                        "indication": "cough",
                        "comparison": "",
                        "findings": "No edema.",
                        "impression": "Normal chest.",
                    }
                ]
            ).to_csv(root / "indiana_reports.csv", index=False)
            pd.DataFrame(
                [{"uid": 1, "filename": "1.png", "projection": "Frontal"}]
            ).to_csv(root / "indiana_projections.csv", index=False)

            out = root / "manifest.jsonl"
            examples = build_iuxray_manifest(root, out)
            loaded = load_jsonl(out)
            self.assertEqual(len(examples), 1)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].study_id, "iu_1")


if __name__ == "__main__":
    unittest.main()

