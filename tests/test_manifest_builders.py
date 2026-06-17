import tempfile
from pathlib import Path
import json
import unittest

import pandas as pd

from nesy_gen.data.manifests import build_iuxray_manifest, build_r2gen_iuxray_manifest
from nesy_gen.data.schema import load_jsonl


class ManifestBuildersTest(unittest.TestCase):
    def test_build_iuxray_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            (image_dir / "1.png").write_text("not-an-image", encoding="utf-8")
            (image_dir / "1b.png").write_text("not-an-image", encoding="utf-8")
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
                [
                    {"uid": 1, "filename": "1.png", "projection": "Frontal"},
                    {"uid": 1, "filename": "1b.png", "projection": "Frontal"},
                ]
            ).to_csv(root / "indiana_projections.csv", index=False)

            out = root / "manifest.jsonl"
            examples = build_iuxray_manifest(root, out)
            loaded = load_jsonl(out)
            self.assertEqual(len(examples), 1)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].study_id, "iu_1")

    def test_build_r2gen_iuxray_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "iuxray"
            images = root / "images"
            images.mkdir(parents=True)
            (images / "a.png").write_text("not-an-image", encoding="utf-8")
            (images / "b.png").write_text("not-an-image", encoding="utf-8")
            annotation = {
                "train": [
                    {
                        "id": "1",
                        "report": "The lungs are clear.",
                        "image_path": ["a.png", "b.png"],
                    }
                ],
                "val": [],
                "test": [],
            }
            (root / "annotation.json").write_text(json.dumps(annotation), encoding="utf-8")

            out = root / "manifest.jsonl"
            examples = build_r2gen_iuxray_manifest(root, out)
            loaded = load_jsonl(out)

            self.assertEqual(len(examples), 2)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].study_id, "r2gen_1_0")
            self.assertTrue(loaded[0].image_path.endswith("a.png"))


if __name__ == "__main__":
    unittest.main()
