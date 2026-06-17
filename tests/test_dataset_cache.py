from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nesy_gen.data.kaggle import populate_dataset_cache, resolve_kaggle_dataset


class DatasetCacheTests(unittest.TestCase):
    def test_resolves_cached_iuxray_without_kagglehub(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "cache"
            dataset_root = cache_root / "iuxray"
            dataset_root.mkdir(parents=True)
            (dataset_root / "indiana_reports.csv").write_text("uid,findings,impression\n", encoding="utf-8")
            (dataset_root / "indiana_projections.csv").write_text(
                "uid,filename,projection\n",
                encoding="utf-8",
            )

            paths = resolve_kaggle_dataset("iuxray", cache_root=cache_root)

            self.assertEqual(paths.source, "drive_cache")
            self.assertEqual(paths.dataset_root, dataset_root)
            self.assertEqual(paths.data_root, dataset_root)

    def test_populates_mimic_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "downloaded"
            source.mkdir()
            (source / "mimic_cxr_aug_train.csv").write_text("subject_id,text\n", encoding="utf-8")
            (source / "mimic_cxr_aug_validate.csv").write_text("subject_id,text\n", encoding="utf-8")

            cached_root = populate_dataset_cache("mimic_cxr", source, root / "cache")

            self.assertEqual(cached_root, root / "cache" / "mimic_cxr")
            self.assertTrue((cached_root / "mimic_cxr_aug_train.csv").exists())
            self.assertTrue((cached_root / "mimic_cxr_aug_validate.csv").exists())


if __name__ == "__main__":
    unittest.main()
