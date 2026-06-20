import unittest

from nesy_gen.baselines.medsiglip_retrieval import _study_key
from nesy_gen.data.schema import RadiologyExample


class MedSiglipRetrievalContractTest(unittest.TestCase):
    def test_mimic_alternate_views_share_underlying_study_key(self):
        first = RadiologyExample(
            "mimic_s123_0", "/tmp/a.png", "", "", "train", {"study_id": "s123"}
        )
        second = RadiologyExample(
            "mimic_s123_1", "/tmp/b.png", "", "", "test", {"study_id": "s123"}
        )

        self.assertEqual(_study_key(first), _study_key(second))

    def test_r2gen_alternate_views_share_underlying_study_key(self):
        first = RadiologyExample(
            "r2gen_7_0", "/tmp/a.png", "", "", "train", {"r2gen_id": "7"}
        )
        second = RadiologyExample(
            "r2gen_7_1", "/tmp/b.png", "", "", "test", {"r2gen_id": "7"}
        )

        self.assertEqual(_study_key(first), _study_key(second))


if __name__ == "__main__":
    unittest.main()
