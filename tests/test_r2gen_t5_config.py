import tempfile
from pathlib import Path
import unittest

from nesy_gen.models.r2gen_t5 import (
    R2GenT5Config,
    clean_r2gen_prediction,
    load_r2gen_t5_config,
    save_r2gen_t5_config,
)


class R2GenT5ConfigTest(unittest.TestCase):
    def test_save_and_load_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r2gen_t5_config.json"
            save_r2gen_t5_config(path, R2GenT5Config(text_model_name="toy/t5"))

            loaded = load_r2gen_t5_config(path)

            self.assertEqual(loaded.text_model_name, "toy/t5")

    def test_clean_prediction_removes_prefix(self):
        cleaned = clean_r2gen_prediction("generate report: lungs are clear.")

        self.assertEqual(cleaned, "lungs are clear.")


if __name__ == "__main__":
    unittest.main()
