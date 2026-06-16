import tempfile
from pathlib import Path
import unittest

from nesy_gen.models.blip_report_generator import BlipGeneratorConfig, save_blip_config


class BlipConfigTest(unittest.TestCase):
    def test_save_blip_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            save_blip_config(path, BlipGeneratorConfig(model_name="toy/blip"))
            self.assertTrue(path.exists())
            self.assertIn("toy/blip", path.read_text())


if __name__ == "__main__":
    unittest.main()

