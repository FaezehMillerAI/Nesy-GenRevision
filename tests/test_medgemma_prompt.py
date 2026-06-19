import unittest

from nesy_gen.models.medgemma import build_medgemma_prompt


class MedGemmaPromptTest(unittest.TestCase):
    def test_zero_shot_prompt_has_indication_without_examples(self):
        prompt = build_medgemma_prompt("chest pain")

        self.assertIn("Clinical indication: chest pain", prompt)
        self.assertNotIn("Example 1", prompt)

    def test_few_shot_prompt_marks_retrieved_reports_as_non_authoritative(self):
        prompt = build_medgemma_prompt("", ["No focal opacity.", "Small effusion."])

        self.assertIn("visually similar training radiographs", prompt)
        self.assertIn("do not copy an unsupported finding", prompt)
        self.assertIn("Example 2: Small effusion.", prompt)


if __name__ == "__main__":
    unittest.main()
