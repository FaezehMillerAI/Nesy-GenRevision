import unittest

from nesy_gen.models.chexagent import (
    build_chexagent_conversation,
    build_chexagent_prompt,
)


class _Tokenizer:
    def from_list_format(self, items):
        return f"IMAGE={items[0]['image']}\n{items[1]['text']}"


class CheXagentPromptTest(unittest.TestCase):
    def test_prompt_requests_narrative_findings_only(self):
        prompt = build_chexagent_prompt("cough")

        self.assertIn("narrative Findings section", prompt)
        self.assertIn("Clinical indication: cough", prompt)
        self.assertIn("Do not include headings, bullets, or an impression", prompt)

    def test_retrieval_examples_are_non_authoritative(self):
        prompt = build_chexagent_prompt("", ["No focal opacity."])

        self.assertIn("non-authoritative", prompt)
        self.assertIn("never copy an unsupported finding", prompt)
        self.assertIn("Example 1: No focal opacity.", prompt)

    def test_conversation_uses_image_and_optional_target(self):
        conversation = build_chexagent_conversation(
            "/tmp/xray.png",
            tokenizer=_Tokenizer(),
            target="  Lungs are clear.  ",
        )

        self.assertIn("IMAGE=/tmp/xray.png", conversation[1]["value"])
        self.assertEqual(conversation[2], {"from": "gpt", "value": "Lungs are clear."})


if __name__ == "__main__":
    unittest.main()
