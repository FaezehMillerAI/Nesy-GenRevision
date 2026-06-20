import unittest

from nesy_gen.models.chexagent import (
    build_chexagent_conversation,
    build_chexagent_prompt,
    patch_chexagent_vision_forward,
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

    def test_current_siglip_api_uses_encoder_last_hidden_state(self):
        class _Output:
            last_hidden_state = "pre-layernorm-features"

            def __getitem__(self, index):
                return self.last_hidden_state

        class _Embeddings:
            def __call__(self, pixels):
                return f"embedded:{pixels}"

        class _Encoder:
            def __call__(self, *, inputs_embeds):
                self.inputs_embeds = inputs_embeds
                return _Output()

        class _Siglip:
            embeddings = _Embeddings()
            encoder = _Encoder()

        class _Visual:
            model = _Siglip()

            def forward_resampler(self, features):
                return f"resampled:{features}"

        class _CheXagentModel:
            visual = _Visual()

        class _Outer:
            model = _CheXagentModel()

        model = _Outer()
        patch_chexagent_vision_forward(model)

        self.assertEqual(
            model.model.visual.forward("pixels"),
            "resampled:pre-layernorm-features",
        )
        self.assertEqual(model.model.visual.model.encoder.inputs_embeds, "embedded:pixels")


if __name__ == "__main__":
    unittest.main()
