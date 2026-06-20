import unittest

from nesy_gen.data.schema import RadiologyExample
from nesy_gen.training.medgemma_lora import build_sft_rows, normalize_findings_target
from scripts.generate_medgemma_adaptive_reports import _evaluation_label
from scripts.train_medgemma_lora import (
    _freeze_vision_encoder,
    _validate_split_contract,
    _vision_quantization_skip_modules,
)


class MedGemmaFinetuningTest(unittest.TestCase):
    def test_findings_section_is_used_without_impression(self):
        report = "FINDINGS: Mild bibasal atelectasis. IMPRESSION: Low lung volumes."

        self.assertEqual(normalize_findings_target(report), "Mild bibasal atelectasis.")

    def test_retrieval_conditioning_is_deterministic_and_training_only(self):
        example = RadiologyExample(
            study_id="query-1",
            image_path="/tmp/query.png",
            indication="cough",
            report="No focal opacity.",
            split="train",
        )
        evidence = {
            "query-1": [
                {"study_id": "train-2", "report": "No focal airspace opacity."},
                {"study_id": "train-3", "report": "The lungs are clear."},
            ]
        }

        first = build_sft_rows(
            [example],
            evidence_by_study_id=evidence,
            retrieval_probability=1.0,
            few_shot_k=1,
            seed=13,
        )
        second = build_sft_rows(
            [example],
            evidence_by_study_id=evidence,
            retrieval_probability=1.0,
            few_shot_k=1,
            seed=13,
        )

        self.assertEqual(first, second)
        self.assertTrue(first[0]["used_retrieval"])
        self.assertEqual(first[0]["evidence_study_ids"], ["train-2"])
        self.assertEqual(first[0]["target"], "No focal opacity.")
        user_text = first[0]["messages"][0]["content"][1]["text"]
        self.assertIn("No focal airspace opacity.", user_text)

    def test_zero_retrieval_probability_matches_image_only_training(self):
        example = RadiologyExample("s1", "/tmp/x.png", "", "Normal chest.", "train")
        rows = build_sft_rows(
            [example],
            evidence_by_study_id={"s1": [{"study_id": "s2", "report": "Normal."}]},
            retrieval_probability=0.0,
        )

        self.assertFalse(rows[0]["used_retrieval"])
        self.assertEqual(rows[0]["evidence_study_ids"], [])

    def test_finetuned_runs_have_an_explicit_evaluation_label(self):
        self.assertEqual(
            _evaluation_label("mimic_aug", "few_shot", finetuned=True),
            "task-specific-finetuned-few-shot",
        )

    def test_training_contract_rejects_test_or_shared_splits(self):
        with self.assertRaises(ValueError):
            _validate_split_contract("train", "test")
        with self.assertRaises(ValueError):
            _validate_split_contract("train", "train")

    def test_vision_and_optional_connector_freezing(self):
        class _Parameter:
            requires_grad = True

        class _Model:
            def __init__(self):
                self.parameters = {
                    "base.vision_tower.encoder.lora_A": _Parameter(),
                    "base.model.visual.encoder.lora_A": _Parameter(),
                    "base.multi_modal_projector.lora_A": _Parameter(),
                    "base.language_model.layers.0.q_proj.lora_A": _Parameter(),
                }

            def named_parameters(self):
                return self.parameters.items()

        frozen = _Model()
        _freeze_vision_encoder(frozen, train_connector=False)
        self.assertFalse(frozen.parameters["base.vision_tower.encoder.lora_A"].requires_grad)
        self.assertFalse(frozen.parameters["base.model.visual.encoder.lora_A"].requires_grad)
        self.assertFalse(frozen.parameters["base.multi_modal_projector.lora_A"].requires_grad)
        self.assertTrue(frozen.parameters["base.language_model.layers.0.q_proj.lora_A"].requires_grad)

        connector = _Model()
        _freeze_vision_encoder(connector, train_connector=True)
        self.assertFalse(connector.parameters["base.vision_tower.encoder.lora_A"].requires_grad)
        self.assertTrue(connector.parameters["base.multi_modal_projector.lora_A"].requires_grad)

    def test_chexagent_vision_tower_is_not_replaced_by_linear4bit(self):
        self.assertEqual(_vision_quantization_skip_modules("chexagent"), ["model.visual"])
        self.assertIsNone(_vision_quantization_skip_modules("medgemma"))


if __name__ == "__main__":
    unittest.main()
