import unittest

from nesy_gen.data.schema import RadiologyExample
from nesy_gen.generation.rag import (
    RagCandidate,
    select_agentic_draft,
    select_primekg_verified_report,
)


class _FakeScores:
    def __init__(self, mean):
        self.mean = mean

    def as_dict(self):
        return {
            "bio_temporal": self.mean,
            "finding_to_diagnosis": self.mean,
            "located_in_type": self.mean,
            "mean": self.mean,
        }


class _FakeAudit:
    def __init__(self, mean):
        self.scores = _FakeScores(mean)
        self.valid_nodes = {"1"}
        self.flagged_nodes = set()
        self.rejected_nodes = set()
        self.mean_satisfaction = mean


class _FakeMention:
    text = "opacity"
    negated = False


class _FakeLink:
    mention = _FakeMention()
    node_id = "1"


class _FakeGate:
    def decide(self, token, audit):
        class Decision:
            accepted = True

        return Decision()


class _FakePipeline:
    gate = _FakeGate()

    def reason(self, indication, draft_report):
        mean = 0.9 if "good" in draft_report else 0.1
        return [_FakeLink()], _FakeAudit(mean)


class RagGenerationTest(unittest.TestCase):
    def test_select_primekg_verified_report(self):
        example = RadiologyExample("s1", None, "", "reference", "test")
        selected, candidates = select_primekg_verified_report(
            _FakePipeline(),
            example,
            [
                RagCandidate("retrieval", 1, "bad report", 0.5, "tr1"),
                RagCandidate("retrieval", 2, "good report", 0.4, "tr2"),
            ],
        )

        self.assertEqual(selected["prediction"], "good report")
        self.assertEqual(selected["selection_status"], "primekg_ltn_gate_selected")
        self.assertEqual(len(candidates), 2)

    def test_hybrid_selection_can_prefer_retrieval_evidence(self):
        example = RadiologyExample("s1", None, "", "reference", "test")
        selected, _ = select_primekg_verified_report(
            _FakePipeline(),
            example,
            [
                RagCandidate("retrieval", 1, "bad but retrieved report", 0.95, "tr1"),
                RagCandidate("vision_t5", 1, "good generated report", 0.30, ""),
            ],
            selection_objective="hybrid",
            graph_score_weight=0.10,
            evidence_weight=0.85,
            gate_weight=0.05,
        )

        self.assertEqual(selected["prediction"], "bad but retrieved report")

    def test_agentic_draft_prefers_image_conditioned_generator(self):
        example = RadiologyExample("s1", None, "", "reference", "test")
        selected, audit = select_agentic_draft(
            example,
            [
                RagCandidate("visual_retrieval", 1, "retrieved", 0.95, "tr1"),
                RagCandidate("vision_t5", 1, "generated", 0.55),
            ],
        )

        self.assertEqual(selected["prediction"], "generated")
        self.assertEqual(selected["selection_status"], "agentic_draft_unverified")
        self.assertEqual(sum(row["selected_as_draft"] for row in audit), 1)


if __name__ == "__main__":
    unittest.main()
