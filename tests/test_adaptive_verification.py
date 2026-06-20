import unittest

import pandas as pd

from nesy_gen.agents.adaptive_verification import AdaptiveClaimVerifier, split_clinical_claims
from nesy_gen.generation.rag import RagCandidate
from nesy_gen.kg.entity_linking import LexicalEntityLinker
from nesy_gen.kg.simple_graph import SimpleDiGraph
from nesy_gen.logic.ltn import AuditReport
from nesy_gen.models.gate import ConsistencyGate


class _Scores:
    def __init__(self, mean):
        self.mean = mean

    def as_dict(self):
        return {
            "bio_temporal": self.mean,
            "finding_to_diagnosis": self.mean,
            "located_in_type": self.mean,
            "mean": self.mean,
        }


class _Builder:
    def build(self, links):
        graph = SimpleDiGraph()
        for link in links:
            graph.add_node(link.node_id, name=link.node_name)
        node_ids = list(graph.nodes)
        for left, right in zip(node_ids, node_ids[1:]):
            graph.add_edge(left, right, confidence=0.9)
        return graph


class _Auditor:
    def __init__(self, mean, accepted):
        self.mean = mean
        self.accepted = accepted

    def audit(self, graph):
        nodes = set(graph.nodes)
        return AuditReport(
            scores=_Scores(self.mean),
            valid_nodes=nodes if self.accepted else set(),
            flagged_nodes=set(),
            rejected_nodes=set() if self.accepted else nodes,
        )


class _Pipeline:
    def __init__(self, *, mean=0.9, accepted=True):
        vocab = pd.DataFrame(
            [
                {"node_id": "1", "node_name": "opacity", "node_type": "effect/phenotype"},
                {"node_id": "2", "node_name": "cardiomegaly", "node_type": "disease"},
            ]
        )
        self.linker = LexicalEntityLinker(vocab)
        self.subgraph_builder = _Builder()
        self.auditor = _Auditor(mean, accepted)
        self.gate = ConsistencyGate()


class AdaptiveVerificationTest(unittest.TestCase):
    def test_claim_splitter_preserves_sentences(self):
        self.assertEqual(
            split_clinical_claims("No opacity. Heart size is normal."),
            ["No opacity.", "Heart size is normal."],
        )

    def test_high_consensus_claim_uses_fast_path_without_graph(self):
        verifier = AdaptiveClaimVerifier(_Pipeline(), fast_accept_threshold=0.8)
        evidence = [
            RagCandidate("visual_retrieval", 1, "Opacity is present.", 0.9, "a"),
            RagCandidate("visual_retrieval", 2, "Persistent opacity.", 0.85, "b"),
        ]

        result = verifier.verify("Opacity is present.", evidence_candidates=evidence)

        self.assertEqual(result.graph_calls, 0)
        self.assertEqual(result.claims[0].decision, "accept_fast_path")
        self.assertFalse(result.claims[0].verification_triggered)
        self.assertEqual(result.claims[0].retrieval_support_study_ids, ["a", "b"])
        self.assertEqual(result.linked_claims, 1)
        self.assertEqual(result.escalation_rate_linked, 0.0)

    def test_medsiglip_evidence_is_used_by_adaptive_router(self):
        verifier = AdaptiveClaimVerifier(
            _Pipeline(), fast_accept_threshold=0.8, min_supporting_reports=1
        )
        evidence = [RagCandidate("medsiglip_retrieval", 1, "Opacity.", 0.9, "a")]

        result = verifier.verify("Opacity.", evidence_candidates=evidence)

        self.assertEqual(result.claims[0].decision, "accept_fast_path")

    def test_disputed_claim_can_use_evidence_bound_replacement(self):
        verifier = AdaptiveClaimVerifier(
            _Pipeline(mean=0.2, accepted=False),
            fast_accept_threshold=0.99,
            revise_threshold=0.5,
        )
        evidence = [
            RagCandidate("visual_retrieval", 1, "Persistent opacity.", 0.8, "train-1")
        ]

        result = verifier.verify("Opacity is present.", evidence_candidates=evidence)

        self.assertEqual(result.graph_calls, 1)
        self.assertEqual(result.claims[0].decision, "revise")
        self.assertEqual(result.claims[0].replacement_source_study_id, "train-1")
        self.assertEqual(result.claims[0].replacement_source_rank, 1)
        self.assertEqual(result.claims[0].replacement_evidence_score, 0.8)
        self.assertEqual(result.final_report, "Persistent opacity.")

    def test_linked_escalation_rate_excludes_abstentions(self):
        verifier = AdaptiveClaimVerifier(_Pipeline(mean=0.2, accepted=False))

        result = verifier.verify("Opacity. Unremarkable examination.")

        self.assertEqual(result.total_claims, 2)
        self.assertEqual(result.linked_claims, 1)
        self.assertEqual(result.escalation_rate, 0.5)
        self.assertEqual(result.escalation_rate_linked, 1.0)

    def test_negated_claim_preserves_polarity_in_retrieval_contract(self):
        verifier = AdaptiveClaimVerifier(
            _Pipeline(), fast_accept_threshold=0.8, min_supporting_reports=2
        )
        evidence = [
            RagCandidate("visual_retrieval", 1, "No opacity.", 0.9, "a"),
            RagCandidate("visual_retrieval", 2, "No opacity.", 0.85, "b"),
            RagCandidate("visual_retrieval", 3, "Opacity.", 0.99, "c"),
        ]

        result = verifier.verify("No opacity.", evidence_candidates=evidence)

        self.assertEqual(result.claims[0].decision, "accept_fast_path")
        self.assertEqual(result.claims[0].retrieval_support_study_ids, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
