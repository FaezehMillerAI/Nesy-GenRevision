import unittest

from nesy_gen.evaluation.explainability import claim_trace_frame, explainability_summary


class ExplainabilityTest(unittest.TestCase):
    def test_claim_trace_summary(self):
        traces = [
            {
                "study_id": "s1",
                "claims": [
                    {
                        "decision": "accept_verified",
                        "linked_entities": [{"node_name": "opacity"}],
                        "primekg_path": [{"node_id": "1"}],
                        "visual_support": 0.4,
                        "retrieval_support": 0.7,
                        "verification_triggered": True,
                        "latency_ms": 10.0,
                    }
                ],
            }
        ]

        frame = claim_trace_frame(traces)
        summary = explainability_summary(frame)

        self.assertEqual(summary["num_claims"], 1)
        self.assertEqual(summary["linked_claim_rate"], 1.0)
        self.assertEqual(summary["graph_path_coverage_when_escalated"], 1.0)


if __name__ == "__main__":
    unittest.main()
