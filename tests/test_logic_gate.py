import unittest

import pandas as pd

from nesy_gen.kg.primekg import PrimeKGGraph
from nesy_gen.logic.ltn import NeuroSymbolicAuditor
from nesy_gen.models.gate import CandidateToken, ConsistencyGate


class LogicGateTest(unittest.TestCase):
    def test_valid_located_in_token_is_accepted(self):
        kg = PrimeKGGraph.from_dataframe(
            pd.DataFrame(
                [
                    {
                        "x_id": "P:opacity",
                        "x_name": "opacity",
                        "x_type": "phenotype",
                        "y_id": "A:lung",
                        "y_name": "lung",
                        "y_type": "anatomy",
                        "display_relation": "located_in",
                        "confidence": 0.9,
                    }
                ]
            )
        )
        audit = NeuroSymbolicAuditor(beta_accept=0.6).audit(kg.graph)
        decision = ConsistencyGate().decide(
            CandidateToken("opacity", "P:opacity", evidence_score=0.8, hallucination_score=0.1),
            audit,
        )
        self.assertTrue(decision.accepted)

    def test_low_grounding_rejects(self):
        kg = PrimeKGGraph.from_dataframe(
            pd.DataFrame(
                [
                    {
                        "x_id": "P:opacity",
                        "x_name": "opacity",
                        "x_type": "phenotype",
                        "y_id": "A:lung",
                        "y_name": "lung",
                        "y_type": "anatomy",
                        "display_relation": "located_in",
                        "confidence": 0.9,
                    }
                ]
            )
        )
        audit = NeuroSymbolicAuditor(beta_accept=0.6).audit(kg.graph)
        decision = ConsistencyGate().decide(
            CandidateToken("opacity", "P:opacity", evidence_score=0.1, hallucination_score=0.1),
            audit,
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "low_visual_grounding")

    def test_assertion_polarity_is_validated(self):
        with self.assertRaises(ValueError):
            CandidateToken("opacity", "P:opacity", 0.8, 0.1, assertion_polarity="unknown")


if __name__ == "__main__":
    unittest.main()
