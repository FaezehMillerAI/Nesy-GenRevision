import unittest

from nesy_gen.evaluation.graph_verification import (
    VerifiedCandidate,
    select_graph_verified_candidate,
)


class GraphVerificationTests(unittest.TestCase):
    def test_selects_highest_graph_score(self):
        weak = VerifiedCandidate(
            candidate_rank=0,
            prediction="weak",
            num_links=5,
            graph_score=0.4,
            bio_temporal=0.1,
            finding_to_diagnosis=1.0,
            located_in_type=1.0,
        )
        strong = VerifiedCandidate(
            candidate_rank=1,
            prediction="strong",
            num_links=2,
            graph_score=0.7,
            bio_temporal=0.2,
            finding_to_diagnosis=1.0,
            located_in_type=1.0,
        )

        selected = select_graph_verified_candidate([weak, strong])

        self.assertEqual(selected, strong)

    def test_falls_back_when_threshold_filters_all_candidates(self):
        candidate = VerifiedCandidate(
            candidate_rank=0,
            prediction="candidate",
            num_links=1,
            graph_score=0.2,
            bio_temporal=0.0,
            finding_to_diagnosis=0.6,
            located_in_type=1.0,
        )

        selected = select_graph_verified_candidate([candidate], min_graph_score=0.9)

        self.assertEqual(selected, candidate)


if __name__ == "__main__":
    unittest.main()
