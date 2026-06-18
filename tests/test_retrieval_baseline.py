import unittest

from nesy_gen.baselines.retrieval import run_tfidf_retrieval, run_tfidf_retrieval_topk
from nesy_gen.data.schema import RadiologyExample


class RetrievalBaselineTest(unittest.TestCase):
    def test_retrieval_returns_train_report(self):
        train = [
            RadiologyExample("tr1", None, "cough", "No pneumonia.", "train"),
            RadiologyExample("tr2", None, "chest pain", "No pneumothorax.", "train"),
        ]
        test = [RadiologyExample("te1", None, "cough", "Reference.", "test")]
        preds = run_tfidf_retrieval(train, test)
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0].study_id, "te1")
        self.assertIn(preds[0].retrieved_study_id, {"tr1", "tr2"})

    def test_topk_retrieval_returns_ranked_predictions(self):
        train = [
            RadiologyExample("tr1", None, "cough", "No pneumonia.", "train"),
            RadiologyExample("tr2", None, "chest pain", "No pneumothorax.", "train"),
        ]
        test = [RadiologyExample("te1", None, "cough", "Reference.", "test")]
        preds = run_tfidf_retrieval_topk(train, test, top_k=2)
        self.assertEqual(len(preds), 1)
        self.assertEqual(len(preds[0]), 2)
        self.assertEqual(preds[0][0].rank, 1)
        self.assertEqual(preds[0][1].rank, 2)

    def test_retrieval_does_not_use_test_reference_as_query(self):
        train = [
            RadiologyExample("tr1", None, "", "Exact hidden reference.", "train"),
            RadiologyExample("tr2", None, "", "Different report.", "train"),
        ]
        test = [RadiologyExample("te1", None, "", "Exact hidden reference.", "test")]

        preds = run_tfidf_retrieval_topk(
            train,
            test,
            top_k=2,
        )

        self.assertEqual([prediction.similarity for prediction in preds[0]], [0.0, 0.0])

    def test_retrieval_never_reads_reference_to_filter_candidates(self):
        train = [
            RadiologyExample("tr1", None, "cough", "Exact hidden reference.", "train"),
            RadiologyExample("tr2", None, "cough", "Different report.", "train"),
        ]
        test = [RadiologyExample("te1", None, "cough", "Exact hidden reference.", "test")]

        preds = run_tfidf_retrieval_topk(train, test, top_k=2)

        self.assertEqual(
            {prediction.retrieved_study_id for prediction in preds[0]},
            {"tr1", "tr2"},
        )

    def test_retrieval_blocks_alternate_view_of_same_study(self):
        train = [
            RadiologyExample(
                "study_1_0", None, "cough", "Same study.", "train", {"r2gen_id": "study_1"}
            ),
            RadiologyExample(
                "study_2", None, "cough", "Other study.", "train", {"r2gen_id": "study_2"}
            ),
        ]
        query = [
            RadiologyExample(
                "study_1_1", None, "cough", "Hidden.", "test", {"r2gen_id": "study_1"}
            )
        ]

        preds = run_tfidf_retrieval_topk(train, query, top_k=2)

        self.assertEqual([row.retrieved_study_id for row in preds[0]], ["study_2"])

    def test_topk_contains_unique_underlying_studies(self):
        train = [
            RadiologyExample("s1_0", None, "cough", "A", "train", {"r2gen_id": "s1"}),
            RadiologyExample("s1_1", None, "cough", "A", "train", {"r2gen_id": "s1"}),
            RadiologyExample("s2_0", None, "cough", "B", "train", {"r2gen_id": "s2"}),
        ]
        query = [RadiologyExample("q", None, "cough", "Hidden", "test")]

        preds = run_tfidf_retrieval_topk(train, query, top_k=3)

        self.assertEqual(len(preds[0]), 2)
        self.assertEqual(
            {row.retrieved_study_id.rsplit("_", 1)[0] for row in preds[0]},
            {"s1", "s2"},
        )


if __name__ == "__main__":
    unittest.main()
