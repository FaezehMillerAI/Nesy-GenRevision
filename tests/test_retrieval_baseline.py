import unittest

from nesy_gen.baselines.retrieval import run_tfidf_retrieval
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


if __name__ == "__main__":
    unittest.main()

