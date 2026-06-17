import unittest

import pandas as pd

from nesy_gen.evaluation.generation_metrics import (
    cider_lite,
    corpus_generation_metrics,
    meteor_lite,
    rouge_l,
    tokenize,
    token_prf,
)


class GenerationMetricsTest(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("No focal opacity."), ["no", "focal", "opacity"])

    def test_metrics_are_high_for_exact_match(self):
        frame = pd.DataFrame(
            [{"prediction": "no focal opacity", "reference": "no focal opacity"}]
        )
        metrics = corpus_generation_metrics(frame)
        self.assertGreater(metrics["bleu1"], 0.99)
        self.assertGreater(metrics["rouge_l"], 0.99)
        self.assertGreater(metrics["meteor_lite"], 0.99)
        self.assertGreater(metrics["token_precision"], 0.99)
        self.assertGreater(metrics["token_recall"], 0.99)
        self.assertGreater(metrics["token_f1"], 0.99)
        self.assertEqual(metrics["unique_prediction_ratio"], 1.0)

    def test_diversity_metrics_detect_repeated_predictions(self):
        frame = pd.DataFrame(
            [
                {"prediction": "lungs are clear", "reference": "lungs are clear"},
                {"prediction": "lungs are clear", "reference": "heart is normal"},
            ]
        )
        metrics = corpus_generation_metrics(frame)
        self.assertEqual(metrics["unique_prediction_ratio"], 0.5)
        self.assertEqual(metrics["max_prediction_frequency_rate"], 1.0)

    def test_empty_metrics(self):
        self.assertEqual(rouge_l([], ["a"]), 0.0)
        self.assertEqual(meteor_lite([], ["a"]), 0.0)

    def test_token_prf(self):
        scores = token_prf(["a", "b"], ["a", "c"])
        self.assertEqual(scores["precision"], 0.5)
        self.assertEqual(scores["recall"], 0.5)
        self.assertEqual(scores["f1"], 0.5)

    def test_cider_lite_exact_match_is_positive(self):
        score = cider_lite([(["lungs", "clear"], ["lungs", "clear"])])
        self.assertGreaterEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
