import unittest

import pandas as pd

from nesy_gen.evaluation.generation_metrics import corpus_generation_metrics, meteor_lite, rouge_l, tokenize


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

    def test_empty_metrics(self):
        self.assertEqual(rouge_l([], ["a"]), 0.0)
        self.assertEqual(meteor_lite([], ["a"]), 0.0)


if __name__ == "__main__":
    unittest.main()

