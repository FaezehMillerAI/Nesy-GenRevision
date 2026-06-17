import unittest

import pandas as pd

from nesy_gen.generation.constrained_decoding import PrimeKGDecodingConstraintBuilder


class _ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = None
    unk_token_id = None

    def __init__(self):
        self.vocab = {
            "pleural": 10,
            "effusion": 11,
            "cardiomegaly": 12,
            "no": 13,
            "without": 14,
            "absent": 15,
        }

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[token] for token in str(text).lower().split() if token in self.vocab]


class ConstrainedDecodingTest(unittest.TestCase):
    def test_supported_tokens_are_boosted_and_unsupported_tokens_penalized(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        nodes = pd.DataFrame(
            [
                {
                    "node_id": "1",
                    "node_name": "Pleural effusion",
                    "node_type": "effect/phenotype",
                },
                {
                    "node_id": "2",
                    "node_name": "Cardiomegaly",
                    "node_type": "disease",
                },
            ]
        )
        builder = PrimeKGDecodingConstraintBuilder(nodes, _ToyTokenizer())
        processor = builder.processor(
            ["no pleural effusion"],
            token_boost=2.0,
            unsupported_token_penalty=1.0,
        )

        scores = torch.zeros((1, 20))
        updated = processor(torch.zeros((1, 1), dtype=torch.long), scores)

        self.assertEqual(updated[0, 10].item(), 2.0)
        self.assertEqual(updated[0, 11].item(), 2.0)
        self.assertEqual(updated[0, 13].item(), 2.0)
        self.assertEqual(updated[0, 12].item(), -1.0)


if __name__ == "__main__":
    unittest.main()
