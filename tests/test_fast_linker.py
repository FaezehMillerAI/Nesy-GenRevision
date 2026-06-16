import unittest

import pandas as pd

from nesy_gen.kg.entity_linking import LexicalEntityLinker


class FastLinkerTest(unittest.TestCase):
    def test_longest_match_uses_token_index(self):
        vocab = pd.DataFrame(
            [
                {"node_id": "1", "node_name": "lower", "node_type": "anatomy", "alias": "lower"},
                {
                    "node_id": "2",
                    "node_name": "right lower lobe",
                    "node_type": "anatomy",
                    "alias": "right lower lobe",
                },
            ]
        )
        linker = LexicalEntityLinker(vocab)
        links = linker.link_text("The right lower lobe is clear.")
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].node_id, "2")


if __name__ == "__main__":
    unittest.main()

