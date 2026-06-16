import unittest

import pandas as pd

from nesy_gen.kg.entity_linking import LexicalEntityLinker, entity_linking_scores


class EntityLinkerTest(unittest.TestCase):
    def test_links_longest_alias_and_negation(self):
        vocab = pd.DataFrame(
            [
                {"node_id": "1", "node_name": "effusion", "node_type": "phenotype", "alias": "effusion"},
                {
                    "node_id": "2",
                    "node_name": "pleural effusion",
                    "node_type": "phenotype",
                    "alias": "pleural effusion",
                },
            ]
        )
        linker = LexicalEntityLinker(vocab)
        links = linker.link_text("No pleural effusion is seen.")
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].node_id, "2")
        self.assertTrue(links[0].mention.negated)

    def test_scores(self):
        vocab = pd.DataFrame(
            [{"node_id": "1", "node_name": "pneumonia", "node_type": "disease", "alias": "pneumonia"}]
        )
        linker = LexicalEntityLinker(vocab)
        scores = entity_linking_scores(linker.link_text("pneumonia"), ["1", "2"])
        self.assertAlmostEqual(scores["precision"], 1.0)
        self.assertAlmostEqual(scores["recall"], 0.5)

    def test_repeated_mentions_are_linked(self):
        vocab = pd.DataFrame(
            [{"node_id": "1", "node_name": "atelectasis", "node_type": "phenotype", "alias": "atelectasis"}]
        )
        linker = LexicalEntityLinker(vocab)
        links = linker.link_text("Atelectasis and mild atelectasis are present.")
        self.assertEqual(len(links), 2)


if __name__ == "__main__":
    unittest.main()
