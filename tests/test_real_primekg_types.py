import unittest

import pandas as pd

from nesy_gen.kg.primekg import PrimeKGGraph
from nesy_gen.logic.constraints import compute_clause_scores, normalize_node_type


class RealPrimeKGTypesTest(unittest.TestCase):
    def test_effect_phenotype_normalizes_to_phenotype(self):
        self.assertEqual(normalize_node_type("effect/phenotype"), "phenotype")

    def test_effect_phenotype_to_anatomy_located_in_is_valid(self):
        kg = PrimeKGGraph.from_dataframe(
            pd.DataFrame(
                [
                    {
                        "x_id": "2202",
                        "x_name": "Pleural effusion",
                        "x_type": "effect/phenotype",
                        "y_id": "1443",
                        "y_name": "chest",
                        "y_type": "anatomy",
                        "display_relation": "located_in",
                    }
                ]
            )
        )
        scores = compute_clause_scores(kg.graph)
        self.assertEqual(scores.bio_temporal, 1.0)
        self.assertEqual(scores.located_in_type, 1.0)


if __name__ == "__main__":
    unittest.main()
