import unittest

import pandas as pd

from nesy_gen.kg.entity_linking import EntityMention, LinkedEntity
from nesy_gen.kg.primekg import PrimeKGGraph
from nesy_gen.kg.temporal import TemporalSubgraphBuilder


class TemporalStrategiesTest(unittest.TestCase):
    def test_ego_strategy_keeps_incident_neighbors(self):
        kg = PrimeKGGraph.from_dataframe(
            pd.DataFrame(
                [
                    {
                        "x_id": "D:pneumonia",
                        "x_name": "pneumonia",
                        "x_type": "disease",
                        "y_id": "P:opacity",
                        "y_name": "opacity",
                        "y_type": "effect/phenotype",
                        "display_relation": "causes",
                    },
                    {
                        "x_id": "P:opacity",
                        "x_name": "opacity",
                        "x_type": "effect/phenotype",
                        "y_id": "A:lung",
                        "y_name": "lung",
                        "y_type": "anatomy",
                        "display_relation": "located_in",
                    },
                ]
            )
        )
        links = [
            LinkedEntity(EntityMention("opacity", 0, 7), "P:opacity", "opacity", "effect/phenotype", 1.0),
        ]
        graph = TemporalSubgraphBuilder(kg, strategy="ego").build(links)
        self.assertIn("D:pneumonia", graph)
        self.assertIn("A:lung", graph)

    def test_hybrid_strategy_audits_negated_entities_and_paths(self):
        kg = PrimeKGGraph.from_dataframe(
            pd.DataFrame(
                [
                    {
                        "x_id": "D:pneumonia",
                        "x_name": "pneumonia",
                        "x_type": "disease",
                        "y_id": "P:opacity",
                        "y_name": "opacity",
                        "y_type": "effect/phenotype",
                        "display_relation": "causes",
                    },
                    {
                        "x_id": "P:opacity",
                        "x_name": "opacity",
                        "x_type": "effect/phenotype",
                        "y_id": "A:lung",
                        "y_name": "lung",
                        "y_type": "anatomy",
                        "display_relation": "located_in",
                    },
                ]
            )
        )
        links = [
            LinkedEntity(
                EntityMention("opacity", 3, 10, negated=True),
                "P:opacity",
                "opacity",
                "effect/phenotype",
                1.0,
            ),
            LinkedEntity(
                EntityMention("pneumonia", 11, 20),
                "D:pneumonia",
                "pneumonia",
                "disease",
                1.0,
            ),
        ]

        graph = TemporalSubgraphBuilder(kg, strategy="hybrid").build(links)

        self.assertIn("P:opacity", graph)
        self.assertIn("D:pneumonia", graph)
        self.assertIn("A:lung", graph)


if __name__ == "__main__":
    unittest.main()
