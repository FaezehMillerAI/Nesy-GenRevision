import tempfile
from pathlib import Path
import unittest

import pandas as pd

from nesy_gen.kg.primekg import PrimeKGGraph, find_primekg_csv


class PrimeKGDataverseTest(unittest.TestCase):
    def test_finds_kg_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kg.csv").write_text("x_id,x_name,x_type,y_id,y_name,y_type,relation\n", encoding="utf-8")
            self.assertEqual(find_primekg_csv(root), root / "kg.csv")

    def test_reconstructs_from_edges_and_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "node_index": 1,
                        "node_id": "D:pneumonia",
                        "node_type": "disease",
                        "node_name": "pneumonia",
                        "node_source": "toy",
                    },
                    {
                        "node_index": 2,
                        "node_id": "P:opacity",
                        "node_type": "phenotype",
                        "node_name": "opacity",
                        "node_source": "toy",
                    },
                ]
            ).to_csv(root / "nodes.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "x_index": 1,
                        "y_index": 2,
                        "relation": "causes",
                        "display_relation": "causes",
                    }
                ]
            ).to_csv(root / "edges.csv", index=False)

            kg = PrimeKGGraph.from_dataverse_dir(root)
            self.assertIn("D:pneumonia", kg.graph)
            self.assertEqual(len(kg.graph.edges()), 1)


if __name__ == "__main__":
    unittest.main()
