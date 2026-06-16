from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.kg.primekg import PrimeKGGraph, find_primekg_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local PrimeKG Dataverse files.")
    parser.add_argument("--dataverse-dir", default="dataverse_files")
    args = parser.parse_args()

    csv_path = find_primekg_csv(args.dataverse_dir)
    print("PrimeKG CSV:", csv_path if csv_path else "not found; will try edges.csv + nodes.csv")
    kg = PrimeKGGraph.from_dataverse_dir(args.dataverse_dir)
    print("Nodes:", len(list(kg.graph.nodes)))
    print("Edges:", len(kg.graph.edges()))


if __name__ == "__main__":
    main()
