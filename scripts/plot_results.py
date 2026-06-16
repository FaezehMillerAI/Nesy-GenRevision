from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.evaluation.visualization import save_standard_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot standard IU/MIMIC result figures.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--graph-scores-csv")
    parser.add_argument("--factuality-csv")
    parser.add_argument("--sensitivity-csv")
    parser.add_argument("--entities-csv")
    args = parser.parse_args()

    paths = save_standard_plots(
        output_dir=args.output_dir,
        run_name=args.run_name,
        graph_scores=_read_optional(args.graph_scores_csv),
        factuality=_read_optional(args.factuality_csv),
        sensitivity=_read_optional(args.sensitivity_csv),
        entities=_read_optional(args.entities_csv),
    )
    for path in paths:
        print(f"Saved figure: {path}")


def _read_optional(path: str | None) -> pd.DataFrame | None:
    return pd.read_csv(path) if path else None


if __name__ == "__main__":
    main()

