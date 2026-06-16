from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.schema import load_jsonl
from nesy_gen.evaluation.factuality import evaluate_report_pairs, examples_to_pairs
from nesy_gen.kg.entity_linking import LexicalEntityLinker


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate entity-level factuality/hallucination.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions-csv", required=True, help="CSV with study_id,prediction columns.")
    parser.add_argument("--nodes-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    references = load_jsonl(args.manifest)
    predictions = pd.read_csv(args.predictions_csv)
    required = {"study_id", "prediction"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing columns: {sorted(missing)}")

    nodes = pd.read_csv(args.nodes_csv)
    vocab = nodes[["node_id", "node_name", "node_type"]].copy()
    vocab["alias"] = vocab["node_name"]
    linker = LexicalEntityLinker(vocab)
    pairs = examples_to_pairs(predictions, references)
    frame = evaluate_report_pairs(pairs, linker)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"Saved {len(frame)} factuality rows to {out}")
    print(frame.describe().to_string())


if __name__ == "__main__":
    main()

