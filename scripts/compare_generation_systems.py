from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.schema import load_jsonl
from nesy_gen.evaluation.factuality import evaluate_report_pairs, examples_to_pairs
from nesy_gen.evaluation.generation_metrics import corpus_generation_metrics
from nesy_gen.kg.entity_linking import LexicalEntityLinker


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple report-generation systems.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--nodes-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--system",
        action="append",
        nargs=2,
        metavar=("NAME", "PREDICTIONS_CSV"),
        required=True,
        help="System name and CSV path with study_id,prediction columns.",
    )
    args = parser.parse_args()

    references = load_jsonl(args.manifest)
    nodes = pd.read_csv(args.nodes_csv)
    vocab = nodes[["node_id", "node_name", "node_type"]].copy()
    vocab["alias"] = vocab["node_name"]
    linker = LexicalEntityLinker(vocab)

    results = {}
    for name, csv_path in args.system:
        predictions = pd.read_csv(csv_path)
        if "reference" not in predictions.columns:
            ref_map = {example.study_id: example.report for example in references}
            predictions["reference"] = predictions["study_id"].map(ref_map)
        lexical = corpus_generation_metrics(predictions)
        pairs = examples_to_pairs(predictions, references)
        factuality = evaluate_report_pairs(pairs, linker)
        results[name] = {
            "csv": csv_path,
            "num_predictions": len(predictions),
            "lexical_metrics": lexical,
            "entity_factuality": factuality.describe().fillna("").to_dict(),
        }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2)[:6000])


if __name__ == "__main__":
    main()
