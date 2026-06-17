from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.schema import load_jsonl
from nesy_gen.evaluation.clinical_metrics import (
    chexpert_lite_frame,
    external_label_metrics,
    radgraph_lite_frame,
)
from nesy_gen.evaluation.factuality import evaluate_report_pairs, examples_to_pairs
from nesy_gen.evaluation.generation_metrics import corpus_generation_metrics
from nesy_gen.kg.entity_linking import LexicalEntityLinker


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated radiology reports.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-factuality-csv")
    parser.add_argument("--output-chexpert-csv")
    parser.add_argument("--output-radgraph-csv")
    parser.add_argument("--nodes-csv")
    parser.add_argument("--chexpert-pred-labels-csv")
    parser.add_argument("--chexpert-ref-labels-csv")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions_csv)
    if "reference" not in predictions.columns:
        references = {example.study_id: example.report for example in load_jsonl(args.manifest)}
        predictions["reference"] = predictions["study_id"].map(references)
    metrics = corpus_generation_metrics(predictions)
    result = {"num_predictions": len(predictions), "lexical_metrics": metrics}

    chexpert = chexpert_lite_frame(predictions)
    result["chexpert_lite_summary"] = chexpert.describe().fillna("").to_dict()
    if args.output_chexpert_csv:
        out_chexpert = Path(args.output_chexpert_csv)
        out_chexpert.parent.mkdir(parents=True, exist_ok=True)
        chexpert.to_csv(out_chexpert, index=False)
        result["chexpert_lite_csv"] = str(out_chexpert)

    if args.chexpert_pred_labels_csv and args.chexpert_ref_labels_csv:
        external = external_label_metrics(
            pd.read_csv(args.chexpert_pred_labels_csv),
            pd.read_csv(args.chexpert_ref_labels_csv),
        )
        result["external_chexpert_summary"] = external.describe().fillna("").to_dict()

    if args.nodes_csv:
        nodes = pd.read_csv(args.nodes_csv)
        vocab = nodes[["node_id", "node_name", "node_type"]].copy()
        vocab["alias"] = vocab["node_name"]
        linker = LexicalEntityLinker(vocab)
        pairs = examples_to_pairs(predictions, load_jsonl(args.manifest))
        factuality = evaluate_report_pairs(pairs, linker)
        factuality_summary = factuality.describe().fillna("").to_dict()
        result["entity_factuality_summary"] = factuality_summary
        if args.output_factuality_csv:
            out_fact = Path(args.output_factuality_csv)
            out_fact.parent.mkdir(parents=True, exist_ok=True)
            factuality.to_csv(out_fact, index=False)
            result["factuality_csv"] = str(out_fact)

        radgraph = radgraph_lite_frame(predictions, linker)
        result["radgraph_lite_summary"] = radgraph.describe().fillna("").to_dict()
        if args.output_radgraph_csv:
            out_radgraph = Path(args.output_radgraph_csv)
            out_radgraph.parent.mkdir(parents=True, exist_ok=True)
            radgraph.to_csv(out_radgraph, index=False)
            result["radgraph_lite_csv"] = str(out_radgraph)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
