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
    radgraph_lite_frame,
)
from nesy_gen.evaluation.factuality import evaluate_report_pairs, examples_to_pairs
from nesy_gen.evaluation.generation_metrics import corpus_generation_metrics
from nesy_gen.evaluation.official_metrics import (
    chexbert_label_metrics,
    official_coco_metrics,
    official_radgraph_metrics,
    prepare_official_report_csv,
)
from nesy_gen.kg.entity_linking import LexicalEntityLinker


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated radiology reports.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-factuality-csv")
    parser.add_argument("--output-chexpert-csv")
    parser.add_argument("--output-radgraph-csv")
    parser.add_argument("--official-coco", action="store_true")
    parser.add_argument("--official-only", action="store_true")
    parser.add_argument("--prepare-official-inputs-dir")
    parser.add_argument("--nodes-csv")
    parser.add_argument("--chexbert-pred-labels-csv")
    parser.add_argument("--chexbert-ref-labels-csv")
    parser.add_argument("--chexpert-pred-labels-csv", help="Alias for --chexbert-pred-labels-csv.")
    parser.add_argument("--chexpert-ref-labels-csv", help="Alias for --chexbert-ref-labels-csv.")
    parser.add_argument("--output-official-chexbert-csv")
    parser.add_argument("--radgraph-pred-json")
    parser.add_argument("--radgraph-ref-json")
    parser.add_argument("--output-official-radgraph-csv")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions_csv)
    if "reference" not in predictions.columns:
        references = {example.study_id: example.report for example in load_jsonl(args.manifest)}
        predictions["reference"] = predictions["study_id"].map(references)
    if args.prepare_official_inputs_dir:
        prepare_official_inputs(predictions, args.prepare_official_inputs_dir)

    if args.official_coco or args.official_only:
        metrics = official_coco_metrics(predictions)
        lexical_key = "official_coco_metrics"
    else:
        metrics = corpus_generation_metrics(predictions)
        lexical_key = "lexical_metrics"
    result = {"num_predictions": len(predictions), lexical_key: metrics}

    chexbert_pred = args.chexbert_pred_labels_csv or args.chexpert_pred_labels_csv
    chexbert_ref = args.chexbert_ref_labels_csv or args.chexpert_ref_labels_csv

    if chexbert_pred and chexbert_ref:
        official_chexbert = chexbert_label_metrics(
            pd.read_csv(chexbert_pred),
            pd.read_csv(chexbert_ref),
        )
        result["official_chexbert_summary"] = official_chexbert.describe().fillna("").to_dict()
        if args.output_official_chexbert_csv:
            out_official_chexbert = Path(args.output_official_chexbert_csv)
            out_official_chexbert.parent.mkdir(parents=True, exist_ok=True)
            official_chexbert.to_csv(out_official_chexbert, index=False)
            result["official_chexbert_csv"] = str(out_official_chexbert)
    elif args.official_only:
        raise ValueError(
            "--official-only requires official CheXbert/CheXpert label CSVs via "
            "--chexbert-pred-labels-csv and --chexbert-ref-labels-csv."
        )
    else:
        chexpert = chexpert_lite_frame(predictions)
        result["chexpert_lite_summary"] = chexpert.describe().fillna("").to_dict()
        if args.output_chexpert_csv:
            out_chexpert = Path(args.output_chexpert_csv)
            out_chexpert.parent.mkdir(parents=True, exist_ok=True)
            chexpert.to_csv(out_chexpert, index=False)
            result["chexpert_lite_csv"] = str(out_chexpert)

    if args.radgraph_pred_json and args.radgraph_ref_json:
        official_radgraph = official_radgraph_metrics(args.radgraph_pred_json, args.radgraph_ref_json)
        result["official_radgraph_summary"] = official_radgraph.describe().fillna("").to_dict()
        if args.output_official_radgraph_csv:
            out_official_radgraph = Path(args.output_official_radgraph_csv)
            out_official_radgraph.parent.mkdir(parents=True, exist_ok=True)
            official_radgraph.to_csv(out_official_radgraph, index=False)
            result["official_radgraph_csv"] = str(out_official_radgraph)
    elif args.official_only:
        raise ValueError(
            "--official-only requires official RadGraph JSON outputs via "
            "--radgraph-pred-json and --radgraph-ref-json."
        )

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

        if not args.official_only and not (args.radgraph_pred_json and args.radgraph_ref_json):
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


def prepare_official_inputs(predictions: pd.DataFrame, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prepare_official_report_csv(
        predictions,
        out / "pred_reports_for_chexbert.csv",
        text_column="prediction",
    )
    prepare_official_report_csv(
        predictions,
        out / "ref_reports_for_chexbert.csv",
        text_column="reference",
    )
    (out / "reports_for_radgraph.json").write_text(
        json.dumps(
            {
                str(row.study_id): {
                    "prediction": str(row.prediction),
                    "reference": str(row.reference),
                }
                for row in predictions.itertuples(index=False)
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
