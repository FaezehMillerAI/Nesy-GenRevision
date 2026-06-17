from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.schema import load_jsonl


TOKEN_RE = re.compile(r"[a-z0-9]+")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit generated predictions for suspicious reference overlap."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--high-overlap-threshold", type=float, default=0.95)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions_csv)
    if "reference" not in predictions.columns:
        references = {example.study_id: example.report for example in load_jsonl(args.manifest)}
        predictions["reference"] = predictions["study_id"].map(references)

    rows = []
    for row in predictions.itertuples(index=False):
        prediction = str(row.prediction)
        reference = str(row.reference)
        token_f1 = _token_f1(prediction, reference)
        rows.append(
            {
                "study_id": row.study_id,
                "exact_match": _normalize(prediction) == _normalize(reference),
                "token_f1": token_f1,
                "high_overlap": token_f1 >= args.high_overlap_threshold,
            }
        )

    audit = pd.DataFrame(rows)
    result = {
        "num_predictions": int(len(audit)),
        "exact_match_rate": float(audit["exact_match"].mean()) if len(audit) else 0.0,
        "high_overlap_threshold": args.high_overlap_threshold,
        "high_overlap_rate": float(audit["high_overlap"].mean()) if len(audit) else 0.0,
        "mean_token_f1": float(audit["token_f1"].mean()) if len(audit) else 0.0,
        "median_token_f1": float(audit["token_f1"].median()) if len(audit) else 0.0,
        "warning": (
            "High exact/high-overlap rates can indicate leakage, duplicated reports, or retrieval "
            "from reference-like text. Inspect candidate sources before using these scores."
        ),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def _normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(str(text).lower()))


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = TOKEN_RE.findall(str(prediction).lower())
    ref_tokens = TOKEN_RE.findall(str(reference).lower())
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = _counts(pred_tokens)
    ref_counts = _counts(ref_tokens)
    overlap = sum(min(pred_counts.get(token, 0), ref_counts.get(token, 0)) for token in pred_counts)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)


def _counts(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


if __name__ == "__main__":
    main()
