from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.schema import load_jsonl
from nesy_gen.evaluation.visualization import build_qualitative_html, save_qualitative_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Build qualitative HTML report.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--graph-scores-csv")
    parser.add_argument("--retrieval-csv")
    parser.add_argument("--factuality-csv")
    parser.add_argument("--max-examples", type=int, default=25)
    args = parser.parse_args()

    html = build_qualitative_html(
        examples=load_jsonl(args.manifest),
        predictions=pd.read_csv(args.predictions_csv),
        graph_scores=_read_optional(args.graph_scores_csv),
        retrieval=_read_optional(args.retrieval_csv),
        factuality=_read_optional(args.factuality_csv),
        run_name=args.run_name,
        max_examples=args.max_examples,
    )
    out = save_qualitative_html(html, args.output_html)
    print(f"Saved qualitative report: {out}")


def _read_optional(path: str | None) -> pd.DataFrame | None:
    return pd.read_csv(path) if path else None


if __name__ == "__main__":
    main()

