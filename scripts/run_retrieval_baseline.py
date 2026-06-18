from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.baselines.retrieval import run_tfidf_retrieval
from nesy_gen.data.schema import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TF-IDF retrieval baseline.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    examples = load_jsonl(args.manifest)
    train = [example for example in examples if example.split == "train"]
    queries = [example for example in examples if example.split == args.split]
    if args.limit is not None:
        queries = queries[: args.limit]
    predictions = run_tfidf_retrieval(train, queries)
    frame = pd.DataFrame([asdict(prediction) for prediction in predictions])
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"Saved {len(frame)} predictions to {out}")


if __name__ == "__main__":
    main()
