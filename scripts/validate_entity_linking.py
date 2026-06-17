from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.schema import load_jsonl
from nesy_gen.evaluation.entity_validation import write_entity_validation_bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create reviewer-facing entity extraction/linking validation tables."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--nodes-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--audit-sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    examples = [example for example in load_jsonl(args.manifest) if example.split == args.split]
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError(f"No examples found for split={args.split}")

    nodes = pd.read_csv(args.nodes_csv)
    vocab = nodes[["node_id", "node_name", "node_type"]].copy()
    vocab["alias"] = vocab["node_name"]
    paths = write_entity_validation_bundle(
        examples,
        vocab,
        args.output_dir,
        prefix=args.prefix,
        audit_sample_size=args.audit_sample_size,
        seed=args.seed,
    )
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
