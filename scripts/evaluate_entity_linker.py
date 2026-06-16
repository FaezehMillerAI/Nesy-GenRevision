from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.kg.entity_linking import LexicalEntityLinker, entity_linking_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate entity linking against JSONL gold nodes.")
    parser.add_argument("--vocab", required=True, help="CSV with node_id,node_name,node_type,alias columns.")
    parser.add_argument("--gold-jsonl", required=True, help="Rows with text and gold_node_ids.")
    args = parser.parse_args()

    linker = LexicalEntityLinker(pd.read_csv(args.vocab))
    scores = []
    with Path(args.gold_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            links = linker.link_text(row["text"])
            scores.append(entity_linking_scores(links, row["gold_node_ids"]))

    mean = {
        key: sum(score[key] for score in scores) / len(scores)
        for key in ["precision", "recall", "f1"]
    }
    print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()
