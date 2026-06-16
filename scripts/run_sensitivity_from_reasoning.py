from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.evaluation.sensitivity import run_linking_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description="Run entity-link perturbation sensitivity.")
    parser.add_argument("--reasoning-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = json.loads(Path(args.reasoning_json).read_text(encoding="utf-8"))
    all_nodes = sorted(
        {
            entity["node_id"]
            for row in rows
            for entity in row.get("linked_entities", [])
            if not entity.get("negated", False)
        }
    )

    output_rows = []
    for row in rows:
        reference = [
            entity["node_id"]
            for entity in row.get("linked_entities", [])
            if not entity.get("negated", False)
        ]
        if not reference:
            continue
        sensitivity = run_linking_sensitivity(
            reference,
            candidate_node_ids=all_nodes,
            drop_rates=[0.0, 0.1, 0.2, 0.3, 0.5],
            swap_rates=[0.0, 0.1, 0.2],
            trials=args.trials,
            seed=args.seed,
        )
        for item in sensitivity:
            output_rows.append({"study_id": row["study_id"], **item})

    frame = pd.DataFrame(output_rows)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"Saved {len(frame)} rows to {out}")


if __name__ == "__main__":
    main()

