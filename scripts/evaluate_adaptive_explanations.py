from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.evaluation.explainability import (  # noqa: E402
    claim_trace_frame,
    explainability_summary,
    load_adaptive_traces,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate faithful adaptive NeSy claim traces.")
    parser.add_argument("--trace-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    traces = load_adaptive_traces(args.trace_jsonl)
    frame = claim_trace_frame(traces)
    summary = explainability_summary(frame)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    frame.to_csv(output_csv, index=False)
    print(json.dumps(summary, indent=2))
    print(f"Claim audit: {output_csv}")


if __name__ == "__main__":
    main()
