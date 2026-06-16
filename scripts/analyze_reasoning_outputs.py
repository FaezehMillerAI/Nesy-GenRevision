from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.evaluation.analysis import write_analysis_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reviewer-facing reasoning analysis tables.")
    parser.add_argument("--reasoning-json", required=True)
    parser.add_argument("--scores-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    paths = write_analysis_bundle(
        args.reasoning_json,
        args.scores_csv,
        args.output_dir,
        prefix=args.prefix,
    )
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

