from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.manifests import (
    build_generic_csv_manifest,
    build_iuxray_manifest,
    build_mimic_aug_manifest,
    build_r2gen_iuxray_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset JSONL manifests.")
    parser.add_argument(
        "--dataset",
        choices=["iuxray", "iuxray_official", "r2gen_iuxray", "mimic_aug", "generic_csv"],
        required=True,
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--csv-path", help="Required for generic_csv.")
    parser.add_argument("--study-id-col", default="study_id")
    parser.add_argument("--image-path-col", default="image_path")
    parser.add_argument("--report-col", default="report")
    parser.add_argument("--indication-col", default="indication")
    parser.add_argument("--split-col")
    args = parser.parse_args()

    if args.dataset == "iuxray":
        examples = build_iuxray_manifest(args.data_root, args.output, seed=args.seed)
    elif args.dataset in {"iuxray_official", "r2gen_iuxray"}:
        examples = build_r2gen_iuxray_manifest(args.data_root, args.output)
    elif args.dataset == "mimic_aug":
        examples = build_mimic_aug_manifest(args.data_root, args.output, seed=args.seed)
    else:
        if not args.csv_path:
            raise ValueError("--csv-path is required for generic_csv")
        examples = build_generic_csv_manifest(
            args.csv_path,
            args.output,
            data_root=args.data_root,
            study_id_col=args.study_id_col,
            image_path_col=args.image_path_col,
            report_col=args.report_col,
            indication_col=args.indication_col,
            split_col=args.split_col,
            seed=args.seed,
        )
    print(f"Saved {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
