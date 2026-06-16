from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.kaggle import print_dataset_paths, resolve_kaggle_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve IU X-ray or MIMIC-CXR KaggleHub paths.")
    parser.add_argument("--run-dataset", choices=["mimic_cxr", "iuxray"], required=True)
    args = parser.parse_args()
    print_dataset_paths(resolve_kaggle_dataset(args.run_dataset))


if __name__ == "__main__":
    main()
