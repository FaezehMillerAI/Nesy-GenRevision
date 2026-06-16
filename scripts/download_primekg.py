from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request


PRIMEKG_CSV_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the ready-to-use PrimeKG CSV.")
    parser.add_argument("--out", default="data/primekg/kg.csv", help="Output CSV path.")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading PrimeKG to {out}")
    urllib.request.urlretrieve(PRIMEKG_CSV_URL, out)
    print("Done")


if __name__ == "__main__":
    main()

