from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_graph_verified_r2gen_t5_reports import main


if __name__ == "__main__":
    main()
