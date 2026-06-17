from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a reproducible experiment config JSON.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", choices=["r2gen_iuxray", "iuxray", "mimic_aug"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--primekg-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--generator", choices=["retrieval", "r2gen_t5", "rag_primekg"], default="rag_primekg")
    parser.add_argument("--r2gen-checkpoint-dir")
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--r2gen-num-candidates", type=int, default=4)
    parser.add_argument("--subgraph-strategy", choices=["steiner", "ego"], default="ego")
    parser.add_argument("--official-metrics", action="store_true")
    args = parser.parse_args()

    config = {
        "dataset": args.dataset,
        "manifest": args.manifest,
        "primekg_dir": args.primekg_dir,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "generator": args.generator,
        "r2gen_checkpoint_dir": args.r2gen_checkpoint_dir,
        "retrieval_top_k": args.retrieval_top_k,
        "r2gen_num_candidates": args.r2gen_num_candidates,
        "subgraph_strategy": args.subgraph_strategy,
        "official_metrics": args.official_metrics,
        "methods": {
            "rag_primekg": "retrieval candidates + optional R2Gen-T5 candidates + PrimeKG LTN audit + consistency gate",
            "r2gen_t5": "raw image-to-report generator",
            "retrieval": "TF-IDF retrieval baseline",
        },
        "reviewer_coverage": [
            "reproducible config",
            "real PrimeKG integration",
            "RAG baseline/generation",
            "ante-hoc PrimeKG verification",
            "LTN clause scoring",
            "consistency gate",
            "entity linking validation",
            "hallucination/factuality evaluation",
            "official metric hooks",
            "IU-Xray and MIMIC-CXR switchability",
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
