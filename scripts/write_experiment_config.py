from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a reproducible experiment config JSON.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dataset",
        choices=["iuxray_official", "r2gen_iuxray", "iuxray", "mimic_aug"],
        required=True,
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--primekg-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--generator",
        choices=["retrieval", "vision_t5", "r2gen_t5", "rag_primekg", "rag_primekg_constrained"],
        default="rag_primekg",
    )
    parser.add_argument("--r2gen-checkpoint-dir", dest="generator_checkpoint_dir")
    parser.add_argument("--generator-checkpoint-dir", dest="generator_checkpoint_dir")
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--retrieval-mode", choices=["metadata", "visual"], default="metadata")
    parser.add_argument("--retrieval-conditioning", action="store_true")
    parser.add_argument("--r2gen-num-candidates", "--generator-num-candidates", dest="generator_num_candidates", type=int, default=4)
    parser.add_argument("--vision-backbone", default="")
    parser.add_argument("--text-model-name", default="")
    parser.add_argument("--freeze-visual-encoder", action="store_true")
    parser.add_argument(
        "--decoding-mode",
        choices=["standard", "graph_constrained"],
        default="standard",
    )
    parser.add_argument("--graph-token-boost", type=float, default=2.0)
    parser.add_argument("--unsupported-token-penalty", type=float, default=0.0)
    parser.add_argument("--generator-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--generator-no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--generator-length-penalty", type=float, default=1.0)
    parser.add_argument("--generator-num-beam-groups", type=int, default=1)
    parser.add_argument("--generator-diversity-penalty", type=float, default=0.0)
    parser.add_argument(
        "--selection-objective",
        choices=["graph", "evidence", "hybrid"],
        default="graph",
    )
    parser.add_argument("--graph-score-weight", type=float, default=0.55)
    parser.add_argument("--evidence-weight", type=float, default=0.35)
    parser.add_argument("--gate-weight", type=float, default=0.10)
    parser.add_argument(
        "--graph-training-mode",
        choices=["none", "primekg_token"],
        default="none",
    )
    parser.add_argument("--graph-token-loss-weight", type=float, default=0.0)
    parser.add_argument("--unsupported-token-loss-weight", type=float, default=0.0)
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
        "generator_checkpoint_dir": args.generator_checkpoint_dir,
        "vision_backbone": args.vision_backbone,
        "text_model_name": args.text_model_name,
        "freeze_visual_encoder": args.freeze_visual_encoder,
        "retrieval_top_k": args.retrieval_top_k,
        "retrieval_mode": args.retrieval_mode,
        "retrieval_conditioning": args.retrieval_conditioning,
        "generator_num_candidates": args.generator_num_candidates,
        "decoding_mode": args.decoding_mode,
        "graph_token_boost": args.graph_token_boost,
        "unsupported_token_penalty": args.unsupported_token_penalty,
        "generator_repetition_penalty": args.generator_repetition_penalty,
        "generator_no_repeat_ngram_size": args.generator_no_repeat_ngram_size,
        "generator_length_penalty": args.generator_length_penalty,
        "generator_num_beam_groups": args.generator_num_beam_groups,
        "generator_diversity_penalty": args.generator_diversity_penalty,
        "selection_objective": args.selection_objective,
        "graph_score_weight": args.graph_score_weight,
        "evidence_weight": args.evidence_weight,
        "gate_weight": args.gate_weight,
        "graph_training_mode": args.graph_training_mode,
        "graph_token_loss_weight": args.graph_token_loss_weight,
        "unsupported_token_loss_weight": args.unsupported_token_loss_weight,
        "subgraph_strategy": args.subgraph_strategy,
        "official_metrics": args.official_metrics,
        "methods": {
            "rag_primekg": "retrieval candidates + optional Vision-T5 candidates + PrimeKG LTN audit + consistency gate",
            "rag_primekg_constrained": "RAG evidence + soft PrimeKG-constrained Vision-T5 decoding + LTN audit + consistency gate",
            "primekg_token_training": "optional Vision-T5 training regularizer that upweights PrimeKG-linked entity tokens in references",
            "vision_t5": "raw image-to-report generator",
            "r2gen_t5": "deprecated alias for the raw image-to-report generator",
            "retrieval": "reference-blind metadata or frozen-image retrieval baseline",
            "visual_rag_conditioning": "leave-one-study-out frozen-image retrieval evidence encoded jointly with visual patches",
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
