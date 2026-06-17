from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper ablation suite for Nesy-Gen: retrieval, raw Vision-T5, "
            "RAG/PrimeKG gate, graph-constrained decoding, and a BLEU-guarded graph profile."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--primekg-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--generator-checkpoint-dir")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=140)
    parser.add_argument("--dry-run", action="store_true", help="Write commands without executing them.")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    commands = build_commands(args)
    manifest_path = out / f"{args.dataset_name}_ablation_commands.json"
    manifest_path.write_text(json.dumps(commands, indent=2), encoding="utf-8")
    print(f"Ablation command manifest: {manifest_path}")

    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return

    for item in commands:
        print("\n" + "=" * 100, flush=True)
        print(f"Running ablation: {item['name']}", flush=True)
        print("$ " + " ".join(item["cmd"]), flush=True)
        subprocess.run(item["cmd"], check=True)

    compare_cmd = compare_command(args, commands)
    print("\n" + "=" * 100, flush=True)
    print("Running final system comparison", flush=True)
    print("$ " + " ".join(compare_cmd), flush=True)
    subprocess.run(compare_cmd, check=True)


def build_commands(args) -> list[dict[str, object]]:
    commands: list[dict[str, object]] = []
    out = Path(args.output_dir)
    prefix = args.dataset_name

    retrieval_csv = out / f"{prefix}_ablation_retrieval.csv"
    commands.append(
        {
            "name": "retrieval",
            "purpose": "Leakage-safe TF-IDF retrieval baseline.",
            "prediction_csv": str(retrieval_csv),
            "cmd": _with_limit(
                [
                    sys.executable,
                    "scripts/run_retrieval_baseline.py",
                    "--manifest",
                    args.manifest,
                    "--output-csv",
                    str(retrieval_csv),
                    "--split",
                    args.split,
                ],
                args.limit,
            ),
        }
    )
    commands.append(_eval_command(args, "retrieval", retrieval_csv))

    if args.generator_checkpoint_dir:
        raw_csv = out / f"{prefix}_ablation_vision_t5_standard.csv"
        commands.append(
            {
                "name": "vision_t5_standard",
                "purpose": "Raw image-to-report generator without retrieval or graph selection.",
                "prediction_csv": str(raw_csv),
                "cmd": _with_limit(
                    [
                        sys.executable,
                        "scripts/generate_vision_t5_reports.py",
                        "--manifest",
                        args.manifest,
                        "--checkpoint-dir",
                        args.generator_checkpoint_dir,
                        "--output-csv",
                        str(raw_csv),
                        "--split",
                        args.split,
                        "--batch-size",
                        str(args.batch_size),
                        "--num-beams",
                        "8",
                        "--max-new-tokens",
                        str(args.max_new_tokens),
                        "--num-return-sequences",
                        "1",
                        "--repetition-penalty",
                        "1.12",
                        "--no-repeat-ngram-size",
                        "3",
                        "--length-penalty",
                        "0.90",
                        "--num-beam-groups",
                        "4",
                        "--diversity-penalty",
                        "0.35",
                    ],
                    args.limit,
                ),
            }
        )
        commands.append(_eval_command(args, "vision_t5_standard", raw_csv))

    rag_variants = [
        {
            "name": "rag_primekg_gate",
            "purpose": "RAG candidates audited by PrimeKG/LTN and selected by balanced graph/evidence/gate score.",
            "decoding_mode": "standard",
            "retrieval_top_k": 10,
            "generator_num_candidates": 8 if args.generator_checkpoint_dir else 0,
            "graph_weight": 0.55,
            "evidence_weight": 0.35,
            "gate_weight": 0.10,
            "generated_evidence_score": 0.55,
        },
        {
            "name": "graph_constrained_balanced",
            "purpose": "Soft PrimeKG-constrained decoding plus balanced PrimeKG/LTN selection.",
            "decoding_mode": "graph_constrained",
            "retrieval_top_k": 10,
            "generator_num_candidates": 8 if args.generator_checkpoint_dir else 0,
            "graph_weight": 0.55,
            "evidence_weight": 0.35,
            "gate_weight": 0.10,
            "generated_evidence_score": 0.55,
        },
        {
            "name": "graph_constrained_bleu_guarded",
            "purpose": (
                "BLEU-oriented but defensible graph-constrained profile: stronger retrieval evidence "
                "weight, PrimeKG/LTN audit retained, and no reference text used at inference."
            ),
            "decoding_mode": "graph_constrained",
            "retrieval_top_k": 20,
            "generator_num_candidates": 8 if args.generator_checkpoint_dir else 0,
            "graph_weight": 0.30,
            "evidence_weight": 0.60,
            "gate_weight": 0.10,
            "generated_evidence_score": 0.40,
        },
    ]
    for variant in rag_variants:
        pred_csv = out / f"{prefix}_ablation_{variant['name']}.csv"
        cand_csv = out / f"{prefix}_ablation_{variant['name']}_candidates.csv"
        cmd = [
            sys.executable,
            "scripts/generate_rag_primekg_reports.py",
            "--manifest",
            args.manifest,
            "--primekg-dir",
            args.primekg_dir,
            "--output-csv",
            str(pred_csv),
            "--candidates-csv",
            str(cand_csv),
            "--split",
            args.split,
            "--retrieval-top-k",
            str(variant["retrieval_top_k"]),
            "--decoding-mode",
            str(variant["decoding_mode"]),
            "--selection-objective",
            "hybrid",
            "--graph-score-weight",
            str(variant["graph_weight"]),
            "--evidence-weight",
            str(variant["evidence_weight"]),
            "--gate-weight",
            str(variant["gate_weight"]),
            "--generated-evidence-score",
            str(variant["generated_evidence_score"]),
            "--generator-num-beams",
            "8",
            "--generator-batch-size",
            str(args.batch_size),
            "--generator-num-beam-groups",
            "4",
            "--generator-diversity-penalty",
            "0.35",
            "--generator-repetition-penalty",
            "1.12",
            "--generator-no-repeat-ngram-size",
            "3",
            "--generator-length-penalty",
            "0.90",
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        if args.generator_checkpoint_dir:
            cmd += [
                "--generator-checkpoint-dir",
                args.generator_checkpoint_dir,
                "--generator-num-candidates",
                str(variant["generator_num_candidates"]),
            ]
        cmd = _with_limit(cmd, args.limit)
        commands.append(
            {
                "name": variant["name"],
                "purpose": variant["purpose"],
                "prediction_csv": str(pred_csv),
                "candidate_csv": str(cand_csv),
                "cmd": cmd,
            }
        )
        commands.append(_eval_command(args, variant["name"], pred_csv))

    return commands


def _eval_command(args, name: str, predictions_csv: Path) -> dict[str, object]:
    out = Path(args.output_dir)
    prefix = args.dataset_name
    return {
        "name": f"eval_{name}",
        "purpose": f"Evaluate {name} with lexical, entity, clinical-proxy, and graph-linked metrics.",
        "metrics_json": str(out / f"{prefix}_ablation_{name}_metrics.json"),
        "cmd": [
            sys.executable,
            "scripts/evaluate_generation.py",
            "--manifest",
            args.manifest,
            "--predictions-csv",
            str(predictions_csv),
            "--output-json",
            str(out / f"{prefix}_ablation_{name}_metrics.json"),
            "--nodes-csv",
            str(Path(args.primekg_dir) / "nodes.csv"),
            "--output-factuality-csv",
            str(out / f"{prefix}_ablation_{name}_entity_factuality.csv"),
            "--output-chexpert-csv",
            str(out / f"{prefix}_ablation_{name}_chexpert_quick.csv"),
            "--output-radgraph-csv",
            str(out / f"{prefix}_ablation_{name}_radgraph_quick.csv"),
        ],
    }


def compare_command(args, commands: list[dict[str, object]]) -> list[str]:
    systems: list[str] = []
    seen = set()
    for item in commands:
        csv_path = item.get("prediction_csv")
        if not csv_path:
            continue
        name = str(item["name"])
        if name in seen:
            continue
        seen.add(name)
        systems.extend(["--system", name, str(csv_path)])
    return [
        sys.executable,
        "scripts/compare_generation_systems.py",
        "--manifest",
        args.manifest,
        "--nodes-csv",
        str(Path(args.primekg_dir) / "nodes.csv"),
        "--output-json",
        str(Path(args.output_dir) / f"{args.dataset_name}_ablation_comparison.json"),
        *systems,
    ]


def _with_limit(cmd: list[str], limit: int | None) -> list[str]:
    if limit is None:
        return [str(part) for part in cmd]
    return [str(part) for part in [*cmd, "--limit", limit]]


if __name__ == "__main__":
    main()
