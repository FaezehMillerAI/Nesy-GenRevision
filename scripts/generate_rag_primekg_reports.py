from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.schema import RadiologyExample, load_jsonl
from nesy_gen.baselines.visual_retrieval import run_visual_retrieval_topk
from nesy_gen.generation.constrained_decoding import PrimeKGDecodingConstraintBuilder
from nesy_gen.agents.adaptive_verification import AdaptiveClaimVerifier
from nesy_gen.generation.rag import (
    RagCandidate,
    retrieval_candidates,
    select_agentic_draft,
    select_primekg_verified_report,
)
from nesy_gen.models.pipeline_factory import build_primekg_pipeline
from nesy_gen.models.r2gen_t5 import (
    R2GenT5Dataset,
    R2GenT5Model,
    collate_r2gen_t5_batch,
    decode_r2gen_predictions,
    require_r2gen_t5_dependencies,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAG-based report generation with PrimeKG LTN reasoning and consistency gate."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--primekg-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument(
        "--retrieval-mode",
        choices=["metadata", "visual"],
        default="metadata",
        help="Visual mode is reference-blind and recommended when indications are unavailable.",
    )
    parser.add_argument("--r2gen-checkpoint-dir", dest="generator_checkpoint_dir")
    parser.add_argument("--generator-checkpoint-dir", dest="generator_checkpoint_dir")
    parser.add_argument("--r2gen-num-candidates", "--generator-num-candidates", dest="generator_num_candidates", type=int, default=0)
    parser.add_argument("--generated-evidence-score", type=float, default=0.50)
    parser.add_argument("--r2gen-num-beams", "--generator-num-beams", dest="generator_num_beams", type=int, default=6)
    parser.add_argument("--r2gen-batch-size", "--generator-batch-size", dest="generator_batch_size", type=int, default=2)
    parser.add_argument("--generator-do-sample", action="store_true")
    parser.add_argument("--generator-top-p", type=float, default=0.9)
    parser.add_argument("--generator-temperature", type=float, default=0.8)
    parser.add_argument("--generator-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--generator-no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--generator-length-penalty", type=float, default=1.0)
    parser.add_argument("--generator-num-beam-groups", type=int, default=1)
    parser.add_argument("--generator-diversity-penalty", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument(
        "--decoding-mode",
        choices=["standard", "graph_constrained"],
        default="standard",
        help="Use standard Vision-T5 decoding or soft PrimeKG-constrained decoding.",
    )
    parser.add_argument("--graph-token-boost", type=float, default=2.0)
    parser.add_argument("--unsupported-token-penalty", type=float, default=0.0)
    parser.add_argument("--constraint-max-terms", type=int, default=2500)
    parser.add_argument("--subgraph-strategy", choices=["steiner", "ego"], default="ego")
    parser.add_argument("--max-neighbors-per-node", type=int, default=250)
    parser.add_argument("--max-path-expansions", type=int, default=200_000)
    parser.add_argument("--min-graph-score", type=float)
    parser.add_argument(
        "--selection-objective",
        choices=["graph", "evidence", "hybrid"],
        default="graph",
        help="Candidate selection objective after PrimeKG/LTN verification.",
    )
    parser.add_argument("--graph-score-weight", type=float, default=0.55)
    parser.add_argument("--evidence-weight", type=float, default=0.35)
    parser.add_argument("--gate-weight", type=float, default=0.10)
    parser.add_argument("--beta-accept", type=float, default=0.65)
    parser.add_argument("--gamma-flag", type=float, default=0.50)
    parser.add_argument("--min-grounding", type=float, default=0.30)
    parser.add_argument("--max-hallucination", type=float, default=0.50)
    parser.add_argument("--min-entailment", type=float, default=0.50)
    parser.add_argument(
        "--verification-mode",
        choices=["report", "adaptive_claim"],
        default="report",
        help="Verify all report candidates or adaptively escalate uncertain claims only.",
    )
    parser.add_argument("--claim-trace-jsonl")
    parser.add_argument("--claim-audit-csv")
    parser.add_argument("--fast-accept-threshold", type=float, default=0.85)
    parser.add_argument("--min-supporting-reports", type=int, default=2)
    parser.add_argument("--claim-revise-threshold", type=float, default=0.50)
    parser.add_argument(
        "--revision-policy",
        choices=["audit_only", "evidence_replace"],
        default="evidence_replace",
    )
    parser.add_argument("--adaptive-disable-ltn", action="store_true")
    parser.add_argument("--adaptive-disable-gate", action="store_true")
    args = parser.parse_args()

    examples = load_jsonl(args.manifest)
    train = [example for example in examples if example.split == "train"]
    queries = [example for example in examples if example.split == args.split]
    if args.limit:
        queries = queries[: args.limit]
    if not train or not queries:
        raise ValueError("Need non-empty train examples and query examples.")

    generator_model = None
    if args.generator_checkpoint_dir:
        generator_model = R2GenT5Model.from_pretrained(args.generator_checkpoint_dir)
        deps = require_r2gen_t5_dependencies()
        torch = deps["torch"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator_model.to(device)
        generator_model.eval()

    print(f"Building {args.retrieval_mode} retrieval candidates...", flush=True)
    if args.retrieval_mode == "visual":
        if generator_model is None:
            raise ValueError("Visual retrieval requires --generator-checkpoint-dir.")
        retrieved = run_visual_retrieval_topk(
            generator_model,
            train,
            queries,
            top_k=args.retrieval_top_k,
            batch_size=args.generator_batch_size,
        )
        candidate_map = {
            example.study_id: [
                RagCandidate(
                    source="visual_retrieval",
                    source_rank=row.rank,
                    prediction=row.prediction,
                    evidence_score=max(0.0, row.similarity),
                    retrieved_study_id=row.retrieved_study_id,
                )
                for row in rows
            ]
            for example, rows in zip(queries, retrieved, strict=True)
        }
    else:
        candidate_map = retrieval_candidates(train, queries, top_k=args.retrieval_top_k)

    if args.generator_checkpoint_dir and args.generator_num_candidates > 0:
        print("Adding Vision-T5 generated candidates...", flush=True)
        generated_map = generate_r2gen_candidates(
            args.generator_checkpoint_dir,
            queries,
            batch_size=args.generator_batch_size,
            num_candidates=args.generator_num_candidates,
            num_beams=args.generator_num_beams,
            do_sample=args.generator_do_sample,
            top_p=args.generator_top_p,
            temperature=args.generator_temperature,
            repetition_penalty=args.generator_repetition_penalty,
            no_repeat_ngram_size=args.generator_no_repeat_ngram_size,
            length_penalty=args.generator_length_penalty,
            num_beam_groups=args.generator_num_beam_groups,
            diversity_penalty=args.generator_diversity_penalty,
            max_new_tokens=args.max_new_tokens,
            decoding_mode=args.decoding_mode,
            primekg_dir=args.primekg_dir,
            retrieval_candidate_map=candidate_map,
            graph_token_boost=args.graph_token_boost,
            unsupported_token_penalty=args.unsupported_token_penalty,
            constraint_max_terms=args.constraint_max_terms,
            generated_evidence_score=args.generated_evidence_score,
            preloaded_model=generator_model,
        )
        for study_id, candidates in generated_map.items():
            candidate_map.setdefault(study_id, []).extend(candidates)

    print("Loading PrimeKG LTN verifier...", flush=True)
    pipeline = build_primekg_pipeline(
        args.primekg_dir,
        subgraph_strategy=args.subgraph_strategy,
        max_path_expansions=args.max_path_expansions,
        max_neighbors_per_node=args.max_neighbors_per_node,
        beta_accept=args.beta_accept,
        gamma_flag=args.gamma_flag,
        min_grounding=args.min_grounding,
        max_hallucination=args.max_hallucination,
        min_entailment=args.min_entailment,
    )

    selected_rows = []
    candidate_rows = []
    trace_rows = []
    claim_rows = []
    adaptive_verifier = None
    if args.verification_mode == "adaptive_claim":
        adaptive_verifier = AdaptiveClaimVerifier(
            pipeline,
            fast_accept_threshold=args.fast_accept_threshold,
            min_supporting_reports=args.min_supporting_reports,
            revise_threshold=args.claim_revise_threshold,
            revision_policy=args.revision_policy,
            use_ltn=not args.adaptive_disable_ltn,
            use_gate=not args.adaptive_disable_gate,
        )

    for example in tqdm(queries, desc="Adaptive NeSy verification"):
        study_candidates = candidate_map.get(example.study_id, [])
        if adaptive_verifier is None:
            selected, candidates = select_primekg_verified_report(
                pipeline,
                example,
                study_candidates,
                min_graph_score=args.min_graph_score,
                selection_objective=args.selection_objective,
                graph_score_weight=args.graph_score_weight,
                evidence_weight=args.evidence_weight,
                gate_weight=args.gate_weight,
            )
        else:
            selected, candidates = select_agentic_draft(example, study_candidates)
            raw_prediction = str(selected["prediction"])
            visual_support = (
                float(selected.get("evidence_score", 0.0))
                if selected.get("source") == "visual_retrieval"
                else 0.0
            )
            result = adaptive_verifier.verify(
                raw_prediction,
                indication=example.indication,
                visual_support=visual_support,
                evidence_candidates=study_candidates,
            )
            selected.update(
                {
                    "raw_prediction": raw_prediction,
                    "prediction": result.final_report,
                    "selection_status": "adaptive_claim_verified",
                    "accepted_claims": result.accepted_claims,
                    "revised_claims": result.revised_claims,
                    "flagged_claims": result.flagged_claims,
                    "graph_calls": result.graph_calls,
                    "total_claims": result.total_claims,
                    "escalation_rate": result.escalation_rate,
                    "adaptive_latency_ms": result.latency_ms,
                }
            )
            trace = {"study_id": example.study_id, **result.as_dict()}
            trace_rows.append(trace)
            for claim in result.claims:
                claim_rows.append({"study_id": example.study_id, **claim.as_dict()})
        selected_rows.append(selected)
        candidate_rows.extend(candidates)

    output_csv = Path(args.output_csv)
    candidates_csv = Path(args.candidates_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected_rows).to_csv(output_csv, index=False)
    pd.DataFrame(candidate_rows).to_csv(candidates_csv, index=False)
    if trace_rows:
        trace_path = Path(args.claim_trace_jsonl or output_csv.with_name(f"{output_csv.stem}_claims.jsonl"))
        audit_path = Path(args.claim_audit_csv or output_csv.with_name(f"{output_csv.stem}_claims.csv"))
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8") as handle:
            for row in trace_rows:
                handle.write(json.dumps(row) + "\n")
        pd.DataFrame(claim_rows).to_csv(audit_path, index=False)
        print(f"Saved {len(trace_rows)} faithful explanation traces to {trace_path}")
        print(f"Saved {len(claim_rows)} claim decisions to {audit_path}")
    print(f"Saved {len(selected_rows)} final RAG+PrimeKG reports to {output_csv}")
    print(f"Saved {len(candidate_rows)} candidate audits to {candidates_csv}")


def generate_r2gen_candidates(
    checkpoint_dir: str | Path,
    examples: list[RadiologyExample],
    *,
    batch_size: int,
    num_candidates: int,
    num_beams: int,
    max_new_tokens: int,
    do_sample: bool = False,
    top_p: float = 0.9,
    temperature: float = 0.8,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    length_penalty: float = 1.0,
    num_beam_groups: int = 1,
    diversity_penalty: float = 0.0,
    decoding_mode: str = "standard",
    primekg_dir: str | Path | None = None,
    retrieval_candidate_map: dict[str, list[RagCandidate]] | None = None,
    graph_token_boost: float = 2.0,
    unsupported_token_penalty: float = 0.0,
    constraint_max_terms: int = 2500,
    generated_evidence_score: float = 0.50,
    preloaded_model=None,
) -> dict[str, list[RagCandidate]]:
    deps = require_r2gen_t5_dependencies()
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]
    LogitsProcessorList = deps["LogitsProcessorList"]
    model = preloaded_model or R2GenT5Model.from_pretrained(checkpoint_dir)
    device = model.device if preloaded_model is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if preloaded_model is None:
        model.to(device)
        model.eval()
    example_by_id = {example.study_id: example for example in examples}
    constraint_builder = None
    if decoding_mode == "graph_constrained":
        if primekg_dir is None:
            raise ValueError("graph_constrained decoding requires primekg_dir.")
        nodes_path = Path(primekg_dir) / "nodes.csv"
        if not nodes_path.exists():
            raise FileNotFoundError("graph_constrained decoding requires PrimeKG nodes.csv.")
        nodes = pd.read_csv(nodes_path)
        constraint_builder = PrimeKGDecodingConstraintBuilder(
            nodes,
            model.tokenizer,
            max_penalty_terms=constraint_max_terms,
        )
    loader = DataLoader(
        R2GenT5Dataset(
            examples,
            model.tokenizer,
            max_target_length=model.config.max_target_length,
            include_labels=False,
            target_prefix=model.config.target_prefix,
            image_size=model.config.image_size,
            evidence_by_study_id=(
                {
                    study_id: [candidate.prediction for candidate in candidates]
                    for study_id, candidates in (retrieval_candidate_map or {}).items()
                }
                if model.config.use_retrieval_conditioning
                else None
            ),
            max_evidence_length=model.config.max_evidence_length,
            evidence_prefix=model.config.evidence_prefix,
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_r2gen_t5_batch,
    )
    rows: dict[str, list[RagCandidate]] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Vision-T5 RAG candidates"):
            logits_processor = None
            if constraint_builder is not None:
                evidence_texts = [
                    _constraint_evidence_text(
                        example_by_id[study_id],
                        (retrieval_candidate_map or {}).get(study_id, []),
                    )
                    for study_id in batch["study_id"]
                ]
                logits_processor = LogitsProcessorList(
                    [
                        constraint_builder.processor(
                            evidence_texts,
                            token_boost=graph_token_boost,
                            unsupported_token_penalty=unsupported_token_penalty,
                        )
                    ]
                )
            generated = model.generate(
                batch["image"].to(device),
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_candidates,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty,
                logits_processor=logits_processor,
                evidence_input_ids=batch.get("evidence_input_ids"),
                evidence_attention_mask=batch.get("evidence_attention_mask"),
            )
            texts = decode_r2gen_predictions(
                model.tokenizer,
                generated,
                target_prefix=model.config.target_prefix,
            )
            for batch_index, study_id in enumerate(batch["study_id"]):
                start = batch_index * num_candidates
                end = start + num_candidates
                rows[study_id] = [
                    RagCandidate(
                        source="vision_t5",
                        source_rank=rank,
                        prediction=prediction,
                        evidence_score=generated_evidence_score,
                    )
                    for rank, prediction in enumerate(texts[start:end], start=1)
                    if prediction
                ]
    return rows


def _constraint_evidence_text(
    example: RadiologyExample,
    retrieval_candidates: list[RagCandidate],
) -> str:
    evidence_parts = [example.indication]
    evidence_parts.extend(
        candidate.prediction for candidate in retrieval_candidates if candidate.evidence_score > 0.0
    )
    return " ".join(part for part in evidence_parts if part)


if __name__ == "__main__":
    main()
