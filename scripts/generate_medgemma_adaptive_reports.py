from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.agents.adaptive_verification import AdaptiveClaimVerifier  # noqa: E402
from nesy_gen.baselines.medsiglip_retrieval import MedSiglipRetriever  # noqa: E402
from nesy_gen.data.schema import load_jsonl  # noqa: E402
from nesy_gen.generation.rag import RagCandidate  # noqa: E402
from nesy_gen.models.medgemma import MedGemmaDrafter  # noqa: E402
from nesy_gen.models.pipeline_factory import build_primekg_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training-free MedGemma drafting with adaptive PrimeKG/LTN verification."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--primekg-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--claim-trace-jsonl", required=True)
    parser.add_argument("--claim-audit-csv", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--draft-mode", choices=["zero_shot", "few_shot"], default="few_shot")
    parser.add_argument("--medgemma-model", default="google/medgemma-4b-it")
    parser.add_argument("--medsiglip-model", default="google/medsiglip-448")
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--few-shot-k", type=int, default=3)
    parser.add_argument("--retrieval-batch-size", type=int, default=16)
    parser.add_argument("--max-retrieval-index-examples", type=int)
    parser.add_argument("--retrieval-cache")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--fast-accept-threshold", type=float, default=0.85)
    parser.add_argument("--min-supporting-reports", type=int, default=2)
    parser.add_argument("--claim-revise-threshold", type=float, default=0.50)
    parser.add_argument(
        "--revision-policy", choices=["audit_only", "evidence_replace"], default="evidence_replace"
    )
    args = parser.parse_args()

    examples = load_jsonl(args.manifest)
    train = [example for example in examples if example.split == "train" and example.image_path]
    queries = [
        example for example in examples if example.split == args.split and example.image_path
    ]
    if args.limit:
        queries = queries[: args.limit]
    if args.max_retrieval_index_examples:
        train = train[: args.max_retrieval_index_examples]
    if not train or not queries:
        raise ValueError("Need non-empty image-bearing train and query splits.")

    print("Loading frozen MedSigLIP retriever...", flush=True)
    retriever = MedSiglipRetriever(args.medsiglip_model)
    retrieved = retriever.retrieve(
        train,
        queries,
        top_k=args.retrieval_top_k,
        batch_size=args.retrieval_batch_size,
        cache_path=args.retrieval_cache,
    )
    del retriever
    _empty_cuda_cache()

    print("Loading MedGemma drafting agent...", flush=True)
    drafter = MedGemmaDrafter(args.medgemma_model)
    print("Loading adaptive PrimeKG/LTN verifier...", flush=True)
    pipeline = build_primekg_pipeline(args.primekg_dir)
    verifier = AdaptiveClaimVerifier(
        pipeline,
        fast_accept_threshold=args.fast_accept_threshold,
        min_supporting_reports=args.min_supporting_reports,
        revise_threshold=args.claim_revise_threshold,
        revision_policy=args.revision_policy,
    )

    selected_rows = []
    candidate_rows = []
    trace_rows = []
    claim_rows = []
    for example, neighbours in tqdm(
        zip(queries, retrieved, strict=True),
        total=len(queries),
        desc="MedGemma + adaptive NeSy",
    ):
        rag_candidates = [
            RagCandidate(
                source="medsiglip_retrieval",
                source_rank=row.rank,
                prediction=row.prediction,
                evidence_score=max(0.0, min(1.0, row.similarity)),
                retrieved_study_id=row.retrieved_study_id,
            )
            for row in neighbours
        ]
        evidence_reports = (
            [candidate.prediction for candidate in rag_candidates[: args.few_shot_k]]
            if args.draft_mode == "few_shot"
            else []
        )
        draft = drafter.draft(
            example.image_path,
            indication=example.indication,
            evidence_reports=evidence_reports,
            max_new_tokens=args.max_new_tokens,
        )
        visual_support = max((candidate.evidence_score for candidate in rag_candidates), default=0.0)
        result = verifier.verify(
            draft,
            indication=example.indication,
            visual_support=visual_support,
            evidence_candidates=rag_candidates,
        )
        selected_rows.append(
            {
                "study_id": example.study_id,
                "prediction": result.final_report,
                "raw_prediction": draft,
                "reference": example.report,
                "source": f"medgemma_{args.draft_mode}",
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
        candidate_rows.extend(
            {
                "study_id": example.study_id,
                "source": candidate.source,
                "source_rank": candidate.source_rank,
                "retrieved_study_id": candidate.retrieved_study_id,
                "prediction": candidate.prediction,
                "evidence_score": candidate.evidence_score,
            }
            for candidate in rag_candidates
        )
        trace_rows.append({"study_id": example.study_id, **result.as_dict()})
        claim_rows.extend(
            {"study_id": example.study_id, **claim.as_dict()} for claim in result.claims
        )

    _write_csv(args.output_csv, selected_rows)
    _write_csv(args.candidates_csv, candidate_rows)
    _write_csv(args.claim_audit_csv, claim_rows)
    trace_path = Path(args.claim_trace_jsonl)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as handle:
        for row in trace_rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Saved {len(selected_rows)} predictions to {args.output_csv}")
    print(f"Saved faithful claim traces to {trace_path}")


def _write_csv(path, rows) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
