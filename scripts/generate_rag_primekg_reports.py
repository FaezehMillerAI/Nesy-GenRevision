from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.schema import RadiologyExample, load_jsonl
from nesy_gen.generation.constrained_decoding import PrimeKGDecodingConstraintBuilder
from nesy_gen.generation.rag import RagCandidate, retrieval_candidates, select_primekg_verified_report
from nesy_gen.kg.entity_linking import LexicalEntityLinker
from nesy_gen.kg.primekg import PrimeKGGraph
from nesy_gen.kg.temporal import TemporalSubgraphBuilder
from nesy_gen.logic.ltn import NeuroSymbolicAuditor
from nesy_gen.models.gate import ConsistencyGate
from nesy_gen.models.nesy_gen import NesyGenPipeline
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
    parser.add_argument("--r2gen-checkpoint-dir")
    parser.add_argument("--r2gen-num-candidates", type=int, default=0)
    parser.add_argument("--generated-evidence-score", type=float, default=0.50)
    parser.add_argument("--r2gen-num-beams", type=int, default=6)
    parser.add_argument("--r2gen-batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument(
        "--decoding-mode",
        choices=["standard", "graph_constrained"],
        default="standard",
        help="Use standard R2Gen decoding or soft PrimeKG-constrained decoding.",
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
    args = parser.parse_args()

    examples = load_jsonl(args.manifest)
    train = [example for example in examples if example.split == "train"]
    queries = [example for example in examples if example.split == args.split]
    if args.limit:
        queries = queries[: args.limit]
    if not train or not queries:
        raise ValueError("Need non-empty train examples and query examples.")

    print("Building retrieval candidates...", flush=True)
    candidate_map = retrieval_candidates(train, queries, top_k=args.retrieval_top_k)

    if args.r2gen_checkpoint_dir and args.r2gen_num_candidates > 0:
        print("Adding R2Gen-T5 generated candidates...", flush=True)
        generated_map = generate_r2gen_candidates(
            args.r2gen_checkpoint_dir,
            queries,
            batch_size=args.r2gen_batch_size,
            num_candidates=args.r2gen_num_candidates,
            num_beams=args.r2gen_num_beams,
            max_new_tokens=args.max_new_tokens,
            decoding_mode=args.decoding_mode,
            primekg_dir=args.primekg_dir,
            retrieval_candidate_map=candidate_map,
            graph_token_boost=args.graph_token_boost,
            unsupported_token_penalty=args.unsupported_token_penalty,
            constraint_max_terms=args.constraint_max_terms,
            generated_evidence_score=args.generated_evidence_score,
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
    for example in tqdm(queries, desc="RAG + PrimeKG LTN gate"):
        selected, candidates = select_primekg_verified_report(
            pipeline,
            example,
            candidate_map.get(example.study_id, []),
            min_graph_score=args.min_graph_score,
            selection_objective=args.selection_objective,
            graph_score_weight=args.graph_score_weight,
            evidence_weight=args.evidence_weight,
            gate_weight=args.gate_weight,
        )
        selected_rows.append(selected)
        candidate_rows.extend(candidates)

    output_csv = Path(args.output_csv)
    candidates_csv = Path(args.candidates_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected_rows).to_csv(output_csv, index=False)
    pd.DataFrame(candidate_rows).to_csv(candidates_csv, index=False)
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
    decoding_mode: str = "standard",
    primekg_dir: str | Path | None = None,
    retrieval_candidate_map: dict[str, list[RagCandidate]] | None = None,
    graph_token_boost: float = 2.0,
    unsupported_token_penalty: float = 0.0,
    constraint_max_terms: int = 2500,
    generated_evidence_score: float = 0.50,
) -> dict[str, list[RagCandidate]]:
    deps = require_r2gen_t5_dependencies()
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]
    LogitsProcessorList = deps["LogitsProcessorList"]
    model = R2GenT5Model.from_pretrained(checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_r2gen_t5_batch,
    )
    rows: dict[str, list[RagCandidate]] = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="R2Gen-T5 RAG candidates"):
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
                logits_processor=logits_processor,
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
                        source="r2gen_t5",
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


def build_primekg_pipeline(
    primekg_dir: str | Path,
    *,
    subgraph_strategy: str,
    max_path_expansions: int,
    max_neighbors_per_node: int,
    beta_accept: float,
    gamma_flag: float,
    min_grounding: float,
    max_hallucination: float,
    min_entailment: float,
) -> NesyGenPipeline:
    primekg_dir = Path(primekg_dir)
    kg = PrimeKGGraph.from_dataverse_dir(primekg_dir)
    nodes_path = primekg_dir / "nodes.csv"
    if nodes_path.exists():
        nodes = pd.read_csv(nodes_path)
        vocab = nodes[["node_id", "node_name", "node_type"]].copy()
        vocab["alias"] = vocab["node_name"]
    else:
        vocab = _vocab_from_edges(kg.edges)
    return NesyGenPipeline(
        linker=LexicalEntityLinker(vocab),
        subgraph_builder=TemporalSubgraphBuilder(
            kg,
            max_path_expansions=max_path_expansions,
            strategy=subgraph_strategy,
            max_neighbors_per_node=max_neighbors_per_node,
        ),
        auditor=NeuroSymbolicAuditor(beta_accept=beta_accept, gamma_flag=gamma_flag),
        gate=ConsistencyGate(
            min_grounding=min_grounding,
            max_hallucination=max_hallucination,
            min_entailment=min_entailment,
        ),
    )


def _vocab_from_edges(edges: pd.DataFrame) -> pd.DataFrame:
    left = edges.rename(
        columns={"source_id": "node_id", "source_name": "node_name", "source_type": "node_type"}
    )[["node_id", "node_name", "node_type"]]
    right = edges.rename(
        columns={"target_id": "node_id", "target_name": "node_name", "target_type": "node_type"}
    )[["node_id", "node_name", "node_type"]]
    vocab = pd.concat([left, right]).drop_duplicates()
    vocab["alias"] = vocab["node_name"]
    return vocab


if __name__ == "__main__":
    main()
