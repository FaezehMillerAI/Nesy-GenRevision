from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.schema import load_jsonl
from nesy_gen.evaluation.graph_verification import (
    select_graph_verified_candidate,
    verify_report_candidates,
)
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
        description="Generate Vision-T5 candidates and select final reports with PrimeKG verification."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--primekg-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--subgraph-strategy", choices=["steiner", "ego"], default="ego")
    parser.add_argument("--max-neighbors-per-node", type=int, default=250)
    parser.add_argument("--max-path-expansions", type=int, default=200_000)
    parser.add_argument("--min-graph-score", type=float)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--num-beam-groups", type=int, default=1)
    parser.add_argument("--diversity-penalty", type=float, default=0.0)
    args = parser.parse_args()

    deps = require_r2gen_t5_dependencies()
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]

    model = R2GenT5Model.from_pretrained(args.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("Loading PrimeKG verifier...", flush=True)
    pipeline = build_primekg_pipeline(
        args.primekg_dir,
        subgraph_strategy=args.subgraph_strategy,
        max_path_expansions=args.max_path_expansions,
        max_neighbors_per_node=args.max_neighbors_per_node,
    )

    examples = [example for example in load_jsonl(args.manifest) if example.split == args.split and example.image_path]
    if args.limit:
        examples = examples[: args.limit]
    loader = DataLoader(
        R2GenT5Dataset(
            examples,
            model.tokenizer,
            max_target_length=model.config.max_target_length,
            include_labels=False,
            target_prefix=model.config.target_prefix,
            image_size=model.config.image_size,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_r2gen_t5_batch,
    )
    example_by_id = {example.study_id: example for example in examples}
    selected_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Vision-T5 candidates + PrimeKG verify"):
            generated = model.generate(
                batch["image"].to(device),
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                num_return_sequences=args.num_candidates,
                do_sample=args.do_sample,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                length_penalty=args.length_penalty,
                num_beam_groups=args.num_beam_groups,
                diversity_penalty=args.diversity_penalty,
            )
            texts = decode_r2gen_predictions(
                model.tokenizer,
                generated,
                target_prefix=model.config.target_prefix,
            )
            for batch_index, study_id in enumerate(batch["study_id"]):
                start = batch_index * args.num_candidates
                end = start + args.num_candidates
                example = example_by_id[study_id]
                verified = verify_report_candidates(
                    pipeline,
                    indication=example.indication,
                    candidates=texts[start:end],
                )
                selected = select_graph_verified_candidate(
                    verified,
                    min_graph_score=args.min_graph_score,
                )
                for candidate in verified:
                    candidate_rows.append(
                        {
                            "study_id": study_id,
                            "reference": example.report,
                            **asdict(candidate),
                        }
                    )
                if selected is None:
                    fallback = texts[start] if start < len(texts) else ""
                    selected_rows.append(
                        {
                            "study_id": study_id,
                            "prediction": fallback,
                            "reference": example.report,
                            "selected_candidate_rank": 0,
                            "graph_score": 0.0,
                            "num_links": 0,
                            "bio_temporal": 0.0,
                            "finding_to_diagnosis": 0.0,
                            "located_in_type": 0.0,
                            "selection_status": "fallback_unverified",
                        }
                    )
                    continue
                selected_rows.append(
                    {
                        "study_id": study_id,
                        "prediction": selected.prediction,
                        "reference": example.report,
                        "selected_candidate_rank": selected.candidate_rank,
                        "graph_score": selected.graph_score,
                        "num_links": selected.num_links,
                        "bio_temporal": selected.bio_temporal,
                        "finding_to_diagnosis": selected.finding_to_diagnosis,
                        "located_in_type": selected.located_in_type,
                        "selection_status": "graph_selected",
                    }
                )

    selected_out = Path(args.output_csv)
    candidates_out = Path(args.candidates_csv)
    selected_out.parent.mkdir(parents=True, exist_ok=True)
    candidates_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected_rows).to_csv(selected_out, index=False)
    pd.DataFrame(candidate_rows).to_csv(candidates_out, index=False)
    print(f"Saved {len(selected_rows)} graph-verified Vision-T5 predictions to {selected_out}")
    print(f"Saved {len(candidate_rows)} Vision-T5 candidate audit rows to {candidates_out}")


def build_primekg_pipeline(
    primekg_dir: str | Path,
    *,
    subgraph_strategy: str,
    max_path_expansions: int,
    max_neighbors_per_node: int,
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
        auditor=NeuroSymbolicAuditor(beta_accept=0.65, gamma_flag=0.50),
        gate=ConsistencyGate(min_grounding=0.30, max_hallucination=0.50, min_entailment=0.50),
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
