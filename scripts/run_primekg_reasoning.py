from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.schema import load_jsonl
from nesy_gen.evaluation.reasoning import (
    measure_pipeline_latency,
    run_reasoning_batch,
    save_reasoning_artifacts,
)
from nesy_gen.kg.entity_linking import LexicalEntityLinker
from nesy_gen.kg.primekg import PrimeKGGraph
from nesy_gen.kg.temporal import TemporalSubgraphBuilder
from nesy_gen.logic.ltn import NeuroSymbolicAuditor
from nesy_gen.models.gate import ConsistencyGate
from nesy_gen.models.nesy_gen import NesyGenPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PrimeKG neuro-symbolic reasoning.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--primekg-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-path-expansions", type=int, default=200_000)
    parser.add_argument("--subgraph-strategy", choices=["steiner", "ego"], default="steiner")
    parser.add_argument("--max-neighbors-per-node", type=int, default=250)
    parser.add_argument("--latency-repeats", type=int, default=1)
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    examples = [example for example in load_jsonl(args.manifest) if example.split == args.split]
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError(f"No examples found for split={args.split}")

    primekg_dir = Path(args.primekg_dir)
    print(f"Loading PrimeKG from {primekg_dir}...", flush=True)
    kg = PrimeKGGraph.from_dataverse_dir(primekg_dir)
    print(f"PrimeKG loaded: {len(list(kg.graph.nodes))} nodes, {len(kg.graph.edges())} edges", flush=True)
    nodes_path = primekg_dir / "nodes.csv"
    if nodes_path.exists():
        print(f"Loading node vocabulary from {nodes_path}...", flush=True)
        nodes = pd.read_csv(nodes_path)
        vocab = nodes.rename(
            columns={"node_id": "node_id", "node_name": "node_name", "node_type": "node_type"}
        )[["node_id", "node_name", "node_type"]]
        vocab["alias"] = vocab["node_name"]
    else:
        print("Building node vocabulary from edge list...", flush=True)
        vocab = _vocab_from_edges(kg.edges)
    print(f"Vocabulary rows: {len(vocab)}", flush=True)

    print("Building pipeline...", flush=True)
    pipeline = NesyGenPipeline(
        linker=LexicalEntityLinker(vocab),
        subgraph_builder=TemporalSubgraphBuilder(
            kg,
            max_path_expansions=args.max_path_expansions,
            strategy=args.subgraph_strategy,
            max_neighbors_per_node=args.max_neighbors_per_node,
        ),
        auditor=NeuroSymbolicAuditor(beta_accept=0.65, gamma_flag=0.50),
        gate=ConsistencyGate(min_grounding=0.30, max_hallucination=0.50, min_entailment=0.50),
    )

    latency = {}
    if not args.skip_latency:
        print("Measuring latency...", flush=True)
        latency = measure_pipeline_latency(pipeline, examples[0], warmup=1, repeats=args.latency_repeats)
        print(f"Latency: {latency}", flush=True)
    rows = []
    for example in tqdm(examples, desc="PrimeKG reasoning"):
        rows.extend(run_reasoning_batch(pipeline, [example]))

    prefix = f"{args.dataset_name}_{args.split}"
    if args.limit is not None:
        prefix += f"_n{args.limit}"
    paths = save_reasoning_artifacts(rows, args.output_dir, prefix=prefix, latency=latency)
    print("Latency:", latency)
    for key, value in paths.items():
        print(f"{key}: {value}")


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
