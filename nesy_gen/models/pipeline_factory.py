from __future__ import annotations

from pathlib import Path

import pandas as pd

from nesy_gen.kg.entity_linking import LexicalEntityLinker
from nesy_gen.kg.primekg import PrimeKGGraph
from nesy_gen.kg.temporal import TemporalSubgraphBuilder
from nesy_gen.logic.ltn import NeuroSymbolicAuditor
from nesy_gen.models.gate import ConsistencyGate
from nesy_gen.models.nesy_gen import NesyGenPipeline


def build_primekg_pipeline(
    primekg_dir: str | Path,
    *,
    subgraph_strategy: str = "hybrid",
    max_path_expansions: int = 200_000,
    max_neighbors_per_node: int = 250,
    beta_accept: float = 0.65,
    gamma_flag: float = 0.50,
    min_grounding: float = 0.30,
    max_hallucination: float = 0.50,
    min_entailment: float = 0.50,
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
    vocab = pd.concat([left, right], ignore_index=True).drop_duplicates("node_id")
    vocab["alias"] = vocab["node_name"]
    return vocab
