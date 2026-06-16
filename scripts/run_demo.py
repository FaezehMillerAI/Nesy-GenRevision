from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.kg.entity_linking import LexicalEntityLinker
from nesy_gen.kg.primekg import PrimeKGGraph
from nesy_gen.kg.temporal import TemporalSubgraphBuilder
from nesy_gen.logic.ltn import NeuroSymbolicAuditor
from nesy_gen.models.gate import CandidateToken, ConsistencyGate
from nesy_gen.models.nesy_gen import NesyGenPipeline
from nesy_gen.evaluation.metrics import entity_f1, hallucination_rate
from nesy_gen.evaluation.profiling import measure_latency


def build_toy_pipeline() -> NesyGenPipeline:
    edges = pd.DataFrame(
        [
            {
                "x_id": "D:pneumonia",
                "x_name": "pneumonia",
                "x_type": "disease",
                "y_id": "P:consolidation",
                "y_name": "consolidation",
                "y_type": "phenotype",
                "display_relation": "causes",
                "confidence": 0.91,
            },
            {
                "x_id": "P:consolidation",
                "x_name": "consolidation",
                "x_type": "phenotype",
                "y_id": "A:right_lower_lobe",
                "y_name": "right lower lobe",
                "y_type": "anatomy",
                "display_relation": "located_in",
                "confidence": 0.88,
            },
            {
                "x_id": "D:heart_failure",
                "x_name": "heart failure",
                "x_type": "disease",
                "y_id": "P:pleural_effusion",
                "y_name": "pleural effusion",
                "y_type": "phenotype",
                "display_relation": "associated_with",
                "confidence": 0.82,
            },
        ]
    )
    kg = PrimeKGGraph.from_dataframe(edges)
    vocab = pd.DataFrame(
        [
            {"node_id": "D:pneumonia", "node_name": "pneumonia", "node_type": "disease", "alias": "pneumonia"},
            {
                "node_id": "P:consolidation",
                "node_name": "consolidation",
                "node_type": "phenotype",
                "alias": "consolidation",
            },
            {
                "node_id": "A:right_lower_lobe",
                "node_name": "right lower lobe",
                "node_type": "anatomy",
                "alias": "right lower lobe",
            },
            {
                "node_id": "P:pleural_effusion",
                "node_name": "pleural effusion",
                "node_type": "phenotype",
                "alias": "pleural effusion",
            },
        ]
    )
    return NesyGenPipeline(
        linker=LexicalEntityLinker(vocab),
        subgraph_builder=TemporalSubgraphBuilder(kg),
        auditor=NeuroSymbolicAuditor(beta_accept=0.65, gamma_flag=0.50),
        gate=ConsistencyGate(min_grounding=0.30, max_hallucination=0.50, min_entailment=0.50),
    )


def main() -> None:
    pipeline = build_toy_pipeline()
    indication = "fever and concern for pneumonia"
    draft = "right lower lobe consolidation without pleural effusion"
    links, audit = pipeline.reason(indication, draft)

    tokens = [
        CandidateToken("consolidation", "P:consolidation", 0.76, 0.12, 0.91),
        CandidateToken("pleural effusion", "P:pleural_effusion", 0.20, 0.61, 0.40),
    ]
    decisions = [pipeline.gate.decide(token, audit) for token in tokens]
    predicted = [decision.token.node_id for decision in decisions if decision.accepted and decision.token.node_id]
    reference = ["P:consolidation", "A:right_lower_lobe"]

    print("Linked entities:", [(link.node_name, link.node_id, link.mention.negated) for link in links])
    print("Clause scores:", audit.scores.as_dict())
    print("Decisions:", [(d.token.text, d.accepted, d.reason, round(d.confidence, 3)) for d in decisions])
    print("Entity F1:", entity_f1(predicted, reference))
    print("Hallucination:", hallucination_rate(predicted, reference))
    print("Latency:", measure_latency(lambda: pipeline.reason(indication, draft), repeats=5))


if __name__ == "__main__":
    main()
