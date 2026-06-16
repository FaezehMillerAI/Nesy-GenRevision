from __future__ import annotations

from dataclasses import dataclass

from nesy_gen.kg.simple_graph import SimpleDiGraph
from nesy_gen.logic.constraints import ClauseScores, compute_clause_scores


@dataclass(frozen=True, slots=True)
class AuditReport:
    scores: ClauseScores
    valid_nodes: set[str]
    flagged_nodes: set[str]
    rejected_nodes: set[str]

    @property
    def mean_satisfaction(self) -> float:
        return self.scores.mean


@dataclass(slots=True)
class NeuroSymbolicAuditor:
    beta_accept: float = 0.65
    gamma_flag: float = 0.50

    def audit(self, graph: SimpleDiGraph) -> AuditReport:
        scores = compute_clause_scores(graph)
        valid: set[str] = set()
        flagged: set[str] = set()
        rejected: set[str] = set()
        for node in graph.nodes:
            local = self._local_node_score(graph, node, scores.mean)
            if local >= self.beta_accept:
                valid.add(str(node))
            elif local >= self.gamma_flag:
                flagged.add(str(node))
            else:
                rejected.add(str(node))
        return AuditReport(scores=scores, valid_nodes=valid, flagged_nodes=flagged, rejected_nodes=rejected)

    def _local_node_score(self, graph: SimpleDiGraph, node: str, fallback: float) -> float:
        edge_truths = []
        for _, _, attrs in graph.in_edges(node, data=True):
            edge_truths.append(float(attrs.get("confidence", fallback)))
        for _, _, attrs in graph.out_edges(node, data=True):
            edge_truths.append(float(attrs.get("confidence", fallback)))
        if not edge_truths:
            return fallback
        return min(fallback, sum(edge_truths) / len(edge_truths))
