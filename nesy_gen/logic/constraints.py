from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from nesy_gen.kg.simple_graph import SimpleDiGraph


DEFAULT_TYPE_COMPATIBILITY = {
    ("disease", "phenotype"),
    ("phenotype", "disease"),
    ("phenotype", "anatomy"),
    ("disease", "anatomy"),
    ("anatomy", "phenotype"),
    ("drug", "disease"),
    ("biological_process", "disease"),
}


@dataclass(frozen=True, slots=True)
class ClauseScores:
    bio_temporal: float
    finding_to_diagnosis: float
    located_in_type: float

    @property
    def mean(self) -> float:
        return (self.bio_temporal + self.finding_to_diagnosis + self.located_in_type) / 3.0

    def as_dict(self) -> dict[str, float]:
        return {
            "bio_temporal": self.bio_temporal,
            "finding_to_diagnosis": self.finding_to_diagnosis,
            "located_in_type": self.located_in_type,
            "mean": self.mean,
        }


def compute_clause_scores(
    graph: SimpleDiGraph,
    *,
    type_compatibility: set[tuple[str, str]] | None = None,
    source_reliability: Mapping[str, float] | None = None,
) -> ClauseScores:
    type_compatibility = type_compatibility or DEFAULT_TYPE_COMPATIBILITY
    source_reliability = source_reliability or {"primekg": 1.0, "synthetic": 0.55}
    edges = list(graph.edges(data=True))
    if not edges:
        return ClauseScores(0.0, 0.0, 0.0)

    bio_truths: list[tuple[float, float]] = []
    located_truths: list[tuple[float, float]] = []
    for source, target, attrs in edges:
        source_type = normalize_node_type(graph.nodes[source].get("type", "unknown"))
        target_type = normalize_node_type(graph.nodes[target].get("type", "unknown"))
        confidence = float(attrs.get("confidence", 1.0))
        reliability = source_reliability.get(str(attrs.get("edge_source", attrs.get("source", "primekg"))), 0.75)
        weight = confidence * reliability
        type_ok = (source_type, target_type) in type_compatibility
        temporal_ok = bool(attrs.get("temporal_ordered", True))
        bio_truths.append((1.0 if type_ok and temporal_ok else 0.0, weight))

        relation = str(attrs.get("display_relation") or attrs.get("relation") or "").lower()
        if relation == "located_in":
            located_ok = source_type == "phenotype" and target_type == "anatomy"
            located_truths.append((1.0 if located_ok else 0.0, weight))

    return ClauseScores(
        bio_temporal=_weighted_mean(bio_truths),
        finding_to_diagnosis=_finding_to_diagnosis_score(graph),
        located_in_type=1.0 if not located_truths else _weighted_mean(located_truths),
    )


def _weighted_mean(values: list[tuple[float, float]]) -> float:
    denom = sum(weight for _, weight in values)
    if denom == 0:
        return 0.0
    return sum(value * weight for value, weight in values) / denom


def normalize_node_type(node_type: object) -> str:
    raw = str(node_type).lower().strip().replace("_", " ")
    if "phenotype" in raw or "effect" in raw or "finding" in raw:
        return "phenotype"
    if "disease" in raw or "diagnosis" in raw:
        return "disease"
    if "anatomy" in raw or "anatomical" in raw:
        return "anatomy"
    if "drug" in raw:
        return "drug"
    if "biological process" in raw or "biological_process" in raw:
        return "biological_process"
    return raw


def _finding_to_diagnosis_score(graph: SimpleDiGraph) -> float:
    findings = [n for n, attrs in graph.nodes(data=True) if normalize_node_type(attrs.get("type", "")) == "phenotype"]
    diagnoses = [n for n, attrs in graph.nodes(data=True) if normalize_node_type(attrs.get("type", "")) == "disease"]
    if not findings:
        return 1.0
    if not diagnoses:
        return 0.0

    undirected = graph.to_undirected()
    connected = 0
    for finding in findings:
        if any(undirected.has_path(finding, diagnosis) for diagnosis in diagnoses):
            connected += 1
    return connected / len(findings)
