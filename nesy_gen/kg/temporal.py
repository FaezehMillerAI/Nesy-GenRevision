from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from nesy_gen.kg.entity_linking import LinkedEntity
from nesy_gen.kg.primekg import PrimeKGGraph
from nesy_gen.kg.simple_graph import SimpleDiGraph


DEFAULT_RELATION_WEIGHTS = {
    "causes": 1.0,
    "located_in": 0.7,
    "associated_with": 0.5,
    "default": 0.3,
}


@dataclass(slots=True)
class TemporalSubgraphBuilder:
    primekg: PrimeKGGraph
    temporal_alpha: float = 0.6
    relation_weights: Mapping[str, float] | None = None
    max_path_expansions: int | None = 200_000
    strategy: str = "steiner"
    max_neighbors_per_node: int = 250

    def __post_init__(self) -> None:
        if self.relation_weights is None:
            self.relation_weights = DEFAULT_RELATION_WEIGHTS

    def build(self, linked_entities: Sequence[LinkedEntity]) -> SimpleDiGraph:
        terminals = [link.node_id for link in linked_entities if not link.mention.negated]
        terminals = [node_id for node_id in dict.fromkeys(terminals) if node_id in self.primekg.graph]
        if self.strategy == "ego":
            return self._build_ego(terminals)
        if len(terminals) <= 1:
            return self.primekg.graph.subgraph(terminals).copy()
        if self.strategy != "steiner":
            raise ValueError("strategy must be one of: steiner, ego")

        sub_nodes: set[str] = set(terminals)
        root = terminals[0]
        for terminal in terminals[1:]:
            try:
                path = self.primekg.graph.shortest_path(
                    root,
                    terminal,
                    weight=self.edge_weight,
                    directed=True,
                    max_expansions=self.max_path_expansions,
                )
            except ValueError:
                try:
                    path = self.primekg.graph.shortest_path(
                        root,
                        terminal,
                        weight=self.edge_weight,
                        directed=False,
                        max_expansions=self.max_path_expansions,
                    )
                except ValueError:
                    continue
            sub_nodes.update(str(node) for node in path)

        return self.primekg.graph.subgraph(sub_nodes)

    def _build_ego(self, terminals: Sequence[str]) -> SimpleDiGraph:
        sub_nodes: set[str] = set(terminals)
        for node_id in terminals:
            incident = []
            incident.extend(self.primekg.graph.out_edges(node_id, data=True))
            incident.extend(self.primekg.graph.in_edges(node_id, data=True))
            incident = sorted(incident, key=lambda edge: self.edge_weight(edge[2]))
            for source, target, _attrs in incident[: self.max_neighbors_per_node]:
                sub_nodes.add(str(source))
                sub_nodes.add(str(target))
        return self.primekg.graph.subgraph(sub_nodes)

    def edge_weight(self, attrs: Mapping[str, object]) -> float:
        relation = str(attrs.get("display_relation") or attrs.get("relation") or "default").lower()
        relation_score = self.relation_weights.get(relation, self.relation_weights.get("default", 0.3))
        temporal_distance = float(attrs.get("temporal_distance", 0.0))
        return self.temporal_alpha * temporal_distance + (1.0 - self.temporal_alpha) * (1.0 - relation_score)
