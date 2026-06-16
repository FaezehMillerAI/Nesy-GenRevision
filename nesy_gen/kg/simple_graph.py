from __future__ import annotations

import heapq
from typing import Callable, Iterable


class NodeAccessor:
    def __init__(self, graph: "SimpleDiGraph") -> None:
        self._graph = graph

    def __iter__(self):
        return iter(self._graph._nodes)

    def __contains__(self, node: str) -> bool:
        return str(node) in self._graph._nodes

    def __getitem__(self, node: str) -> dict:
        return self._graph._nodes[str(node)]

    def __call__(self, data: bool = False):
        if data:
            return list(self._graph._nodes.items())
        return list(self._graph._nodes)


class SimpleDiGraph:
    """Tiny directed graph API covering the repo's reasoning needs."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict] = {}
        self._adj: dict[str, dict[str, dict]] = {}
        self._radj: dict[str, dict[str, dict]] = {}
        self.nodes = NodeAccessor(self)

    def __contains__(self, node: str) -> bool:
        return str(node) in self._nodes

    def add_node(self, node: str, **attrs) -> None:
        node = str(node)
        self._nodes.setdefault(node, {}).update(attrs)
        self._adj.setdefault(node, {})
        self._radj.setdefault(node, {})

    def add_edge(self, source: str, target: str, **attrs) -> None:
        source = str(source)
        target = str(target)
        self.add_node(source)
        self.add_node(target)
        edge_attrs = dict(attrs)
        self._adj[source][target] = edge_attrs
        self._radj[target][source] = edge_attrs

    def edges(self, data: bool = False):
        rows = []
        for source, targets in self._adj.items():
            for target, attrs in targets.items():
                rows.append((source, target, attrs.copy()) if data else (source, target))
        return rows

    def in_edges(self, node: str, data: bool = False):
        node = str(node)
        return [
            (source, node, attrs.copy()) if data else (source, node)
            for source, attrs in self._radj.get(node, {}).items()
        ]

    def out_edges(self, node: str, data: bool = False):
        node = str(node)
        rows = []
        for target, attrs in self._adj.get(node, {}).items():
            rows.append((node, target, attrs.copy()) if data else (node, target))
        return rows

    def subgraph(self, nodes: Iterable[str]) -> "SimpleDiGraph":
        keep = {str(node) for node in nodes}
        graph = SimpleDiGraph()
        for node in keep:
            if node in self._nodes:
                graph.add_node(node, **self._nodes[node])
        for source in keep:
            for target, attrs in self._adj.get(source, {}).items():
                if target not in keep:
                    continue
                graph.add_edge(source, target, **attrs)
        return graph

    def copy(self) -> "SimpleDiGraph":
        return self.subgraph(self._nodes.keys())

    def to_undirected(self) -> "SimpleDiGraph":
        graph = self.copy()
        for source, target, attrs in self.edges(data=True):
            if source not in graph._adj.get(target, {}):
                graph.add_edge(target, source, **attrs)
        return graph

    def has_path(self, source: str, target: str) -> bool:
        try:
            self.shortest_path(source, target)
            return True
        except ValueError:
            return False

    def shortest_path(
        self,
        source: str,
        target: str,
        weight: str | Callable[[dict], float] = "weight",
        *,
        directed: bool = True,
        max_expansions: int | None = None,
    ) -> list[str]:
        source = str(source)
        target = str(target)
        if source not in self._nodes or target not in self._nodes:
            raise ValueError("source or target missing from graph")

        queue: list[tuple[float, str, list[str]]] = [(0.0, source, [source])]
        seen: dict[str, float] = {}
        expansions = 0
        while queue:
            cost, node, path = heapq.heappop(queue)
            if node == target:
                return path
            if node in seen and seen[node] <= cost:
                continue
            seen[node] = cost
            expansions += 1
            if max_expansions is not None and expansions > max_expansions:
                raise ValueError("path search exceeded expansion budget")
            for next_node, attrs in self._neighbors(node, directed=directed):
                edge_cost = float(weight(attrs) if callable(weight) else attrs.get(weight, 1.0))
                heapq.heappush(queue, (cost + edge_cost, next_node, path + [next_node]))
        raise ValueError("no path")

    def _neighbors(self, node: str, *, directed: bool) -> Iterable[tuple[str, dict]]:
        yield from self._adj.get(node, {}).items()
        if not directed:
            for prev_node, attrs in self._radj.get(node, {}).items():
                if prev_node not in self._adj.get(node, {}):
                    yield prev_node, attrs
