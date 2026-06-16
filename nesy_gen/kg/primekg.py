from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from nesy_gen.kg.simple_graph import SimpleDiGraph


NODE_COLUMNS = {
    "x_id": "source_id",
    "x_name": "source_name",
    "x_type": "source_type",
    "y_id": "target_id",
    "y_name": "target_name",
    "y_type": "target_type",
    "relation": "relation",
    "display_relation": "display_relation",
}

PRIMEKG_CANDIDATES = (
    "kg.csv",
    "kg_giant.csv",
    "kg_raw.csv",
    "kg_grouped.csv",
)


@dataclass(slots=True)
class PrimeKGGraph:
    """Light wrapper around PrimeKG's edge-list CSV.

    The public PrimeKG CSV uses x/y endpoint columns. This wrapper normalizes
    them to source/target terminology and keeps enough metadata for audit trails.
    """

    edges: pd.DataFrame
    graph: SimpleDiGraph

    @classmethod
    def from_csv(cls, path: str | Path, *, low_memory: bool = False) -> "PrimeKGGraph":
        frame = pd.read_csv(path, low_memory=low_memory)
        return cls.from_dataframe(frame)

    @classmethod
    def from_dataverse_dir(
        cls,
        dataverse_dir: str | Path = "dataverse_files",
        *,
        low_memory: bool = False,
    ) -> "PrimeKGGraph":
        """Load PrimeKG from a Dataverse export folder.

        Preferred input is the complete `kg.csv` edge list. If that is absent,
        the loader reconstructs an edge list from `edges.csv` and `nodes.csv`.
        """

        csv_path = find_primekg_csv(dataverse_dir)
        if csv_path is not None:
            return cls.from_csv(csv_path, low_memory=low_memory)

        dataverse_path = Path(dataverse_dir)
        edges_path = dataverse_path / "edges.csv"
        nodes_path = dataverse_path / "nodes.csv"
        if not edges_path.exists() or not nodes_path.exists():
            raise FileNotFoundError(
                f"No PrimeKG edge list found in {dataverse_path}. Expected one of "
                f"{PRIMEKG_CANDIDATES} or both edges.csv and nodes.csv."
            )
        edges = pd.read_csv(edges_path, low_memory=low_memory)
        nodes = pd.read_csv(nodes_path, low_memory=low_memory)
        frame = edge_list_from_nodes_edges(edges, nodes)
        return cls.from_dataframe(frame)

    @classmethod
    def from_dataframe(cls, frame: pd.DataFrame) -> "PrimeKGGraph":
        renamed = frame.rename(columns={k: v for k, v in NODE_COLUMNS.items() if k in frame})
        required = {"source_id", "source_name", "source_type", "target_id", "target_name", "target_type"}
        missing = sorted(required - set(renamed.columns))
        if missing:
            raise ValueError(f"PrimeKG edge list is missing required columns: {missing}")

        if "relation" not in renamed.columns and "display_relation" in renamed.columns:
            renamed["relation"] = renamed["display_relation"]
        if "display_relation" not in renamed.columns:
            renamed["display_relation"] = renamed["relation"]

        graph = SimpleDiGraph()
        for row in renamed.itertuples(index=False):
            source_id = str(getattr(row, "source_id"))
            target_id = str(getattr(row, "target_id"))
            graph.add_node(
                source_id,
                name=str(getattr(row, "source_name")),
                type=str(getattr(row, "source_type")),
            )
            graph.add_node(
                target_id,
                name=str(getattr(row, "target_name")),
                type=str(getattr(row, "target_type")),
            )
            graph.add_edge(
                source_id,
                target_id,
                relation=str(getattr(row, "relation")),
                display_relation=str(getattr(row, "display_relation")),
                confidence=float(getattr(row, "confidence", 1.0)),
                edge_source=str(getattr(row, "source", "primekg")),
            )
        return cls(edges=renamed, graph=graph)

    def node_type(self, node_id: str) -> str:
        return str(self.graph.nodes[str(node_id)].get("type", "unknown"))

    def node_name(self, node_id: str) -> str:
        return str(self.graph.nodes[str(node_id)].get("name", node_id))

    def has_nodes(self, node_ids: Iterable[str]) -> bool:
        return all(str(node_id) in self.graph for node_id in node_ids)

    def coverage(self, node_ids: Iterable[str]) -> dict[str, object]:
        requested = [str(node_id) for node_id in node_ids]
        missing = [node_id for node_id in requested if node_id not in self.graph]
        return {
            "requested": len(requested),
            "covered": len(requested) - len(missing),
            "missing": missing,
            "coverage": 0.0 if not requested else (len(requested) - len(missing)) / len(requested),
        }


def find_primekg_csv(dataverse_dir: str | Path = "dataverse_files") -> Path | None:
    """Return the best complete PrimeKG CSV in a Dataverse folder."""

    root = Path(dataverse_dir)
    for name in PRIMEKG_CANDIDATES:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def edge_list_from_nodes_edges(edges: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the complete edge-list schema from separate nodes/edges files."""

    required_edge_cols = {"x_index", "y_index", "relation", "display_relation"}
    required_node_cols = {"node_index", "node_id", "node_type", "node_name"}
    missing_edges = sorted(required_edge_cols - set(edges.columns))
    missing_nodes = sorted(required_node_cols - set(nodes.columns))
    if missing_edges or missing_nodes:
        raise ValueError(
            f"Cannot reconstruct PrimeKG edge list. Missing edge columns={missing_edges}; "
            f"missing node columns={missing_nodes}."
        )

    node_table = nodes.rename(
        columns={
            "node_index": "x_index",
            "node_id": "x_id",
            "node_type": "x_type",
            "node_name": "x_name",
            "node_source": "x_source",
        }
    )
    merged = edges.merge(node_table, on="x_index", how="left")
    node_table = nodes.rename(
        columns={
            "node_index": "y_index",
            "node_id": "y_id",
            "node_type": "y_type",
            "node_name": "y_name",
            "node_source": "y_source",
        }
    )
    merged = merged.merge(node_table, on="y_index", how="left")
    return merged
