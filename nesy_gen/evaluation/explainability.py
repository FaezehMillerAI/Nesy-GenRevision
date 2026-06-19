from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_adaptive_traces(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def claim_trace_frame(traces: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for trace in traces:
        study_id = str(trace.get("study_id", ""))
        for claim in trace.get("claims", []):
            entities = claim.get("linked_entities", [])
            rows.append(
                {
                    "study_id": study_id,
                    **claim,
                    "num_entities": len(entities),
                    "entity_names": "; ".join(
                        str(entity.get("node_name", "")) for entity in entities
                    ),
                    "graph_path_length": len(claim.get("primekg_path", [])),
                    "has_grounding": max(
                        float(claim.get("visual_support", 0.0)),
                        float(claim.get("retrieval_support", 0.0)),
                    )
                    > 0.0,
                }
            )
    return pd.DataFrame(rows)


def explainability_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {"num_claims": 0, "num_studies": 0}
    linked = frame["num_entities"] > 0
    escalated = frame["verification_triggered"].astype(bool)
    return {
        "num_claims": int(len(frame)),
        "num_studies": int(frame["study_id"].nunique()),
        "linked_claim_rate": float(linked.mean()),
        "grounded_claim_rate": float(frame["has_grounding"].mean()),
        "adaptive_escalation_rate": float(escalated.mean()),
        "graph_path_coverage_when_escalated": float(
            (frame.loc[escalated, "graph_path_length"] > 0).mean() if escalated.any() else 0.0
        ),
        "revision_rate": float((frame["decision"] == "revise").mean()),
        "flag_rate": float(frame["decision"].isin(["flag", "abstain"]).mean()),
        "mean_claim_latency_ms": float(frame["latency_ms"].mean()),
        "decision_counts": {
            str(key): int(value) for key, value in frame["decision"].value_counts().items()
        },
    }
