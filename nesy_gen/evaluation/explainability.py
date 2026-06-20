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
                    "has_visual_evidence": float(claim.get("visual_support", 0.0)) > 0.0,
                    "has_entity_grounding": (
                        float(claim.get("retrieval_support", 0.0)) > 0.0
                        and int(claim.get("retrieval_support_count", 0)) > 0
                    ),
                    "report_end_to_end_latency_ms": float(
                        trace.get("end_to_end_latency_ms", 0.0)
                    ),
                    "trace_complete": _trace_complete(claim),
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
        "report_visual_evidence_rate": float(frame["has_visual_evidence"].mean()),
        "entity_grounding_rate": float(frame["has_entity_grounding"].mean()),
        "linked_entity_grounding_rate": float(
            frame.loc[linked, "has_entity_grounding"].mean() if linked.any() else 0.0
        ),
        "adaptive_escalation_rate": float(escalated.mean()),
        "adaptive_escalation_rate_linked": float(
            escalated.loc[linked].mean() if linked.any() else 0.0
        ),
        "graph_path_coverage_when_escalated": float(
            (frame.loc[escalated, "graph_path_length"] > 0).mean() if escalated.any() else 0.0
        ),
        "revision_rate": float((frame["decision"] == "revise").mean()),
        "flag_rate": float(frame["decision"].isin(["flag", "abstain"]).mean()),
        "mean_claim_latency_ms": float(frame["latency_ms"].mean()),
        "mean_end_to_end_latency_ms": float(frame["report_end_to_end_latency_ms"].mean()),
        "explanation_completeness": float(frame["trace_complete"].mean()),
        "decision_counts": {
            str(key): int(value) for key, value in frame["decision"].value_counts().items()
        },
    }


def _trace_complete(claim: dict[str, object]) -> bool:
    required = {
        "original_claim",
        "final_claim",
        "linked_entities",
        "visual_support",
        "retrieval_support",
        "retrieval_support_count",
        "verification_triggered",
        "decision",
        "reason",
        "latency_ms",
    }
    if not required.issubset(claim):
        return False
    if bool(claim.get("verification_triggered")):
        escalated = {"primekg_status", "primekg_score", "ltn_truth", "ltn_clause_scores"}
        return escalated.issubset(claim)
    return True
