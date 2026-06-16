from __future__ import annotations

from dataclasses import dataclass

from nesy_gen.logic.ltn import AuditReport


@dataclass(frozen=True, slots=True)
class CandidateToken:
    text: str
    node_id: str | None
    evidence_score: float
    hallucination_score: float
    entailment_score: float = 1.0


@dataclass(frozen=True, slots=True)
class ConsistencyDecision:
    token: CandidateToken
    accepted: bool
    reason: str
    confidence: float


@dataclass(slots=True)
class ConsistencyGate:
    min_grounding: float = 0.30
    max_hallucination: float = 0.50
    min_entailment: float = 0.50

    def decide(self, token: CandidateToken, audit: AuditReport) -> ConsistencyDecision:
        if token.node_id is None:
            return ConsistencyDecision(token, False, "unlinked_entity", 0.0)
        if token.node_id not in audit.valid_nodes:
            if token.node_id in audit.flagged_nodes:
                return ConsistencyDecision(token, False, "flagged_graph_satisfaction", audit.mean_satisfaction)
            return ConsistencyDecision(token, False, "graph_unreachable_or_rejected", 0.0)
        if token.evidence_score < self.min_grounding:
            return ConsistencyDecision(token, False, "low_visual_grounding", token.evidence_score)
        if token.entailment_score < self.min_entailment:
            return ConsistencyDecision(token, False, "nli_contradiction_or_neutral", token.entailment_score)
        if token.hallucination_score > self.max_hallucination:
            return ConsistencyDecision(token, False, "high_hallucination_score", 1 - token.hallucination_score)
        confidence = min(token.evidence_score, token.entailment_score, audit.mean_satisfaction)
        return ConsistencyDecision(token, True, "accepted", confidence)

