from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from nesy_gen.models.nesy_gen import NesyGenPipeline
from nesy_gen.models.gate import CandidateToken


@dataclass(frozen=True, slots=True)
class VerifiedCandidate:
    candidate_rank: int
    prediction: str
    num_links: int
    graph_score: float
    bio_temporal: float
    finding_to_diagnosis: float
    located_in_type: float
    gate_acceptance_rate: float = 0.0
    rejected_entities: int = 0
    gate_passed: bool = False


def verify_report_candidates(
    pipeline: NesyGenPipeline,
    *,
    indication: str,
    candidates: Iterable[str],
    evidence_scores: Iterable[float] | None = None,
) -> list[VerifiedCandidate]:
    """Score report candidates with the neuro-symbolic PrimeKG verifier."""

    verified: list[VerifiedCandidate] = []
    seen: set[str] = set()
    evidence_values = list(evidence_scores or [])
    for rank, candidate in enumerate(candidates, start=1):
        prediction = " ".join(str(candidate).split())
        if not prediction or prediction in seen:
            continue
        seen.add(prediction)
        links, audit = pipeline.reason(indication=indication, draft_report=prediction)
        scores = audit.scores.as_dict()
        evidence_index = rank - 1
        evidence_score = evidence_values[evidence_index] if evidence_index < len(evidence_values) else 1.0
        decisions = [
            pipeline.gate.decide(
                CandidateToken(
                    text=link.mention.text,
                    node_id=link.node_id,
                    evidence_score=float(evidence_score),
                    hallucination_score=max(0.0, 1.0 - float(scores["mean"])),
                    entailment_score=1.0,
                ),
                audit,
            )
            for link in links
        ]
        accepted = sum(1 for decision in decisions if decision.accepted)
        gate_acceptance_rate = accepted / len(decisions) if decisions else 0.0
        verified.append(
            VerifiedCandidate(
                candidate_rank=rank,
                prediction=prediction,
                num_links=len(links),
                graph_score=float(scores["mean"]),
                bio_temporal=float(scores["bio_temporal"]),
                finding_to_diagnosis=float(scores["finding_to_diagnosis"]),
                located_in_type=float(scores["located_in_type"]),
                gate_acceptance_rate=gate_acceptance_rate,
                rejected_entities=len(decisions) - accepted,
                gate_passed=bool(decisions) and gate_acceptance_rate >= 0.5,
            )
        )
    return verified


def select_graph_verified_candidate(
    candidates: list[VerifiedCandidate],
    *,
    min_graph_score: float | None = None,
) -> VerifiedCandidate | None:
    """Select the best candidate before final output using PrimeKG consistency."""

    eligible = [
        candidate
        for candidate in candidates
        if min_graph_score is None or candidate.graph_score >= min_graph_score
    ]
    if not eligible:
        eligible = candidates
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda candidate: (
            candidate.gate_passed,
            candidate.graph_score,
            candidate.gate_acceptance_rate,
            candidate.finding_to_diagnosis,
            candidate.located_in_type,
            candidate.num_links,
            -candidate.candidate_rank,
        ),
    )
