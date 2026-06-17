from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from nesy_gen.models.nesy_gen import NesyGenPipeline


@dataclass(frozen=True, slots=True)
class VerifiedCandidate:
    candidate_rank: int
    prediction: str
    num_links: int
    graph_score: float
    bio_temporal: float
    finding_to_diagnosis: float
    located_in_type: float


def verify_report_candidates(
    pipeline: NesyGenPipeline,
    *,
    indication: str,
    candidates: Iterable[str],
) -> list[VerifiedCandidate]:
    """Score report candidates with the neuro-symbolic PrimeKG verifier."""

    verified: list[VerifiedCandidate] = []
    seen: set[str] = set()
    for rank, candidate in enumerate(candidates):
        prediction = " ".join(str(candidate).split())
        if not prediction or prediction in seen:
            continue
        seen.add(prediction)
        links, audit = pipeline.reason(indication=indication, draft_report=prediction)
        scores = audit.scores.as_dict()
        verified.append(
            VerifiedCandidate(
                candidate_rank=rank,
                prediction=prediction,
                num_links=len(links),
                graph_score=float(scores["mean"]),
                bio_temporal=float(scores["bio_temporal"]),
                finding_to_diagnosis=float(scores["finding_to_diagnosis"]),
                located_in_type=float(scores["located_in_type"]),
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
            candidate.graph_score,
            candidate.finding_to_diagnosis,
            candidate.located_in_type,
            candidate.num_links,
            -candidate.candidate_rank,
        ),
    )
