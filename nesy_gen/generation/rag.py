from __future__ import annotations

from dataclasses import asdict, dataclass

from nesy_gen.baselines.retrieval import run_tfidf_retrieval_topk
from nesy_gen.data.schema import RadiologyExample
from nesy_gen.evaluation.graph_verification import (
    VerifiedCandidate,
    select_graph_verified_candidate,
    verify_report_candidates,
)
from nesy_gen.models.nesy_gen import NesyGenPipeline


@dataclass(frozen=True, slots=True)
class RagCandidate:
    source: str
    source_rank: int
    prediction: str
    evidence_score: float
    retrieved_study_id: str = ""


def retrieval_candidates(
    train_examples: list[RadiologyExample],
    query_examples: list[RadiologyExample],
    *,
    top_k: int = 5,
) -> dict[str, list[RagCandidate]]:
    retrieved = run_tfidf_retrieval_topk(train_examples, query_examples, top_k=top_k)
    rows: dict[str, list[RagCandidate]] = {}
    for example, predictions in zip(query_examples, retrieved, strict=True):
        rows[example.study_id] = [
            RagCandidate(
                source="retrieval",
                source_rank=prediction.rank,
                prediction=prediction.prediction,
                evidence_score=prediction.similarity,
                retrieved_study_id=prediction.retrieved_study_id,
            )
            for prediction in predictions
        ]
    return rows


def select_agentic_draft(
    example: RadiologyExample,
    candidates: list[RagCandidate],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Choose one draft before adaptive claim verification.

    The image-conditioned generator is preferred when present. Retrieval is
    retained as evidence and as a leakage-safe fallback, not as a test oracle.
    """

    generated = [candidate for candidate in candidates if candidate.source == "vision_t5"]
    pool = generated or candidates
    selected = max(
        pool,
        key=lambda candidate: (
            candidate.evidence_score,
            -candidate.source_rank,
            len(candidate.prediction),
        ),
        default=RagCandidate("none", 0, "", 0.0),
    )
    candidate_rows = [
        {
            "study_id": example.study_id,
            "reference": example.report,
            "source": candidate.source,
            "source_rank": candidate.source_rank,
            "retrieved_study_id": candidate.retrieved_study_id,
            "prediction": candidate.prediction,
            "evidence_score": candidate.evidence_score,
            "candidate_rank": rank,
            "selected_as_draft": candidate is selected,
        }
        for rank, candidate in enumerate(candidates, start=1)
    ]
    return (
        {
            "study_id": example.study_id,
            "prediction": selected.prediction,
            "reference": example.report,
            "selection_status": "agentic_draft_unverified",
            "source": selected.source,
            "source_rank": selected.source_rank,
            "retrieved_study_id": selected.retrieved_study_id,
            "evidence_score": selected.evidence_score,
        },
        candidate_rows,
    )


def select_primekg_verified_report(
    pipeline: NesyGenPipeline,
    example: RadiologyExample,
    candidates: list[RagCandidate],
    *,
    min_graph_score: float | None = None,
    selection_objective: str = "graph",
    graph_score_weight: float = 0.55,
    evidence_weight: float = 0.35,
    gate_weight: float = 0.10,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    verified = verify_report_candidates(
        pipeline,
        indication=example.indication,
        candidates=[candidate.prediction for candidate in candidates],
        evidence_scores=[candidate.evidence_score for candidate in candidates],
    )
    source_by_prediction = {_normalize(candidate.prediction): candidate for candidate in candidates}
    candidate_rows = []
    for candidate in verified:
        source = source_by_prediction.get(_normalize(candidate.prediction))
        candidate_rows.append(
            {
                "study_id": example.study_id,
                "reference": example.report,
                "source": source.source if source else "",
                "source_rank": source.source_rank if source else candidate.candidate_rank,
                "retrieved_study_id": source.retrieved_study_id if source else "",
                **asdict(candidate),
            }
        )
    selected = _select_candidate(
        verified,
        candidates,
        min_graph_score=min_graph_score,
        selection_objective=selection_objective,
        graph_score_weight=graph_score_weight,
        evidence_weight=evidence_weight,
        gate_weight=gate_weight,
    )
    if selected is None:
        fallback = candidates[0] if candidates else RagCandidate("none", 0, "", 0.0)
        selected_row = {
            "study_id": example.study_id,
            "prediction": fallback.prediction,
            "reference": example.report,
            "selection_status": "fallback_unverified",
            "source": fallback.source,
            "source_rank": fallback.source_rank,
            "retrieved_study_id": fallback.retrieved_study_id,
            "graph_score": 0.0,
            "gate_acceptance_rate": 0.0,
            "rejected_entities": 0,
            "gate_passed": False,
        }
        return selected_row, candidate_rows

    selected_source = source_by_prediction.get(_normalize(selected.prediction))
    selected_row = {
        "study_id": example.study_id,
        "prediction": selected.prediction,
        "reference": example.report,
        "selection_status": "primekg_ltn_gate_selected",
        "source": selected_source.source if selected_source else "",
        "source_rank": selected_source.source_rank if selected_source else selected.candidate_rank,
        "retrieved_study_id": selected_source.retrieved_study_id if selected_source else "",
        "selected_candidate_rank": selected.candidate_rank,
        "num_links": selected.num_links,
        "graph_score": selected.graph_score,
        "bio_temporal": selected.bio_temporal,
        "finding_to_diagnosis": selected.finding_to_diagnosis,
        "located_in_type": selected.located_in_type,
        "gate_acceptance_rate": selected.gate_acceptance_rate,
        "rejected_entities": selected.rejected_entities,
        "gate_passed": selected.gate_passed,
    }
    return selected_row, candidate_rows


def _normalize(text: str) -> str:
    return " ".join(str(text).split())


def _select_candidate(
    verified: list[VerifiedCandidate],
    candidates: list[RagCandidate],
    *,
    min_graph_score: float | None,
    selection_objective: str,
    graph_score_weight: float,
    evidence_weight: float,
    gate_weight: float,
) -> VerifiedCandidate | None:
    if selection_objective == "graph":
        return select_graph_verified_candidate(verified, min_graph_score=min_graph_score)

    evidence_by_prediction = {
        _normalize(candidate.prediction): candidate.evidence_score for candidate in candidates
    }
    eligible = [
        candidate
        for candidate in verified
        if min_graph_score is None or candidate.graph_score >= min_graph_score
    ]
    if not eligible:
        eligible = verified
    if not eligible:
        return None

    if selection_objective == "evidence":
        return max(
            eligible,
            key=lambda candidate: (
                evidence_by_prediction.get(_normalize(candidate.prediction), 0.0),
                candidate.gate_passed,
                candidate.graph_score,
                -candidate.candidate_rank,
            ),
        )
    if selection_objective != "hybrid":
        raise ValueError("selection_objective must be one of: graph, evidence, hybrid")

    return max(
        eligible,
        key=lambda candidate: (
            graph_score_weight * candidate.graph_score
            + evidence_weight * evidence_by_prediction.get(_normalize(candidate.prediction), 0.0)
            + gate_weight * candidate.gate_acceptance_rate
            + (0.05 if candidate.gate_passed else 0.0),
            candidate.graph_score,
            evidence_by_prediction.get(_normalize(candidate.prediction), 0.0),
            -candidate.candidate_rank,
        ),
    )
