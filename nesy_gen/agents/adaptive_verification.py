from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import time
from typing import Iterable

from nesy_gen.generation.rag import RagCandidate
from nesy_gen.kg.entity_linking import LinkedEntity
from nesy_gen.logic.constraints import ClauseScores
from nesy_gen.logic.ltn import AuditReport
from nesy_gen.models.gate import CandidateToken
from nesy_gen.models.nesy_gen import NesyGenPipeline


CLAIM_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True, slots=True)
class ClaimEvidenceTrace:
    claim_id: int
    original_claim: str
    final_claim: str
    linked_entities: list[dict[str, object]]
    visual_support: float
    retrieval_support: float
    retrieval_support_count: int
    ltn_truth: float | None
    ltn_clause_scores: dict[str, float]
    primekg_status: str
    primekg_path: list[dict[str, object]]
    gate_confidence: float
    decision: str
    reason: str
    verification_triggered: bool
    replacement_source_study_id: str
    latency_ms: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AdaptiveVerificationResult:
    original_report: str
    final_report: str
    claims: list[ClaimEvidenceTrace]
    accepted_claims: int
    revised_claims: int
    flagged_claims: int
    graph_calls: int
    total_claims: int
    escalation_rate: float
    latency_ms: float

    def as_dict(self) -> dict[str, object]:
        return {
            "original_report": self.original_report,
            "final_report": self.final_report,
            "accepted_claims": self.accepted_claims,
            "revised_claims": self.revised_claims,
            "flagged_claims": self.flagged_claims,
            "graph_calls": self.graph_calls,
            "total_claims": self.total_claims,
            "escalation_rate": self.escalation_rate,
            "latency_ms": self.latency_ms,
            "claims": [claim.as_dict() for claim in self.claims],
        }


@dataclass(frozen=True, slots=True)
class _EvidenceSentence:
    text: str
    study_id: str
    score: float
    links: tuple[LinkedEntity, ...]


class AdaptiveClaimVerifier:
    """Evidence-first claim router with selective PrimeKG/LTN escalation.

    High-consensus claims take a transparent fast path. Ambiguous claims are
    verified with the actual graph and gate used by the system. Revision is
    extractive and evidence-bound: no post-hoc LLM rationale is introduced.
    """

    def __init__(
        self,
        pipeline: NesyGenPipeline,
        *,
        fast_accept_threshold: float = 0.85,
        min_supporting_reports: int = 2,
        revise_threshold: float = 0.50,
        revision_policy: str = "evidence_replace",
        use_ltn: bool = True,
        use_gate: bool = True,
    ) -> None:
        if revision_policy not in {"audit_only", "evidence_replace"}:
            raise ValueError("revision_policy must be audit_only or evidence_replace")
        self.pipeline = pipeline
        self.fast_accept_threshold = fast_accept_threshold
        self.min_supporting_reports = min_supporting_reports
        self.revise_threshold = revise_threshold
        self.revision_policy = revision_policy
        self.use_ltn = use_ltn
        self.use_gate = use_gate

    def verify(
        self,
        report: str,
        *,
        indication: str = "",
        visual_support: float = 0.0,
        evidence_candidates: Iterable[RagCandidate] = (),
    ) -> AdaptiveVerificationResult:
        started = time.perf_counter()
        claims = split_clinical_claims(report)
        evidence = self._prepare_evidence(evidence_candidates)
        indication_links = self.pipeline.linker.link_text(indication)
        traces: list[ClaimEvidenceTrace] = []

        for claim_id, claim in enumerate(claims, start=1):
            traces.append(
                self._verify_claim(
                    claim_id,
                    claim,
                    indication_links=indication_links,
                    visual_support=visual_support,
                    evidence=evidence,
                )
            )

        final_claims = [trace.final_claim for trace in traces if trace.final_claim.strip()]
        final_report = " ".join(final_claims).strip() or report.strip()
        accepted = sum(trace.decision.startswith("accept") for trace in traces)
        revised = sum(trace.decision == "revise" for trace in traces)
        flagged = sum(trace.decision in {"flag", "abstain"} for trace in traces)
        graph_calls = sum(trace.verification_triggered for trace in traces)
        total = len(traces)
        return AdaptiveVerificationResult(
            original_report=report,
            final_report=final_report,
            claims=traces,
            accepted_claims=accepted,
            revised_claims=revised,
            flagged_claims=flagged,
            graph_calls=graph_calls,
            total_claims=total,
            escalation_rate=graph_calls / total if total else 0.0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _verify_claim(
        self,
        claim_id: int,
        claim: str,
        *,
        indication_links: list[LinkedEntity],
        visual_support: float,
        evidence: list[_EvidenceSentence],
    ) -> ClaimEvidenceTrace:
        started = time.perf_counter()
        links = self.pipeline.linker.link_text(claim)
        retrieval_score, support_count = _retrieval_support(links, evidence)
        grounding = max(_unit_score(visual_support), retrieval_score)
        entities = [_link_record(link) for link in links]

        if not links:
            return self._trace(
                claim_id,
                claim,
                claim,
                entities,
                visual_support,
                retrieval_score,
                support_count,
                decision="abstain",
                reason="no_linked_clinical_entity",
                started=started,
            )

        if grounding >= self.fast_accept_threshold and support_count >= self.min_supporting_reports:
            return self._trace(
                claim_id,
                claim,
                claim,
                entities,
                visual_support,
                retrieval_score,
                support_count,
                gate_confidence=grounding,
                decision="accept_fast_path",
                reason="high_visual_retrieval_consensus",
                started=started,
            )

        graph_links = _deduplicate_links([*indication_links, *links])
        graph = self.pipeline.subgraph_builder.build(graph_links)
        audit = self.pipeline.auditor.audit(graph) if self.use_ltn else _connectivity_audit(graph)
        scores = {key: float(value) for key, value in audit.scores.as_dict().items()}
        decisions = (
            [
                self.pipeline.gate.decide(
                    CandidateToken(
                        text=link.mention.text,
                        node_id=link.node_id,
                        evidence_score=grounding,
                        hallucination_score=max(0.0, 1.0 - audit.mean_satisfaction),
                        entailment_score=1.0,
                    ),
                    audit,
                )
                for link in links
            ]
            if self.use_gate
            else []
        )
        accepted_rate = (
            sum(decision.accepted for decision in decisions) / len(decisions)
            if self.use_gate
            else sum(link.node_id in audit.valid_nodes for link in links) / len(links)
        )
        confidence = min(grounding, audit.mean_satisfaction) if decisions else 0.0
        status = _aggregate_graph_status(links, audit)
        path = _explanation_path(graph, links, graph_links)

        if accepted_rate == 1.0 and confidence >= self.revise_threshold:
            return self._trace(
                claim_id,
                claim,
                claim,
                entities,
                visual_support,
                retrieval_score,
                support_count,
                ltn_truth=audit.mean_satisfaction,
                clause_scores=scores,
                primekg_status=status,
                primekg_path=path,
                gate_confidence=confidence,
                decision="accept_verified",
                reason="primekg_ltn_gate_agreement",
                verification_triggered=True,
                started=started,
            )

        replacement = self._replacement(claim, links, evidence)
        if replacement is not None and self.revision_policy == "evidence_replace":
            return self._trace(
                claim_id,
                claim,
                replacement.text,
                entities,
                visual_support,
                retrieval_score,
                support_count,
                ltn_truth=audit.mean_satisfaction,
                clause_scores=scores,
                primekg_status=status,
                primekg_path=path,
                gate_confidence=confidence,
                decision="revise",
                reason="replaced_by_higher_support_matching_assertion",
                verification_triggered=True,
                replacement_source_study_id=replacement.study_id,
                started=started,
            )

        reasons = sorted({decision.reason for decision in decisions if not decision.accepted})
        if not self.use_gate:
            reasons = ["consistency_gate_disabled_ablation"]
        return self._trace(
            claim_id,
            claim,
            claim,
            entities,
            visual_support,
            retrieval_score,
            support_count,
            ltn_truth=audit.mean_satisfaction,
            clause_scores=scores,
            primekg_status=status,
            primekg_path=path,
            gate_confidence=confidence,
            decision="flag",
            reason=";".join(reasons) or "uncertain_claim_preserved_for_review",
            verification_triggered=True,
            started=started,
        )

    def _prepare_evidence(self, candidates: Iterable[RagCandidate]) -> list[_EvidenceSentence]:
        evidence: list[_EvidenceSentence] = []
        for candidate in candidates:
            if candidate.source not in {"retrieval", "visual_retrieval", "medsiglip_retrieval"}:
                continue
            for sentence in split_clinical_claims(candidate.prediction):
                links = tuple(self.pipeline.linker.link_text(sentence))
                if links:
                    evidence.append(
                        _EvidenceSentence(
                            sentence,
                            candidate.retrieved_study_id,
                            _unit_score(candidate.evidence_score),
                            links,
                        )
                    )
        return evidence

    def _replacement(
        self,
        claim: str,
        links: list[LinkedEntity],
        evidence: list[_EvidenceSentence],
    ) -> _EvidenceSentence | None:
        target = {(link.node_id, link.mention.negated) for link in links}
        candidates = [
            sentence
            for sentence in evidence
            if sentence.text.strip().lower() != claim.strip().lower()
            and target == {(link.node_id, link.mention.negated) for link in sentence.links}
            and sentence.score >= self.revise_threshold
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda sentence: (
                len(target & {(link.node_id, link.mention.negated) for link in sentence.links}),
                sentence.score,
                -len(sentence.text),
            ),
        )

    @staticmethod
    def _trace(
        claim_id: int,
        original_claim: str,
        final_claim: str,
        linked_entities: list[dict[str, object]],
        visual_support: float,
        retrieval_support: float,
        retrieval_support_count: int,
        *,
        ltn_truth: float | None = None,
        clause_scores: dict[str, float] | None = None,
        primekg_status: str = "not_invoked",
        primekg_path: list[dict[str, object]] | None = None,
        gate_confidence: float = 0.0,
        decision: str,
        reason: str,
        verification_triggered: bool = False,
        replacement_source_study_id: str = "",
        started: float,
    ) -> ClaimEvidenceTrace:
        return ClaimEvidenceTrace(
            claim_id=claim_id,
            original_claim=original_claim,
            final_claim=final_claim,
            linked_entities=linked_entities,
            visual_support=_unit_score(visual_support),
            retrieval_support=retrieval_support,
            retrieval_support_count=retrieval_support_count,
            ltn_truth=ltn_truth,
            ltn_clause_scores=clause_scores or {},
            primekg_status=primekg_status,
            primekg_path=primekg_path or [],
            gate_confidence=gate_confidence,
            decision=decision,
            reason=reason,
            verification_triggered=verification_triggered,
            replacement_source_study_id=replacement_source_study_id,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


def split_clinical_claims(report: str) -> list[str]:
    return [part.strip() for part in CLAIM_BOUNDARY_RE.split(str(report)) if part.strip()]


def _retrieval_support(
    links: list[LinkedEntity], evidence: list[_EvidenceSentence]
) -> tuple[float, int]:
    if not links:
        return 0.0, 0
    target = {(link.node_id, link.mention.negated) for link in links}
    matches = [
        sentence
        for sentence in evidence
        if target & {(link.node_id, link.mention.negated) for link in sentence.links}
    ]
    studies = {sentence.study_id for sentence in matches if sentence.study_id}
    return (max((sentence.score for sentence in matches), default=0.0), len(studies))


def _deduplicate_links(links: list[LinkedEntity]) -> list[LinkedEntity]:
    rows: list[LinkedEntity] = []
    seen: set[tuple[str, bool]] = set()
    for link in links:
        key = (link.node_id, link.mention.negated)
        if key not in seen:
            seen.add(key)
            rows.append(link)
    return rows


def _aggregate_graph_status(links, audit) -> str:
    node_ids = {link.node_id for link in links}
    if node_ids & audit.rejected_nodes:
        return "rejected"
    if node_ids & audit.flagged_nodes:
        return "flagged"
    if node_ids and node_ids <= audit.valid_nodes:
        return "valid"
    return "unreachable"


def _connectivity_audit(graph) -> AuditReport:
    """PrimeKG-only control that removes LTN clause discrimination."""

    nodes = {str(node) for node in graph.nodes}
    connected = {node for node in nodes if graph.in_edges(node) or graph.out_edges(node)}
    return AuditReport(
        scores=ClauseScores(1.0, 1.0, 1.0),
        valid_nodes=connected,
        flagged_nodes=set(),
        rejected_nodes=nodes - connected,
    )


def _explanation_path(graph, claim_links, all_links) -> list[dict[str, object]]:
    claim_ids = [link.node_id for link in claim_links if link.node_id in graph]
    context_ids = [
        link.node_id for link in all_links if link.node_id in graph and link.node_id not in claim_ids
    ]
    if not claim_ids:
        return []
    source = claim_ids[0]
    target = next(iter(context_ids or claim_ids[1:]), None)
    node_lookup = {link.node_id: link.node_name for link in all_links}
    if target is None:
        return [{"node_id": source, "node_name": node_lookup.get(source, source)}]
    try:
        node_path = graph.shortest_path(source, target, directed=False, max_expansions=20_000)
    except ValueError:
        node_path = [source]
    return [
        {
            "node_id": node_id,
            "node_name": node_lookup.get(node_id, graph.nodes[node_id].get("name", node_id)),
        }
        for node_id in node_path
    ]


def _link_record(link: LinkedEntity) -> dict[str, object]:
    return {
        "mention": link.mention.text,
        "node_id": link.node_id,
        "node_name": link.node_name,
        "node_type": link.node_type,
        "negated": link.mention.negated,
        "link_confidence": link.confidence,
    }


def _unit_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
