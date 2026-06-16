from __future__ import annotations

from dataclasses import dataclass

from nesy_gen.kg.entity_linking import LexicalEntityLinker, LinkedEntity
from nesy_gen.kg.temporal import TemporalSubgraphBuilder
from nesy_gen.logic.ltn import AuditReport, NeuroSymbolicAuditor
from nesy_gen.models.gate import CandidateToken, ConsistencyDecision, ConsistencyGate


@dataclass(slots=True)
class NesyGenPipeline:
    """Composable neuro-symbolic path used by training, evaluation, and demos."""

    linker: LexicalEntityLinker
    subgraph_builder: TemporalSubgraphBuilder
    auditor: NeuroSymbolicAuditor
    gate: ConsistencyGate

    def reason(self, indication: str, draft_report: str = "") -> tuple[list[LinkedEntity], AuditReport]:
        text = f"{indication} {draft_report}".strip()
        links = self.linker.link_text(text)
        graph = self.subgraph_builder.build(links)
        audit = self.auditor.audit(graph)
        return links, audit

    def verify_tokens(
        self,
        indication: str,
        tokens: list[CandidateToken],
        draft_report: str = "",
    ) -> list[ConsistencyDecision]:
        _, audit = self.reason(indication=indication, draft_report=draft_report)
        return [self.gate.decide(token, audit) for token in tokens]

