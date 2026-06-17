from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from nesy_gen.kg.entity_linking import LexicalEntityLinker, LinkedEntity


CLINICAL_NODE_TYPES = (
    "anatomy",
    "disease",
    "effect/phenotype",
    "drug",
    "biological_process",
)


@dataclass(frozen=True, slots=True)
class DecodingConstraint:
    supported_token_ids: frozenset[int]
    penalized_token_ids: frozenset[int]
    linked_node_ids: tuple[str, ...]
    linked_node_names: tuple[str, ...]


class PrimeKGEntityLogitsProcessor:
    """Softly steer decoding toward patient/evidence-specific PrimeKG terms.

    This is intentionally a soft constraint. It can improve graph grounding
    without making the decoder unable to produce fluent text when the evidence
    terms are incomplete.
    """

    def __init__(
        self,
        constraints: Sequence[DecodingConstraint],
        *,
        token_boost: float = 2.0,
        unsupported_token_penalty: float = 0.0,
    ) -> None:
        self.constraints = list(constraints)
        self.token_boost = float(token_boost)
        self.unsupported_token_penalty = float(unsupported_token_penalty)
        self._tensor_cache: dict[tuple[int, str, tuple[int, ...]], object] = {}

    def __call__(self, input_ids, scores):
        if not self.constraints:
            return scores
        expanded_rows = scores.shape[0]
        expansion = max(1, expanded_rows // len(self.constraints))
        for row_idx in range(expanded_rows):
            constraint_idx = min(row_idx // expansion, len(self.constraints) - 1)
            constraint = self.constraints[constraint_idx]
            if constraint.supported_token_ids and self.token_boost:
                supported = self._tensor(
                    tuple(sorted(constraint.supported_token_ids)),
                    device=scores.device,
                    torch_module=_torch_from_scores(scores),
                )
                scores[row_idx, supported] += self.token_boost
            if constraint.penalized_token_ids and self.unsupported_token_penalty:
                penalized = self._tensor(
                    tuple(sorted(constraint.penalized_token_ids)),
                    device=scores.device,
                    torch_module=_torch_from_scores(scores),
                )
                scores[row_idx, penalized] -= self.unsupported_token_penalty
        return scores

    def _tensor(self, values: tuple[int, ...], *, device, torch_module):
        key = (id(device), str(device), values)
        if key not in self._tensor_cache:
            self._tensor_cache[key] = torch_module.tensor(values, dtype=torch_module.long, device=device)
        return self._tensor_cache[key]


class PrimeKGDecodingConstraintBuilder:
    """Build per-example decoding constraints from PrimeKG-linked evidence."""

    def __init__(
        self,
        nodes: pd.DataFrame,
        tokenizer,
        *,
        max_penalty_terms: int = 2500,
    ) -> None:
        vocab = nodes[["node_id", "node_name", "node_type"]].copy()
        vocab["alias"] = vocab["node_name"]
        self.linker = LexicalEntityLinker(vocab)
        self.tokenizer = tokenizer
        self.special_token_ids = {
            token_id
            for token_id in [
                getattr(tokenizer, "pad_token_id", None),
                getattr(tokenizer, "eos_token_id", None),
                getattr(tokenizer, "bos_token_id", None),
                getattr(tokenizer, "unk_token_id", None),
            ]
            if token_id is not None
        }
        self.clinical_token_ids = self._clinical_token_ids(vocab, max_terms=max_penalty_terms)

    def build(self, evidence_text: str) -> DecodingConstraint:
        links = self.linker.link_text(evidence_text)
        terms = _terms_from_links(links)
        supported_token_ids = set(self._token_ids_for_terms(terms))
        if any(link.mention.negated for link in links):
            supported_token_ids.update(self._token_ids_for_terms(["no", "without", "absent"]))
        penalized_token_ids = self.clinical_token_ids - supported_token_ids
        return DecodingConstraint(
            supported_token_ids=frozenset(supported_token_ids),
            penalized_token_ids=frozenset(penalized_token_ids),
            linked_node_ids=tuple(dict.fromkeys(link.node_id for link in links)),
            linked_node_names=tuple(dict.fromkeys(link.node_name for link in links)),
        )

    def processor(
        self,
        evidence_texts: Sequence[str],
        *,
        token_boost: float = 2.0,
        unsupported_token_penalty: float = 0.0,
    ) -> PrimeKGEntityLogitsProcessor:
        return PrimeKGEntityLogitsProcessor(
            [self.build(text) for text in evidence_texts],
            token_boost=token_boost,
            unsupported_token_penalty=unsupported_token_penalty,
        )

    def _clinical_token_ids(self, vocab: pd.DataFrame, *, max_terms: int) -> set[int]:
        clinical = vocab[
            vocab["node_type"]
            .astype(str)
            .str.lower()
            .map(lambda node_type: any(fragment in node_type for fragment in CLINICAL_NODE_TYPES))
        ]
        terms = (
            clinical["node_name"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values(key=lambda col: col.str.len(), ascending=False)
            .head(max_terms)
            .tolist()
        )
        return set(self._token_ids_for_terms(terms))

    def _token_ids_for_terms(self, terms: Iterable[str]) -> list[int]:
        token_ids: list[int] = []
        for term in terms:
            encoded = self.tokenizer.encode(str(term), add_special_tokens=False)
            token_ids.extend(token_id for token_id in encoded if token_id not in self.special_token_ids)
        return token_ids


def _terms_from_links(links: Sequence[LinkedEntity]) -> list[str]:
    terms: list[str] = []
    for link in links:
        terms.append(link.node_name)
        terms.append(link.mention.text)
    return [term for term in dict.fromkeys(terms) if term]


def _torch_from_scores(scores):
    import torch

    return torch
