from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import pandas as pd


TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


@dataclass(frozen=True, slots=True)
class EntityMention:
    text: str
    start: int
    end: int
    label: str = "clinical_entity"
    negated: bool = False


@dataclass(frozen=True, slots=True)
class LinkedEntity:
    mention: EntityMention
    node_id: str
    node_name: str
    node_type: str
    confidence: float
    source: str = "lexical"


class LexicalEntityLinker:
    """Deterministic linker used for reproducible experiments and audits.

    Production runs can swap this with scispaCy, RadGraph, CheXbert, or RaTEScore
    extractors while preserving the same LinkedEntity output contract.
    """

    def __init__(self, vocabulary: pd.DataFrame):
        required = {"node_id", "node_name", "node_type"}
        missing = sorted(required - set(vocabulary.columns))
        if missing:
            raise ValueError(f"Vocabulary missing columns: {missing}")
        vocab = vocabulary.copy()
        vocab["alias"] = vocab.get("alias", vocab["node_name"])
        vocab["norm_alias"] = vocab["alias"].map(normalize_text)
        vocab = vocab[vocab["norm_alias"].str.len() > 0]
        self.vocabulary = vocab.sort_values("norm_alias", key=lambda col: col.str.len(), ascending=False)

    @classmethod
    def from_primekg_nodes(cls, nodes: pd.DataFrame) -> "LexicalEntityLinker":
        return cls(nodes.rename(columns={"id": "node_id", "name": "node_name", "type": "node_type"}))

    @classmethod
    def from_crosswalk_csv(cls, path: str) -> "LexicalEntityLinker":
        return cls(pd.read_csv(path))

    def link_text(self, text: str) -> list[LinkedEntity]:
        norm = normalize_text(text)
        links: list[LinkedEntity] = []
        occupied: set[int] = set()
        seen_mentions: set[tuple[str, int, int]] = set()
        for row in self.vocabulary.itertuples(index=False):
            alias = str(row.norm_alias)
            pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
            for match in pattern.finditer(norm):
                span = set(range(match.start(), match.end()))
                if span & occupied:
                    continue
                mention_key = (str(row.node_id), match.start(), match.end())
                if mention_key in seen_mentions:
                    continue
                seen_mentions.add(mention_key)
                occupied |= span
                mention = EntityMention(
                    text=str(row.alias),
                    start=match.start(),
                    end=match.end(),
                    label=str(row.node_type),
                    negated=_is_negated(norm, match.start()),
                )
                links.append(
                    LinkedEntity(
                        mention=mention,
                        node_id=str(row.node_id),
                        node_name=str(row.node_name),
                        node_type=str(row.node_type),
                        confidence=float(getattr(row, "confidence", 1.0)),
                    )
                )
        return links


def _is_negated(norm_text: str, start: int, window: int = 28) -> bool:
    prefix = norm_text[max(0, start - window) : start]
    return any(marker in prefix.split() for marker in {"no", "not", "without", "absent"})


def entity_linking_scores(
    predicted: Iterable[LinkedEntity],
    gold_node_ids: Iterable[str],
) -> dict[str, float]:
    pred = {link.node_id for link in predicted}
    gold = {str(node_id) for node_id in gold_node_ids}
    tp = len(pred & gold)
    precision = 0.0 if not pred else tp / len(pred)
    recall = 0.0 if not gold else tp / len(gold)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}
