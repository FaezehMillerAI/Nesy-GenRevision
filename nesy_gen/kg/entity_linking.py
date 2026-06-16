from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, NamedTuple

import pandas as pd


TOKEN_RE = re.compile(r"[a-z0-9]+")

DEFAULT_BLOCKED_ALIASES = {
    "acute",
    "ap",
    "borderline",
    "disease",
    "focal",
    "large",
    "lateral",
    "left",
    "mild",
    "moderate",
    "normal",
    "pa",
    "portable",
    "right",
    "small",
    "stable",
    "unchanged",
    "view",
    "views",
}

DEFAULT_BLOCKED_NODE_TYPE_FRAGMENTS = {
    "pathway",
}


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


class _AliasRecord(NamedTuple):
    alias: str
    norm_alias: str
    tokens: tuple[str, ...]
    node_id: str
    node_name: str
    node_type: str
    confidence: float


class LexicalEntityLinker:
    """Deterministic linker used for reproducible experiments and audits.

    Production runs can swap this with scispaCy, RadGraph, CheXbert, or RaTEScore
    extractors while preserving the same LinkedEntity output contract.
    """

    def __init__(
        self,
        vocabulary: pd.DataFrame,
        *,
        blocked_aliases: Iterable[str] | None = None,
        blocked_node_type_fragments: Iterable[str] | None = None,
    ):
        required = {"node_id", "node_name", "node_type"}
        missing = sorted(required - set(vocabulary.columns))
        if missing:
            raise ValueError(f"Vocabulary missing columns: {missing}")
        blocked_alias_set = {
            normalize_text(alias)
            for alias in (DEFAULT_BLOCKED_ALIASES if blocked_aliases is None else blocked_aliases)
        }
        blocked_type_fragments = tuple(
            fragment.lower()
            for fragment in (
                DEFAULT_BLOCKED_NODE_TYPE_FRAGMENTS
                if blocked_node_type_fragments is None
                else blocked_node_type_fragments
            )
        )
        vocab = vocabulary.copy()
        vocab["alias"] = vocab.get("alias", vocab["node_name"])
        vocab["norm_alias"] = vocab["alias"].map(normalize_text)
        vocab = vocab[vocab["norm_alias"].str.len() > 0]
        vocab = vocab[~vocab["norm_alias"].isin(blocked_alias_set)]
        vocab = vocab[
            ~vocab["node_type"]
            .astype(str)
            .str.lower()
            .map(lambda node_type: any(fragment in node_type for fragment in blocked_type_fragments))
        ]
        self.vocabulary = vocab.sort_values("norm_alias", key=lambda col: col.str.len(), ascending=False)
        self._aliases_by_first_token: dict[str, list[_AliasRecord]] = {}
        for row in self.vocabulary.itertuples(index=False):
            tokens = tuple(str(row.norm_alias).split())
            if not tokens:
                continue
            record = _AliasRecord(
                alias=str(row.alias),
                norm_alias=str(row.norm_alias),
                tokens=tokens,
                node_id=str(row.node_id),
                node_name=str(row.node_name),
                node_type=str(row.node_type),
                confidence=float(getattr(row, "confidence", 1.0)),
            )
            self._aliases_by_first_token.setdefault(tokens[0], []).append(record)
        for first_token, records in self._aliases_by_first_token.items():
            self._aliases_by_first_token[first_token] = sorted(
                records,
                key=lambda record: (len(record.tokens), len(record.norm_alias)),
                reverse=True,
            )

    @classmethod
    def from_primekg_nodes(cls, nodes: pd.DataFrame, **kwargs) -> "LexicalEntityLinker":
        return cls(nodes.rename(columns={"id": "node_id", "name": "node_name", "type": "node_type"}), **kwargs)

    @classmethod
    def from_crosswalk_csv(cls, path: str, **kwargs) -> "LexicalEntityLinker":
        return cls(pd.read_csv(path), **kwargs)

    def link_text(self, text: str) -> list[LinkedEntity]:
        norm = normalize_text(text)
        tokens = norm.split()
        spans = _token_spans(norm, tokens)
        links: list[LinkedEntity] = []
        occupied_tokens: set[int] = set()
        seen_mentions: set[tuple[str, int, int]] = set()
        for idx, token in enumerate(tokens):
            if idx in occupied_tokens:
                continue
            for record in self._aliases_by_first_token.get(token, []):
                end_idx = idx + len(record.tokens)
                if end_idx > len(tokens):
                    continue
                if any(pos in occupied_tokens for pos in range(idx, end_idx)):
                    continue
                if tuple(tokens[idx:end_idx]) != record.tokens:
                    continue
                start_char = spans[idx][0]
                end_char = spans[end_idx - 1][1]
                mention_key = (record.node_id, start_char, end_char)
                if mention_key in seen_mentions:
                    continue
                seen_mentions.add(mention_key)
                occupied_tokens.update(range(idx, end_idx))
                mention = EntityMention(
                    text=record.alias,
                    start=start_char,
                    end=end_char,
                    label=record.node_type,
                    negated=_is_negated(norm, start_char),
                )
                links.append(
                    LinkedEntity(
                        mention=mention,
                        node_id=record.node_id,
                        node_name=record.node_name,
                        node_type=record.node_type,
                        confidence=record.confidence,
                    )
                )
                break
        return links


def _token_spans(norm_text: str, tokens: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = norm_text.find(token, cursor)
        end = start + len(token)
        spans.append((start, end))
        cursor = end
    return spans


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
