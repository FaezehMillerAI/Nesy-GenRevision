from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from nesy_gen.data.schema import RadiologyExample
from nesy_gen.evaluation.metrics import entity_f1, hallucination_rate
from nesy_gen.kg.entity_linking import LexicalEntityLinker, LinkedEntity


@dataclass(frozen=True, slots=True)
class ReportPair:
    study_id: str
    prediction: str
    reference: str


def positive_node_ids(links: Iterable[LinkedEntity]) -> set[str]:
    return {link.node_id for link in links if not link.mention.negated}


def negated_node_ids(links: Iterable[LinkedEntity]) -> set[str]:
    return {link.node_id for link in links if link.mention.negated}


def evaluate_report_pairs(pairs: Iterable[ReportPair], linker: LexicalEntityLinker) -> pd.DataFrame:
    rows = []
    for pair in pairs:
        pred_links = linker.link_text(pair.prediction)
        ref_links = linker.link_text(pair.reference)
        pred_positive = positive_node_ids(pred_links)
        ref_positive = positive_node_ids(ref_links)
        pred_negated = negated_node_ids(pred_links)
        ref_negated = negated_node_ids(ref_links)
        f1 = entity_f1(pred_positive, ref_positive)
        halluc = hallucination_rate(pred_positive, ref_positive)
        negation_mismatch = len((pred_positive & ref_negated) | (pred_negated & ref_positive))
        rows.append(
            {
                "study_id": pair.study_id,
                "pred_positive": len(pred_positive),
                "ref_positive": len(ref_positive),
                "pred_negated": len(pred_negated),
                "ref_negated": len(ref_negated),
                "precision": f1["precision"],
                "recall": f1["recall"],
                "f1": f1["f1"],
                "unsupported_count": halluc["unsupported_count"],
                "hallucination_rate": halluc["hallucination_rate"],
                "negation_mismatch_count": negation_mismatch,
            }
        )
    return pd.DataFrame(rows)


def examples_to_pairs(predictions: pd.DataFrame, references: list[RadiologyExample]) -> list[ReportPair]:
    reference_by_id = {example.study_id: example for example in references}
    pairs: list[ReportPair] = []
    for row in predictions.itertuples(index=False):
        study_id = str(getattr(row, "study_id"))
        if study_id not in reference_by_id:
            continue
        pairs.append(
            ReportPair(
                study_id=study_id,
                prediction=str(getattr(row, "prediction")),
                reference=reference_by_id[study_id].report,
            )
        )
    return pairs

