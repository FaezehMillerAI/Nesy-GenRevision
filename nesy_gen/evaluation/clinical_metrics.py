from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re

import pandas as pd

from nesy_gen.evaluation.generation_metrics import tokenize
from nesy_gen.kg.entity_linking import LexicalEntityLinker


CHEXPERT_CONDITIONS: dict[str, tuple[str, ...]] = {
    "atelectasis": ("atelectasis",),
    "cardiomegaly": ("cardiomegaly", "enlarged heart", "heart enlargement"),
    "consolidation": ("consolidation", "airspace opacity", "airspace disease"),
    "edema": ("edema", "pulmonary edema", "vascular congestion"),
    "pleural_effusion": ("pleural effusion", "effusion"),
    "pneumonia": ("pneumonia",),
    "pneumothorax": ("pneumothorax",),
    "fracture": ("fracture",),
    "lung_opacity": ("opacity", "opacities", "infiltrate", "infiltrates"),
    "lung_lesion": ("nodule", "mass", "lesion", "granuloma"),
}

NEGATION_MARKERS = {"no", "not", "without", "absent", "negative", "clear", "free"}


def chexpert_lite_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in predictions.itertuples(index=False):
        study_id = str(getattr(row, "study_id"))
        pred_labels = chexpert_lite_labels(str(getattr(row, "prediction")))
        ref_labels = chexpert_lite_labels(str(getattr(row, "reference")))
        row_values = {"study_id": study_id}
        for condition in CHEXPERT_CONDITIONS:
            row_values[f"pred_{condition}"] = pred_labels[condition]
            row_values[f"ref_{condition}"] = ref_labels[condition]
        row_values.update(_label_prf(pred_labels, ref_labels))
        row_values["positive_label_hallucination_rate"] = _positive_label_hallucination_rate(
            pred_labels,
            ref_labels,
        )
        row_values["label_mismatch_count"] = sum(
            1
            for condition in CHEXPERT_CONDITIONS
            if pred_labels[condition] != -1
            and ref_labels[condition] != -1
            and pred_labels[condition] != ref_labels[condition]
        )
        rows.append(row_values)
    return pd.DataFrame(rows)


def chexpert_lite_labels(text: str) -> dict[str, int]:
    """Return CheXpert-style labels: 1 positive, 0 negated, -1 absent.

    This deterministic labeler is intended for reproducible screening and
    ablations. For final clinical claims, prefer official CheXpert/CheXbert
    labels and feed them through the same comparison metrics.
    """

    labels = {condition: -1 for condition in CHEXPERT_CONDITIONS}
    for condition, aliases in CHEXPERT_CONDITIONS.items():
        for sentence in _sentences(text):
            normalized = " ".join(tokenize(sentence))
            for alias in aliases:
                match = re.search(rf"\b{re.escape(alias.lower())}\b", normalized)
                if not match:
                    continue
                labels[condition] = 0 if _is_negated(normalized, match.start()) else 1
                break
            if labels[condition] != -1:
                break
    return labels


def external_label_metrics(
    pred_labels: pd.DataFrame,
    ref_labels: pd.DataFrame,
    *,
    study_id_col: str = "study_id",
) -> pd.DataFrame:
    pred = pred_labels.copy()
    ref = ref_labels.copy()
    common = sorted((set(pred.columns) & set(ref.columns)) - {study_id_col})
    merged = pred[[study_id_col, *common]].merge(
        ref[[study_id_col, *common]],
        on=study_id_col,
        suffixes=("_pred", "_ref"),
    )
    rows = []
    for row in merged.itertuples(index=False):
        pred_map = {condition: getattr(row, f"{condition}_pred") for condition in common}
        ref_map = {condition: getattr(row, f"{condition}_ref") for condition in common}
        values = {"study_id": getattr(row, study_id_col)}
        values.update(_label_prf(pred_map, ref_map))
        values["positive_label_hallucination_rate"] = _positive_label_hallucination_rate(
            pred_map,
            ref_map,
        )
        rows.append(values)
    return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class RadGraphLiteItem:
    kind: str
    value: tuple[str, ...]


def radgraph_lite_frame(predictions: pd.DataFrame, linker: LexicalEntityLinker) -> pd.DataFrame:
    rows = []
    for row in predictions.itertuples(index=False):
        pred_items = radgraph_lite_items(str(getattr(row, "prediction")), linker)
        ref_items = radgraph_lite_items(str(getattr(row, "reference")), linker)
        prf = _set_prf(pred_items, ref_items)
        rows.append(
            {
                "study_id": str(getattr(row, "study_id")),
                "pred_items": len(pred_items),
                "ref_items": len(ref_items),
                "radgraph_lite_precision": prf["precision"],
                "radgraph_lite_recall": prf["recall"],
                "radgraph_lite_f1": prf["f1"],
            }
        )
    return pd.DataFrame(rows)


def radgraph_lite_items(text: str, linker: LexicalEntityLinker) -> set[tuple[str, ...]]:
    links = linker.link_text(text)
    items: set[tuple[str, ...]] = set()
    sorted_links = sorted(links, key=lambda link: link.mention.start)
    for link in sorted_links:
        assertion = "negated" if link.mention.negated else "positive"
        items.add(("entity", link.node_id, assertion))
    for left, right in zip(sorted_links, sorted_links[1:]):
        items.add(("cooccurs", left.node_id, right.node_id))
    return items


def _label_prf(pred_labels: dict[str, object], ref_labels: dict[str, object]) -> dict[str, float]:
    pred_positive = {key for key, value in pred_labels.items() if _is_positive(value)}
    ref_positive = {key for key, value in ref_labels.items() if _is_positive(value)}
    return _set_prf(pred_positive, ref_positive)


def _positive_label_hallucination_rate(
    pred_labels: dict[str, object],
    ref_labels: dict[str, object],
) -> float:
    pred_positive = {key for key, value in pred_labels.items() if _is_positive(value)}
    ref_positive = {key for key, value in ref_labels.items() if _is_positive(value)}
    if not pred_positive:
        return 0.0
    return len(pred_positive - ref_positive) / len(pred_positive)


def _set_prf(pred: Iterable[object], ref: Iterable[object]) -> dict[str, float]:
    pred_set = set(pred)
    ref_set = set(ref)
    true_positive = len(pred_set & ref_set)
    precision = 0.0 if not pred_set else true_positive / len(pred_set)
    recall = 0.0 if not ref_set else true_positive / len(ref_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _is_positive(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "positive", "pos", "present", "yes", "true"}
    return value == 1 or value is True


def _is_negated(norm_text: str, start: int, window: int = 36) -> bool:
    prefix = norm_text[max(0, start - window) : start]
    return bool(set(prefix.split()) & NEGATION_MARKERS)


def _sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in re.split(r"[.;\n]+", str(text)) if sentence.strip()]
    return sentences or [str(text)]
