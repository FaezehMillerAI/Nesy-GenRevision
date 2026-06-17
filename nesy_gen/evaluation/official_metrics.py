from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def official_coco_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Compute official COCO-caption metrics via pycocoevalcap.

    Requires:
      pip install pycocoevalcap

    METEOR also requires Java at runtime, matching the upstream COCO tooling.
    """

    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Official COCO metrics require `pip install pycocoevalcap`. "
            "For METEOR, Java must also be available."
        ) from exc

    gts: dict[int, list[str]] = {}
    res: dict[int, list[str]] = {}
    for idx, row in enumerate(predictions.itertuples(index=False)):
        gts[idx] = [str(getattr(row, "reference"))]
        res[idx] = [str(getattr(row, "prediction"))]

    bleu_score, _ = Bleu(4).compute_score(gts, res)
    meteor_score, _ = Meteor().compute_score(gts, res)
    rouge_score, _ = Rouge().compute_score(gts, res)
    cider_score, _ = Cider().compute_score(gts, res)
    return {
        "bleu1": float(bleu_score[0]),
        "bleu2": float(bleu_score[1]),
        "bleu3": float(bleu_score[2]),
        "bleu4": float(bleu_score[3]),
        "meteor": float(meteor_score),
        "rouge_l": float(rouge_score),
        "cider": float(cider_score),
    }


def chexbert_label_metrics(
    pred_labels: pd.DataFrame,
    ref_labels: pd.DataFrame,
    *,
    study_id_col: str = "study_id",
) -> pd.DataFrame:
    """Compare official CheXbert/CheXpert label CSVs.

    The input CSVs should contain one row per study and the same condition
    columns. Values can follow CheXbert/CheXpert conventions: 1 positive,
    0 negative, -1 uncertain, blank/NaN absent.
    """

    pred = _normalize_label_frame(pred_labels, study_id_col=study_id_col)
    ref = _normalize_label_frame(ref_labels, study_id_col=study_id_col)
    common = sorted((set(pred.columns) & set(ref.columns)) - {study_id_col})
    if not common:
        raise ValueError("No common label columns found between prediction and reference labels.")
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
        values["label_mismatch_count"] = sum(
            1
            for condition in common
            if not pd.isna(pred_map[condition])
            and not pd.isna(ref_map[condition])
            and pred_map[condition] != ref_map[condition]
        )
        rows.append(values)
    return pd.DataFrame(rows)


def official_radgraph_metrics(
    pred_annotations: str | Path | dict[str, Any],
    ref_annotations: str | Path | dict[str, Any],
) -> pd.DataFrame:
    """Compare official RadGraph annotation outputs.

    This function expects annotation JSON produced by the official RadGraph
    inference pipeline, keyed by study/report id. It flexibly handles common
    RadGraph-like structures with `entities` dictionaries containing labels,
    tokens, relations, and assertion/status fields.
    """

    pred = _load_json(pred_annotations)
    ref = _load_json(ref_annotations)
    common_ids = sorted(set(pred) & set(ref))
    rows = []
    for study_id in common_ids:
        pred_items = _radgraph_items(pred[study_id])
        ref_items = _radgraph_items(ref[study_id])
        prf = _set_prf(pred_items, ref_items)
        rows.append(
            {
                "study_id": study_id,
                "pred_items": len(pred_items),
                "ref_items": len(ref_items),
                "radgraph_precision": prf["precision"],
                "radgraph_recall": prf["recall"],
                "radgraph_f1": prf["f1"],
            }
        )
    return pd.DataFrame(rows)


def prepare_official_report_csv(
    predictions: pd.DataFrame,
    output_csv: str | Path,
    *,
    text_column: str,
    output_text_column: str = "Report Impression",
) -> None:
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = predictions[["study_id", text_column]].rename(columns={text_column: output_text_column})
    frame.to_csv(out, index=False)


def _normalize_label_frame(frame: pd.DataFrame, *, study_id_col: str) -> pd.DataFrame:
    result = frame.copy()
    if study_id_col not in result.columns:
        raise ValueError(f"Label CSV is missing study id column: {study_id_col}")
    rename = {column: _normalize_label_name(column) for column in result.columns if column != study_id_col}
    return result.rename(columns=rename)


def _normalize_label_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


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


def _is_positive(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "positive", "pos", "present", "yes", "true"}
    return value == 1 or value is True


def _set_prf(pred: set[object], ref: set[object]) -> dict[str, float]:
    true_positive = len(pred & ref)
    precision = 0.0 if not pred else true_positive / len(pred)
    recall = 0.0 if not ref else true_positive / len(ref)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _load_json(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _radgraph_items(annotation: dict[str, Any]) -> set[tuple[str, ...]]:
    entities = annotation.get("entities", annotation.get("entity", {}))
    if isinstance(entities, list):
        entities = {str(idx): entity for idx, entity in enumerate(entities)}
    items: set[tuple[str, ...]] = set()
    for entity_id, entity in entities.items():
        if not isinstance(entity, dict):
            continue
        label = str(entity.get("label", entity.get("type", entity.get("entity_label", ""))))
        tokens = entity.get("tokens", entity.get("text", entity.get("mention", "")))
        if isinstance(tokens, list):
            text = " ".join(str(token) for token in tokens)
        else:
            text = str(tokens)
        assertion = str(entity.get("status", entity.get("assertion", entity.get("observation", ""))))
        items.add(("entity", label.lower(), text.lower(), assertion.lower()))
        relations = entity.get("relations", entity.get("relation", []))
        for relation in relations:
            if isinstance(relation, dict):
                relation_label = str(relation.get("label", relation.get("type", "")))
                target = str(relation.get("target", relation.get("object", "")))
            elif isinstance(relation, (list, tuple)) and len(relation) >= 2:
                relation_label = str(relation[0])
                target = str(relation[1])
            else:
                continue
            target_entity = entities.get(str(target), {})
            target_label = ""
            target_text = str(target)
            if isinstance(target_entity, dict):
                target_label = str(target_entity.get("label", target_entity.get("type", ""))).lower()
                target_tokens = target_entity.get("tokens", target_entity.get("text", target))
                target_text = (
                    " ".join(str(token) for token in target_tokens)
                    if isinstance(target_tokens, list)
                    else str(target_tokens)
                )
            items.add(
                (
                    "relation",
                    label.lower(),
                    text.lower(),
                    relation_label.lower(),
                    target_label,
                    target_text.lower(),
                )
            )
    return items
