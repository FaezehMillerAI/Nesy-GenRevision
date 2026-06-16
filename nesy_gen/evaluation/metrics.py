from __future__ import annotations

from collections.abc import Iterable


def entity_f1(predicted_node_ids: Iterable[str], reference_node_ids: Iterable[str]) -> dict[str, float]:
    predicted = {str(node_id) for node_id in predicted_node_ids}
    reference = {str(node_id) for node_id in reference_node_ids}
    tp = len(predicted & reference)
    precision = 0.0 if not predicted else tp / len(predicted)
    recall = 0.0 if not reference else tp / len(reference)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def hallucination_rate(predicted_node_ids: Iterable[str], reference_node_ids: Iterable[str]) -> dict[str, float]:
    predicted = {str(node_id) for node_id in predicted_node_ids}
    reference = {str(node_id) for node_id in reference_node_ids}
    unsupported = predicted - reference
    return {
        "unsupported_count": float(len(unsupported)),
        "predicted_count": float(len(predicted)),
        "hallucination_rate": 0.0 if not predicted else len(unsupported) / len(predicted),
    }

