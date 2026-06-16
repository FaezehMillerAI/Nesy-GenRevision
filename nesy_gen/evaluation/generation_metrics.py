from __future__ import annotations

from collections import Counter
import math
import re

import pandas as pd


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def corpus_generation_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    required = {"prediction", "reference"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions frame missing columns: {sorted(missing)}")
    rows = []
    for row in predictions.itertuples(index=False):
        pred = tokenize(getattr(row, "prediction"))
        ref = tokenize(getattr(row, "reference"))
        rows.append(
            {
                "bleu1": sentence_bleu(pred, ref, max_n=1),
                "bleu2": sentence_bleu(pred, ref, max_n=2),
                "bleu3": sentence_bleu(pred, ref, max_n=3),
                "bleu4": sentence_bleu(pred, ref, max_n=4),
                "rouge_l": rouge_l(pred, ref),
                "meteor_lite": meteor_lite(pred, ref),
            }
        )
    if not rows:
        return {key: 0.0 for key in ["bleu1", "bleu2", "bleu3", "bleu4", "rouge_l", "meteor_lite"]}
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def sentence_bleu(pred: list[str], ref: list[str], *, max_n: int = 4) -> float:
    if not pred or not ref:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = _ngrams(pred, n)
        ref_ngrams = _ngrams(ref, n)
        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = max(1, sum(pred_ngrams.values()))
        precisions.append((overlap + 1.0) / (total + 1.0))
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    brevity = 1.0 if len(pred) > len(ref) else math.exp(1.0 - len(ref) / max(1, len(pred)))
    return brevity * geo_mean


def rouge_l(pred: list[str], ref: list[str]) -> float:
    if not pred or not ref:
        return 0.0
    lcs = _lcs_len(pred, ref)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def meteor_lite(pred: list[str], ref: list[str]) -> float:
    if not pred or not ref:
        return 0.0
    pred_counts = Counter(pred)
    ref_counts = Counter(ref)
    matches = sum((pred_counts & ref_counts).values())
    if matches == 0:
        return 0.0
    precision = matches / len(pred)
    recall = matches / len(ref)
    return (10 * precision * recall) / (recall + 9 * precision) if (recall + 9 * precision) else 0.0


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[idx : idx + n]) for idx in range(max(0, len(tokens) - n + 1)))


def _lcs_len(left: list[str], right: list[str]) -> int:
    prev = [0] * (len(right) + 1)
    for token in left:
        curr = [0]
        for idx, other in enumerate(right, start=1):
            curr.append(prev[idx - 1] + 1 if token == other else max(prev[idx], curr[-1]))
        prev = curr
    return prev[-1]

