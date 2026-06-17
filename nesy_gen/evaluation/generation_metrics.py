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
    tokenized = [
        (tokenize(getattr(row, "prediction")), tokenize(getattr(row, "reference")))
        for row in predictions.itertuples(index=False)
    ]
    rows = []
    for pred, ref in tokenized:
        prf = token_prf(pred, ref)
        rows.append(
            {
                "bleu1": sentence_bleu(pred, ref, max_n=1),
                "bleu2": sentence_bleu(pred, ref, max_n=2),
                "bleu3": sentence_bleu(pred, ref, max_n=3),
                "bleu4": sentence_bleu(pred, ref, max_n=4),
                "rouge_l": rouge_l(pred, ref),
                "meteor_lite": meteor_lite(pred, ref),
                "token_precision": prf["precision"],
                "token_recall": prf["recall"],
                "token_f1": prf["f1"],
            }
        )
    if not rows:
        return {
            key: 0.0
            for key in [
                "bleu1",
                "bleu2",
                "bleu3",
                "bleu4",
                "rouge_l",
                "meteor_lite",
                "token_precision",
                "token_recall",
                "token_f1",
                "cider_lite",
                "unique_prediction_ratio",
                "max_prediction_frequency_rate",
                "distinct_1",
                "distinct_2",
            ]
        }
    metrics = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
    metrics["cider_lite"] = cider_lite(tokenized)
    metrics.update(diversity_metrics(predictions, tokenized))
    return metrics


def diversity_metrics(
    predictions: pd.DataFrame,
    tokenized_pairs: list[tuple[list[str], list[str]]],
) -> dict[str, float]:
    prediction_texts = [" ".join(str(text).split()) for text in predictions["prediction"].tolist()]
    total = len(prediction_texts)
    if total == 0:
        return {
            "unique_prediction_ratio": 0.0,
            "max_prediction_frequency_rate": 0.0,
            "distinct_1": 0.0,
            "distinct_2": 0.0,
        }
    counts = Counter(prediction_texts)
    pred_tokens = [pred for pred, _ in tokenized_pairs]
    unigrams = [token for tokens in pred_tokens for token in tokens]
    bigrams = [ngram for tokens in pred_tokens for ngram in _ngrams(tokens, 2)]
    return {
        "unique_prediction_ratio": len(counts) / total,
        "max_prediction_frequency_rate": max(counts.values()) / total,
        "distinct_1": len(set(unigrams)) / max(1, len(unigrams)),
        "distinct_2": len(set(bigrams)) / max(1, len(bigrams)),
    }


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


def token_prf(pred: list[str], ref: list[str]) -> dict[str, float]:
    if not pred or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    pred_counts = Counter(pred)
    ref_counts = Counter(ref)
    matches = sum((pred_counts & ref_counts).values())
    precision = matches / len(pred) if pred else 0.0
    recall = matches / len(ref) if ref else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def cider_lite(tokenized_pairs: list[tuple[list[str], list[str]]], *, max_n: int = 4) -> float:
    """A lightweight CIDEr-style TF-IDF n-gram cosine score.

    This is not a replacement for the official COCO CIDEr implementation, but
    it gives a reproducible corpus-level CIDEr-like signal without adding a
    heavy dependency. Scores are scaled by 10, following common CIDEr reporting.
    """

    if not tokenized_pairs:
        return 0.0
    document_frequency: dict[tuple[int, tuple[str, ...]], int] = {}
    for _, ref in tokenized_pairs:
        seen = set()
        for n in range(1, max_n + 1):
            seen.update((n, ngram) for ngram in _ngrams(ref, n))
        for key in seen:
            document_frequency[key] = document_frequency.get(key, 0) + 1

    num_docs = len(tokenized_pairs)
    scores = []
    for pred, ref in tokenized_pairs:
        n_scores = []
        for n in range(1, max_n + 1):
            pred_vec = _tfidf_vector(pred, n, document_frequency, num_docs)
            ref_vec = _tfidf_vector(ref, n, document_frequency, num_docs)
            n_scores.append(_cosine(pred_vec, ref_vec))
        scores.append(10.0 * sum(n_scores) / max_n)
    return sum(scores) / len(scores)


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[idx : idx + n]) for idx in range(max(0, len(tokens) - n + 1)))


def _tfidf_vector(
    tokens: list[str],
    n: int,
    document_frequency: dict[tuple[int, tuple[str, ...]], int],
    num_docs: int,
) -> dict[tuple[str, ...], float]:
    counts = _ngrams(tokens, n)
    total = max(1, sum(counts.values()))
    vector = {}
    for ngram, count in counts.items():
        tf = count / total
        df = document_frequency.get((n, ngram), 0)
        idf = math.log((num_docs + 1.0) / (df + 1.0))
        vector[ngram] = tf * idf
    return vector


def _cosine(left: dict[tuple[str, ...], float], right: dict[tuple[str, ...], float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return 0.0 if left_norm == 0 or right_norm == 0 else numerator / (left_norm * right_norm)


def _lcs_len(left: list[str], right: list[str]) -> int:
    prev = [0] * (len(right) + 1)
    for token in left:
        curr = [0]
        for idx, other in enumerate(right, start=1):
            curr.append(prev[idx - 1] + 1 if token == other else max(prev[idx], curr[-1]))
        prev = curr
    return prev[-1]
