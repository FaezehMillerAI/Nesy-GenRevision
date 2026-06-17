from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

from nesy_gen.data.schema import RadiologyExample


TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True, slots=True)
class RetrievalPrediction:
    study_id: str
    prediction: str
    retrieved_study_id: str
    similarity: float
    rank: int = 1


def run_tfidf_retrieval(
    train_examples: list[RadiologyExample],
    query_examples: list[RadiologyExample],
) -> list[RetrievalPrediction]:
    return [
        predictions[0]
        for predictions in run_tfidf_retrieval_topk(train_examples, query_examples, top_k=1)
        if predictions
    ]


def run_tfidf_retrieval_topk(
    train_examples: list[RadiologyExample],
    query_examples: list[RadiologyExample],
    *,
    top_k: int = 5,
) -> list[list[RetrievalPrediction]]:
    if not train_examples:
        raise ValueError("train_examples cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    train_queries = [_query_text(example) for example in train_examples]
    test_queries = [_query_text(example) for example in query_examples]
    idf = _idf(train_queries)
    train_vectors = [_tfidf_vector(text, idf) for text in train_queries]
    all_predictions: list[list[RetrievalPrediction]] = []
    for row_idx, example in enumerate(query_examples):
        query_vector = _tfidf_vector(test_queries[row_idx], idf)
        scores = [_cosine(query_vector, train_vector) for train_vector in train_vectors]
        ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        predictions = []
        for rank, train_idx in enumerate(ranked, start=1):
            retrieved = train_examples[train_idx]
            predictions.append(
                RetrievalPrediction(
                    study_id=example.study_id,
                    prediction=retrieved.report,
                    retrieved_study_id=retrieved.study_id,
                    similarity=float(scores[train_idx]),
                    rank=rank,
                )
            )
        all_predictions.append(predictions)
    return all_predictions


def _query_text(example: RadiologyExample) -> str:
    text = f"{example.indication} {example.metadata.get('problems', '')} {example.metadata.get('mesh', '')}"
    text = " ".join(text.split())
    return text if text else example.report


def _tokens(text: str) -> list[str]:
    words = TOKEN_RE.findall(text.lower())
    bigrams = [f"{words[idx]}_{words[idx + 1]}" for idx in range(len(words) - 1)]
    return words + bigrams


def _idf(texts: list[str]) -> dict[str, float]:
    doc_freq: Counter[str] = Counter()
    for text in texts:
        doc_freq.update(set(_tokens(text)))
    n_docs = len(texts)
    return {token: math.log((1 + n_docs) / (1 + freq)) + 1.0 for token, freq in doc_freq.items()}


def _tfidf_vector(text: str, idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(token for token in _tokens(text) if token in idf)
    if not counts:
        return {}
    total = sum(counts.values())
    return {token: (count / total) * idf[token] for token, count in counts.items()}


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
