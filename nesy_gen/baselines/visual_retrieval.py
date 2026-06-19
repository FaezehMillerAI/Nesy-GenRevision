from __future__ import annotations

from collections.abc import Sequence

from tqdm.auto import tqdm

from nesy_gen.baselines.retrieval import RetrievalPrediction
from nesy_gen.data.schema import RadiologyExample
from nesy_gen.models.r2gen_t5 import R2GenT5Dataset, collate_r2gen_t5_batch


def run_visual_retrieval_topk(
    model,
    train_examples: Sequence[RadiologyExample],
    query_examples: Sequence[RadiologyExample],
    *,
    top_k: int = 5,
    batch_size: int = 16,
    progress_desc: str = "visual retrieval",
) -> list[list[RetrievalPrediction]]:
    """Retrieve training reports using only frozen visual representations."""

    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    train = [example for example in train_examples if example.image_path]
    queries = [example for example in query_examples if example.image_path]
    if not train or not queries:
        raise ValueError("Visual retrieval needs train and query images.")

    deps = model_dependencies()
    torch = deps["torch"]
    train_features = extract_visual_embeddings(
        model,
        train,
        batch_size=batch_size,
        progress_desc=f"{progress_desc}: training index",
    )
    if train is queries or _same_example_sequence(train, queries):
        query_features = train_features
    else:
        query_features = extract_visual_embeddings(
            model,
            queries,
            batch_size=batch_size,
            progress_desc=f"{progress_desc}: query images",
        )

    predictions: list[list[RetrievalPrediction]] = []
    # Chunked scoring avoids materializing an N_query x N_train matrix for MIMIC-CXR.
    query_chunks = range(0, len(queries), 256)
    for start in tqdm(
        query_chunks,
        total=(len(queries) + 255) // 256,
        desc=f"{progress_desc}: nearest neighbours",
        dynamic_ncols=True,
    ):
        scores = query_features[start : start + 256] @ train_features.T
        for local_idx, row_scores in enumerate(scores):
            query = queries[start + local_idx]
            blocked = {
                idx for idx, candidate in enumerate(train) if _study_key(candidate) == _study_key(query)
            }
            ranked = _unique_study_ranking(
                train,
                torch.argsort(row_scores, descending=True).tolist(),
                blocked=blocked,
                top_k=top_k,
            )
            predictions.append(
                [
                    RetrievalPrediction(
                        study_id=query.study_id,
                        prediction=train[idx].report,
                        retrieved_study_id=train[idx].study_id,
                        similarity=float(row_scores[idx]),
                        rank=rank,
                    )
                    for rank, idx in enumerate(ranked, start=1)
                ]
            )
    return predictions


def visual_evidence_map(
    model,
    train_examples: Sequence[RadiologyExample],
    query_examples: Sequence[RadiologyExample],
    *,
    top_k: int = 3,
    batch_size: int = 16,
    progress_desc: str = "visual RAG evidence",
) -> dict[str, list[str]]:
    rows = run_visual_retrieval_topk(
        model,
        train_examples,
        query_examples,
        top_k=top_k,
        batch_size=batch_size,
        progress_desc=progress_desc,
    )
    return {
        query.study_id: [prediction.prediction for prediction in predictions]
        for query, predictions in zip(query_examples, rows, strict=True)
    }


def extract_visual_embeddings(
    model,
    examples,
    *,
    batch_size: int = 16,
    progress_desc: str = "visual embeddings",
):
    deps = model_dependencies()
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]
    loader = DataLoader(
        R2GenT5Dataset(
            examples,
            model.tokenizer,
            max_target_length=model.config.max_target_length,
            include_labels=False,
            image_size=model.config.image_size,
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_r2gen_t5_batch,
    )
    was_training = model.text_model.training
    model.eval()
    features = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=progress_desc, dynamic_ncols=True):
            patch_tokens = model.visual_extractor(batch["image"].to(model.device))
            # Retrieval must remain stable while the projection/T5 layers train.
            # The backbone is frozen, so its pooled representation is the index key.
            pooled = patch_tokens.mean(dim=1)
            features.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
    if was_training:
        model.train()
    return torch.cat(features, dim=0)


def model_dependencies():
    from nesy_gen.models.r2gen_t5 import require_r2gen_t5_dependencies

    return require_r2gen_t5_dependencies()


def _study_key(example: RadiologyExample) -> str:
    return str(example.metadata.get("r2gen_id") or example.study_id)


def _same_example_sequence(left, right) -> bool:
    return len(left) == len(right) and all(
        first.study_id == second.study_id for first, second in zip(left, right, strict=True)
    )


def _unique_study_ranking(examples, ranked_indices, *, blocked, top_k):
    selected = []
    seen_studies = set()
    for idx in ranked_indices:
        key = _study_key(examples[idx])
        if idx in blocked or key in seen_studies:
            continue
        selected.append(idx)
        seen_studies.add(key)
        if len(selected) == top_k:
            break
    return selected
