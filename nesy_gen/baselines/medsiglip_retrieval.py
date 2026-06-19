from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm.auto import tqdm

from nesy_gen.baselines.retrieval import RetrievalPrediction
from nesy_gen.data.schema import RadiologyExample


class MedSiglipRetriever:
    """Frozen medical image retrieval with an optional persistent training index."""

    def __init__(self, model_name: str = "google/medsiglip-448") -> None:
        deps = _dependencies()
        torch = deps["torch"]
        self.torch = torch
        self.processor = deps["AutoProcessor"].from_pretrained(model_name)
        self.model = deps["AutoModel"].from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def retrieve(
        self,
        train_examples: Sequence[RadiologyExample],
        query_examples: Sequence[RadiologyExample],
        *,
        top_k: int = 5,
        batch_size: int = 16,
        cache_path: str | Path | None = None,
    ) -> list[list[RetrievalPrediction]]:
        train = [example for example in train_examples if example.image_path]
        queries = [example for example in query_examples if example.image_path]
        train_features = self._load_or_build_index(train, batch_size, cache_path)
        query_features = self.encode(queries, batch_size=batch_size, desc="MedSigLIP query images")
        outputs = []
        for query_index, query in enumerate(tqdm(queries, desc="MedSigLIP nearest reports")):
            scores = query_features[query_index] @ train_features.T
            order = np.argsort(-scores)
            selected = []
            seen = set()
            query_key = _study_key(query)
            for index in order:
                candidate = train[int(index)]
                key = _study_key(candidate)
                if key == query_key or key in seen:
                    continue
                seen.add(key)
                selected.append(int(index))
                if len(selected) == top_k:
                    break
            outputs.append(
                [
                    RetrievalPrediction(
                        study_id=query.study_id,
                        prediction=train[index].report,
                        retrieved_study_id=train[index].study_id,
                        similarity=float(scores[index]),
                        rank=rank,
                    )
                    for rank, index in enumerate(selected, start=1)
                ]
            )
        return outputs

    def encode(self, examples, *, batch_size: int, desc: str) -> np.ndarray:
        features = []
        for start in tqdm(range(0, len(examples), batch_size), desc=desc):
            batch = examples[start : start + batch_size]
            images = [_open_rgb(example.image_path) for example in batch]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                embeddings = self.model.get_image_features(**inputs)
                if hasattr(embeddings, "pooler_output"):
                    embeddings = embeddings.pooler_output
                embeddings = self.torch.nn.functional.normalize(embeddings, dim=-1)
            features.append(embeddings.float().cpu().numpy())
        if not features:
            raise ValueError("No images available for MedSigLIP retrieval.")
        return np.concatenate(features, axis=0)

    def _load_or_build_index(self, train, batch_size, cache_path):
        path = Path(cache_path) if cache_path else None
        study_ids = np.asarray([example.study_id for example in train])
        if path and path.exists():
            cached = np.load(path)
            if np.array_equal(cached["study_ids"].astype(str), study_ids.astype(str)):
                print(f"Reusing MedSigLIP training index: {path}")
                return cached["features"].astype(np.float32)
        features = self.encode(train, batch_size=batch_size, desc="MedSigLIP training index")
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, study_ids=study_ids, features=features.astype(np.float16))
            print(f"Saved MedSigLIP training index: {path}")
        return features


def _study_key(example: RadiologyExample) -> str:
    return str(example.metadata.get("r2gen_id") or example.study_id)


def _open_rgb(path):
    deps = _dependencies()
    with deps["Image"].open(path) as image:
        return image.convert("RGB")


def _dependencies():
    try:
        import torch
        from PIL import Image
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Install dependencies with `pip install -e .[torch]`.") from exc
    return {"torch": torch, "Image": Image, "AutoModel": AutoModel, "AutoProcessor": AutoProcessor}
