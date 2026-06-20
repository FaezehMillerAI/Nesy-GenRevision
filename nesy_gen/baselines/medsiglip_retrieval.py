from __future__ import annotations

from pathlib import Path
import time
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
        self.model_name = model_name
        self.last_profile: dict[str, object] = {}
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
        if any(example.split != "train" for example in train):
            raise ValueError("MedSigLIP retrieval index may contain only training-split examples.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        retrieval_started = time.perf_counter()
        train_features, index_profile = self._load_or_build_index(train, batch_size, cache_path)
        query_started = time.perf_counter()
        query_features = self.encode(queries, batch_size=batch_size, desc="MedSigLIP query images")
        query_encoding_ms = (time.perf_counter() - query_started) * 1000.0
        search_started = time.perf_counter()
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
        search_ms = (time.perf_counter() - search_started) * 1000.0
        self.last_profile = {
            **index_profile,
            "query_count": len(queries),
            "query_encoding_ms": query_encoding_ms,
            "search_ms": search_ms,
            "online_retrieval_ms": query_encoding_ms + search_ms,
            "retrieval_total_ms": (time.perf_counter() - retrieval_started) * 1000.0,
            "embedding_dimension": int(train_features.shape[1]),
            "index_examples": int(train_features.shape[0]),
        }
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
        study_keys = np.asarray([_study_key(example) for example in train])
        image_paths = np.asarray([str(example.image_path) for example in train])
        if path and path.exists():
            load_started = time.perf_counter()
            with np.load(path) as cached:
                identity_matches = (
                    "study_keys" in cached
                    and "image_paths" in cached
                    and "model_name" in cached
                    and np.array_equal(cached["study_ids"].astype(str), study_ids.astype(str))
                    and np.array_equal(cached["study_keys"].astype(str), study_keys.astype(str))
                    and np.array_equal(cached["image_paths"].astype(str), image_paths.astype(str))
                    and str(cached["model_name"].item()) == self.model_name
                )
                if identity_matches:
                    features = cached["features"].astype(np.float32)
                    print(f"Reusing MedSigLIP training index: {path}")
                    return features, {
                        "index_reused": True,
                        "index_build_ms": 0.0,
                        "index_load_ms": (time.perf_counter() - load_started) * 1000.0,
                        "index_size_bytes": path.stat().st_size,
                        "index_path": str(path),
                    }
        build_started = time.perf_counter()
        features = self.encode(train, batch_size=batch_size, desc="MedSigLIP training index")
        index_build_ms = (time.perf_counter() - build_started) * 1000.0
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                study_ids=study_ids,
                study_keys=study_keys,
                image_paths=image_paths,
                model_name=np.asarray(self.model_name),
                features=features.astype(np.float16),
            )
            print(f"Saved MedSigLIP training index: {path}")
        return features, {
            "index_reused": False,
            "index_build_ms": index_build_ms,
            "index_load_ms": 0.0,
            "index_size_bytes": path.stat().st_size if path else features.nbytes,
            "index_path": str(path) if path else "",
        }


def _study_key(example: RadiologyExample) -> str:
    return str(
        example.metadata.get("underlying_study_id")
        or example.metadata.get("r2gen_id")
        or example.metadata.get("study_id")
        or example.metadata.get("uid")
        or example.study_id
    )


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
