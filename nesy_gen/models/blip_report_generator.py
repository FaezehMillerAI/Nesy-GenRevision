from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from nesy_gen.data.schema import RadiologyExample


@dataclass(frozen=True, slots=True)
class BlipGeneratorConfig:
    model_name: str = "Salesforce/blip-image-captioning-base"
    max_target_length: int = 160


def require_blip_dependencies():
    try:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader
        from transformers import BlipForConditionalGeneration, BlipProcessor
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Install dependencies with `pip install -e .[torch]`.") from exc
    return {
        "torch": torch,
        "Image": Image,
        "DataLoader": DataLoader,
        "BlipForConditionalGeneration": BlipForConditionalGeneration,
        "BlipProcessor": BlipProcessor,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }


def build_blip_processor_model(config: BlipGeneratorConfig):
    deps = require_blip_dependencies()
    processor = deps["BlipProcessor"].from_pretrained(config.model_name)
    model = deps["BlipForConditionalGeneration"].from_pretrained(config.model_name)
    return processor, model


class BlipReportDataset:
    def __init__(
        self,
        examples: Sequence[RadiologyExample],
        processor,
        *,
        max_target_length: int,
        include_labels: bool = True,
        prompt: str = "a chest x-ray report:",
    ) -> None:
        deps = require_blip_dependencies()
        self.Image = deps["Image"]
        self.examples = list(examples)
        self.processor = processor
        self.max_target_length = max_target_length
        self.include_labels = include_labels
        self.prompt = prompt

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        example = self.examples[idx]
        if not example.image_path:
            raise ValueError(f"Example {example.study_id} has no image_path")
        image = self.Image.open(example.image_path).convert("RGB")
        if self.include_labels:
            encoding = self.processor(
                images=image,
                text=example.report,
                padding="max_length",
                truncation=True,
                max_length=self.max_target_length,
                return_tensors="pt",
            )
            labels = encoding["input_ids"].squeeze(0).clone()
            pad_token_id = self.processor.tokenizer.pad_token_id
            labels[labels == pad_token_id] = -100
            return {
                "study_id": example.study_id,
                "pixel_values": encoding["pixel_values"].squeeze(0),
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": labels,
            }
        encoding = self.processor(images=image, text=self.prompt, return_tensors="pt")
        return {
            "study_id": example.study_id,
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def collate_blip_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    deps = require_blip_dependencies()
    torch = deps["torch"]
    result = {
        "study_id": [item["study_id"] for item in batch],
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
    }
    if "labels" in batch[0]:
        result["labels"] = torch.stack([item["labels"] for item in batch])
    return result


def save_blip_config(path: str | Path, config: BlipGeneratorConfig) -> None:
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

