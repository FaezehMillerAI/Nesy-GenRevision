from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from nesy_gen.data.schema import RadiologyExample


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    encoder_model: str = "microsoft/swin-tiny-patch4-window7-224"
    decoder_model: str = "distilgpt2"
    pretrained_vision_encoder_decoder_model: str | None = None
    max_target_length: int = 160
    image_size: int = 224


def require_torch_transformers():
    try:
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoTokenizer,
            VisionEncoderDecoderModel,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Install training dependencies with `pip install -e .[torch] pillow` "
            "or run the Colab install cell before using report generation."
        ) from exc
    return {
        "torch": torch,
        "Image": Image,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "AutoConfig": AutoConfig,
        "AutoImageProcessor": AutoImageProcessor,
        "AutoTokenizer": AutoTokenizer,
        "VisionEncoderDecoderModel": VisionEncoderDecoderModel,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }


def build_processor_tokenizer_model(config: GeneratorConfig):
    deps = require_torch_transformers()
    tokenizer = deps["AutoTokenizer"].from_pretrained(config.decoder_model)
    if config.pretrained_vision_encoder_decoder_model:
        tokenizer = deps["AutoTokenizer"].from_pretrained(
            config.pretrained_vision_encoder_decoder_model
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if config.pretrained_vision_encoder_decoder_model:
        image_processor = deps["AutoImageProcessor"].from_pretrained(
            config.pretrained_vision_encoder_decoder_model
        )
        model = deps["VisionEncoderDecoderModel"].from_pretrained(
            config.pretrained_vision_encoder_decoder_model
        )
    else:
        image_processor = deps["AutoImageProcessor"].from_pretrained(config.encoder_model)
        decoder_config = deps["AutoConfig"].from_pretrained(config.decoder_model)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        decoder_config.pad_token_id = tokenizer.pad_token_id
        decoder_config.bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        decoder_config.eos_token_id = tokenizer.eos_token_id
        model = deps["VisionEncoderDecoderModel"].from_encoder_decoder_pretrained(
            config.encoder_model,
            config.decoder_model,
            decoder_config=decoder_config,
        )
    model.config.decoder_start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.max_length = config.max_target_length
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.early_stopping = False
    model.generation_config.num_beams = 1
    model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    model.generation_config.eos_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.pad_token_id
    return image_processor, tokenizer, model


class ReportGenerationDataset:
    def __init__(
        self,
        examples: Sequence[RadiologyExample],
        image_processor,
        tokenizer,
        *,
        max_target_length: int,
        include_labels: bool = True,
    ) -> None:
        deps = require_torch_transformers()
        self.torch = deps["torch"]
        self.Image = deps["Image"]
        self.examples = list(examples)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.include_labels = include_labels

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        example = self.examples[idx]
        if not example.image_path:
            raise ValueError(f"Example {example.study_id} has no image_path")
        image = self.Image.open(example.image_path).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        item: dict[str, object] = {
            "study_id": example.study_id,
            "pixel_values": pixel_values,
        }
        if self.include_labels:
            tokens = self.tokenizer(
                example.report,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            labels = tokens.input_ids.squeeze(0)
            labels[labels == self.tokenizer.pad_token_id] = -100
            item["labels"] = labels
            item["decoder_attention_mask"] = tokens.attention_mask.squeeze(0)
        return item


def collate_generation_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    deps = require_torch_transformers()
    torch = deps["torch"]
    result = {
        "study_id": [item["study_id"] for item in batch],
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
    }
    if "labels" in batch[0]:
        result["labels"] = torch.stack([item["labels"] for item in batch])
        result["decoder_attention_mask"] = torch.stack(
            [item["decoder_attention_mask"] for item in batch]
        )
    return result


def save_generator_config(path: str | Path, config: GeneratorConfig) -> None:
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
