from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

from nesy_gen.data.schema import RadiologyExample


@dataclass(frozen=True, slots=True)
class R2GenT5Config:
    text_model_name: str = "t5-small"
    tokenizer_name: str = "t5-small"
    visual_backbone: str = "resnet101"
    max_target_length: int = 160
    visual_seq_len: int = 512
    dropout_prob: float = 0.1
    target_prefix: str = "generate report: "


def require_r2gen_t5_dependencies():
    try:
        import torch
        import torch.nn as nn
        from PIL import Image
        from torch.utils.data import DataLoader
        from torchvision import models, transforms
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        from transformers.modeling_outputs import BaseModelOutput
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Install dependencies with `pip install -e .[torch]`.") from exc
    return {
        "torch": torch,
        "nn": nn,
        "Image": Image,
        "DataLoader": DataLoader,
        "models": models,
        "transforms": transforms,
        "AutoTokenizer": AutoTokenizer,
        "T5ForConditionalGeneration": T5ForConditionalGeneration,
        "BaseModelOutput": BaseModelOutput,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }


class R2GenT5Model:
    """R2Gen-style visual encoder plus T5 decoder.

    This follows the attached notebook's core idea: a ResNet visual extractor
    projects image features into T5's encoder hidden size, then T5 decodes the
    report from those visual features.
    """

    def __init__(self, config: R2GenT5Config):
        deps = require_r2gen_t5_dependencies()
        torch = deps["torch"]
        nn = deps["nn"]
        models = deps["models"]
        AutoTokenizer = deps["AutoTokenizer"]
        T5ForConditionalGeneration = deps["T5ForConditionalGeneration"]

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.text_model = T5ForConditionalGeneration.from_pretrained(config.text_model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.text_model.config.pad_token_id is None:
            self.text_model.config.pad_token_id = self.tokenizer.pad_token_id

        if config.visual_backbone != "resnet101":
            raise ValueError("Only visual_backbone='resnet101' is currently supported.")
        try:
            weights = models.ResNet101_Weights.DEFAULT
            self.visual_extractor = models.resnet101(weights=weights)
        except AttributeError:  # pragma: no cover - old torchvision
            self.visual_extractor = models.resnet101(pretrained=True)
        self.visual_extractor.fc = nn.Linear(
            self.visual_extractor.fc.in_features,
            self.text_model.config.d_model,
        )
        self.dropout = nn.Dropout(config.dropout_prob)
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        self.text_model.to(device)
        self.visual_extractor.to(device)
        self.dropout.to(device)
        return self

    def train(self):
        self.text_model.train()
        self.visual_extractor.train()
        self.dropout.train()

    def eval(self):
        self.text_model.eval()
        self.visual_extractor.eval()
        self.dropout.eval()

    def parameters(self):
        yield from self.text_model.parameters()
        yield from self.visual_extractor.parameters()

    def forward(self, images, labels=None):
        encoder_outputs, attention_mask = self._visual_encoder_outputs(images)
        return self.text_model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
        )

    def generate(
        self,
        images,
        *,
        max_new_tokens: int = 120,
        num_beams: int = 3,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        top_p: float = 0.9,
        temperature: float = 0.8,
    ):
        encoder_outputs, attention_mask = self._visual_encoder_outputs(images)
        kwargs = {
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": num_return_sequences,
        }
        if do_sample:
            kwargs.update(
                {
                    "do_sample": True,
                    "top_p": top_p,
                    "temperature": temperature,
                    "num_beams": 1,
                }
            )
        else:
            kwargs["num_beams"] = max(num_beams, num_return_sequences)
        return self.text_model.generate(**kwargs)

    def _visual_encoder_outputs(self, images):
        deps = require_r2gen_t5_dependencies()
        torch = deps["torch"]
        BaseModelOutput = deps["BaseModelOutput"]

        images = images.to(self.device)
        features = self.visual_extractor(images)
        features = self.dropout(features)
        features = features.unsqueeze(1).expand(
            images.shape[0],
            self.config.visual_seq_len,
            features.shape[-1],
        )
        attention_mask = torch.ones(
            features.shape[:2],
            dtype=torch.long,
            device=features.device,
        )
        return BaseModelOutput(last_hidden_state=features), attention_mask

    def save_pretrained(self, output_dir: str | Path) -> None:
        deps = require_r2gen_t5_dependencies()
        torch = deps["torch"]
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.text_model.save_pretrained(out / "text_model")
        self.tokenizer.save_pretrained(out / "tokenizer")
        torch.save(self.visual_extractor.state_dict(), out / "visual_extractor.pt")
        save_r2gen_t5_config(out / "r2gen_t5_config.json", self.config)

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str | Path):
        deps = require_r2gen_t5_dependencies()
        torch = deps["torch"]
        AutoTokenizer = deps["AutoTokenizer"]
        T5ForConditionalGeneration = deps["T5ForConditionalGeneration"]

        checkpoint = Path(checkpoint_dir)
        config = load_r2gen_t5_config(checkpoint / "r2gen_t5_config.json")
        model = cls(config)
        model.text_model = T5ForConditionalGeneration.from_pretrained(checkpoint / "text_model")
        model.tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer")
        state = torch.load(checkpoint / "visual_extractor.pt", map_location="cpu")
        model.visual_extractor.load_state_dict(state)
        return model


class R2GenT5Dataset:
    def __init__(
        self,
        examples: Sequence[RadiologyExample],
        tokenizer,
        *,
        max_target_length: int,
        include_labels: bool = True,
        target_prefix: str = "generate report: ",
    ) -> None:
        deps = require_r2gen_t5_dependencies()
        transforms = deps["transforms"]
        self.Image = deps["Image"]
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.include_labels = include_labels
        self.target_prefix = target_prefix
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        example = self.examples[idx]
        if not example.image_path:
            raise ValueError(f"Example {example.study_id} has no image_path")
        image = self.Image.open(example.image_path).convert("RGB")
        image = self.image_transform(image)
        result: dict[str, object] = {
            "study_id": example.study_id,
            "image": image,
        }
        if self.include_labels:
            target = f"{self.target_prefix}{example.report}".strip()
            encoding = self.tokenizer(
                target,
                return_tensors="pt",
                max_length=self.max_target_length,
                truncation=True,
                padding="max_length",
            )
            labels = encoding["input_ids"].squeeze(0).clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            result["labels"] = labels
        return result


def collate_r2gen_t5_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    deps = require_r2gen_t5_dependencies()
    torch = deps["torch"]
    result = {
        "study_id": [item["study_id"] for item in batch],
        "image": torch.stack([item["image"] for item in batch]),
    }
    if "labels" in batch[0]:
        result["labels"] = torch.stack([item["labels"] for item in batch])
    return result


def decode_r2gen_predictions(tokenizer, generated_ids, *, target_prefix: str = "generate report: ") -> list[str]:
    texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return [clean_r2gen_prediction(text, target_prefix=target_prefix) for text in texts]


def clean_r2gen_prediction(text: str, *, target_prefix: str = "generate report: ") -> str:
    cleaned = " ".join(str(text).split())
    prefix = target_prefix.strip()
    if prefix and cleaned.lower().startswith(prefix.lower()):
        cleaned = cleaned[len(prefix) :].strip()
    return cleaned


def save_r2gen_t5_config(path: str | Path, config: R2GenT5Config) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def load_r2gen_t5_config(path: str | Path) -> R2GenT5Config:
    return R2GenT5Config(**json.loads(Path(path).read_text(encoding="utf-8")))
