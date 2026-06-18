from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import warnings
from typing import Sequence

from nesy_gen.data.schema import RadiologyExample


@dataclass(frozen=True, slots=True)
class R2GenT5Config:
    text_model_name: str = "t5-small"
    tokenizer_name: str = "t5-small"
    visual_backbone: str = "resnet101"
    freeze_visual_encoder: bool = False
    image_size: int = 224
    max_target_length: int = 160
    visual_seq_len: int = 512  # retained for config file compatibility; no longer used
    dropout_prob: float = 0.1
    target_prefix: str = "generate report: "
    use_retrieval_conditioning: bool = False
    max_evidence_length: int = 192
    evidence_prefix: str = "retrieved evidence: "


def require_r2gen_t5_dependencies():
    try:
        import torch
        import torch.nn as nn
        from PIL import Image
        from torch.utils.data import DataLoader
        from torchvision import models, transforms
        from transformers import AutoTokenizer, LogitsProcessorList, T5ForConditionalGeneration
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
        "LogitsProcessorList": LogitsProcessorList,
        "T5ForConditionalGeneration": T5ForConditionalGeneration,
        "BaseModelOutput": BaseModelOutput,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }


class R2GenT5Model:
    """Vision-T5 image-to-report generator.

    The visual backbone produces a spatial feature map (e.g. 7×7 = 49 patch
    tokens for a 224×224 input).  Those patch tokens are projected into T5's
    encoder hidden size and used as the encoder sequence, so T5 cross-attention
    can attend to different image regions when generating each report token.

    The class name is retained for checkpoint compatibility with earlier
    experiments; paper-facing text should call this component the Vision-T5
    generator.
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

        self.visual_extractor, self.visual_projection = _build_visual_extractor(
            models,
            nn,
            config.visual_backbone,
            self.text_model.config.d_model,
        )
        if config.freeze_visual_encoder:
            _freeze_visual_backbone(self.visual_extractor, self.visual_projection)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        self.text_model.to(device)
        self.visual_extractor.to(device)
        self.visual_projection.to(device)
        self.dropout.to(device)
        return self

    def train(self):
        self.text_model.train()
        self.visual_projection.train()
        if self.config.freeze_visual_encoder:
            self.visual_extractor.eval()
        else:
            self.visual_extractor.train()
        self.dropout.train()

    def eval(self):
        self.text_model.eval()
        self.visual_extractor.eval()
        self.visual_projection.eval()
        self.dropout.eval()

    def parameters(self):
        yield from self.text_model.parameters()
        yield from (p for p in self.visual_extractor.parameters() if p.requires_grad)
        yield from (p for p in self.visual_projection.parameters() if p.requires_grad)

    def forward(self, images, labels=None, evidence_input_ids=None, evidence_attention_mask=None):
        encoder_outputs, attention_mask = self._multimodal_encoder_outputs(
            images,
            evidence_input_ids=evidence_input_ids,
            evidence_attention_mask=evidence_attention_mask,
        )
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
        logits_processor=None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        length_penalty: float = 1.0,
        num_beam_groups: int = 1,
        diversity_penalty: float = 0.0,
        evidence_input_ids=None,
        evidence_attention_mask=None,
    ):
        encoder_outputs, attention_mask = self._multimodal_encoder_outputs(
            images,
            evidence_input_ids=evidence_input_ids,
            evidence_attention_mask=evidence_attention_mask,
        )
        kwargs = {
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": num_return_sequences,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "length_penalty": length_penalty,
        }
        if logits_processor is not None:
            kwargs["logits_processor"] = logits_processor
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
            effective_beams = max(num_beams, num_return_sequences)
            kwargs["num_beams"] = effective_beams
            if num_beam_groups > 1:
                if effective_beams % num_beam_groups != 0:
                    raise ValueError("num_beams must be divisible by num_beam_groups.")
                kwargs["num_beam_groups"] = num_beam_groups
                kwargs["diversity_penalty"] = diversity_penalty
        return self.text_model.generate(**kwargs)

    def _multimodal_encoder_outputs(
        self,
        images,
        *,
        evidence_input_ids=None,
        evidence_attention_mask=None,
    ):
        deps = require_r2gen_t5_dependencies()
        torch = deps["torch"]
        BaseModelOutput = deps["BaseModelOutput"]
        visual_outputs, visual_mask = self._visual_encoder_outputs(images)
        if evidence_input_ids is None:
            return visual_outputs, visual_mask
        evidence_input_ids = evidence_input_ids.to(self.device)
        evidence_attention_mask = evidence_attention_mask.to(self.device)
        text_model = getattr(self.text_model, "_orig_mod", self.text_model)
        evidence_outputs = text_model.encoder(
            input_ids=evidence_input_ids,
            attention_mask=evidence_attention_mask,
            return_dict=True,
        ).last_hidden_state
        hidden = torch.cat([visual_outputs.last_hidden_state, evidence_outputs], dim=1)
        attention_mask = torch.cat([visual_mask, evidence_attention_mask], dim=1)
        return BaseModelOutput(last_hidden_state=hidden), attention_mask

    def _visual_encoder_outputs(self, images):
        deps = require_r2gen_t5_dependencies()
        torch = deps["torch"]
        BaseModelOutput = deps["BaseModelOutput"]

        images = images.to(self.device)
        # visual_extractor returns [B, num_patches, backbone_channels]
        patch_tokens = self.visual_extractor(images)
        patch_tokens = self.dropout(patch_tokens)
        # project backbone channels → T5 d_model
        features = self.visual_projection(patch_tokens)  # [B, num_patches, d_model]
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
        text_model_to_save = getattr(self.text_model, "_orig_mod", self.text_model)
        text_model_to_save.save_pretrained(out / "text_model")
        self.tokenizer.save_pretrained(out / "tokenizer")
        torch.save(self.visual_extractor.state_dict(), out / "visual_extractor.pt")
        torch.save(self.visual_projection.state_dict(), out / "visual_projection.pt")
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
        extractor_state = torch.load(checkpoint / "visual_extractor.pt", map_location="cpu")
        _load_visual_extractor_state(model, extractor_state, checkpoint)
        proj_path = checkpoint / "visual_projection.pt"
        if proj_path.exists():
            proj_state = torch.load(proj_path, map_location="cpu")
            model.visual_projection.load_state_dict(proj_state)
        elif _load_legacy_projection_state(model, extractor_state):
            warnings.warn(
                f"Loaded legacy projection weights from visual_extractor.pt in {checkpoint}. "
                "Retraining is still recommended for spatial patch features.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"No visual_projection.pt found in {checkpoint}. "
                "Checkpoint was saved with the old architecture — projection weights are random. "
                "Retrain the model to use spatial patch features.",
                UserWarning,
                stacklevel=2,
            )
        return model


def _load_visual_extractor_state(model, extractor_state: dict, checkpoint: Path) -> None:
    try:
        model.visual_extractor.load_state_dict(extractor_state)
        return
    except RuntimeError as exc:
        legacy_state = _legacy_resnet_state_for_spatial_extractor(extractor_state)
        if legacy_state:
            missing, unexpected = model.visual_extractor.load_state_dict(legacy_state, strict=False)
            warnings.warn(
                (
                    f"Loaded legacy global-pool ResNet checkpoint from {checkpoint} into the "
                    f"spatial visual extractor with missing={len(missing)}, unexpected={len(unexpected)}. "
                    "Retraining is recommended before using this checkpoint for final experiments."
                ),
                UserWarning,
                stacklevel=2,
            )
            return
        raise RuntimeError(
            f"Could not load visual_extractor.pt from {checkpoint}. "
            "The checkpoint may use an incompatible visual backbone. Retrain the Vision-T5 checkpoint."
        ) from exc


def _legacy_resnet_state_for_spatial_extractor(state: dict) -> dict[str, object]:
    prefixes = {
        "conv1.": "backbone.0.",
        "bn1.": "backbone.1.",
        "layer1.": "backbone.4.",
        "layer2.": "backbone.5.",
        "layer3.": "backbone.6.",
        "layer4.": "backbone.7.",
    }
    converted = {}
    for key, value in state.items():
        for old, new in prefixes.items():
            if key.startswith(old):
                converted[f"{new}{key[len(old):]}"] = value
                break
    return converted


def _load_legacy_projection_state(model, extractor_state: dict) -> bool:
    if "fc.weight" not in extractor_state or "fc.bias" not in extractor_state:
        return False
    try:
        model.visual_projection.load_state_dict(
            {"weight": extractor_state["fc.weight"], "bias": extractor_state["fc.bias"]}
        )
        return True
    except RuntimeError:
        return False


def _build_visual_extractor(models, nn, visual_backbone: str, output_dim: int):
    """Return (spatial_extractor, projection) for the requested backbone.

    spatial_extractor: nn.Module that takes [B, 3, H, W] images and returns
        [B, num_patches, backbone_channels] patch tokens (no global pooling).
    projection: nn.Linear(backbone_channels, output_dim) — kept separate so
        it can be unfrozen independently of the backbone during fine-tuning.
    """

    class _SpatialExtractor(nn.Module):
        """Wraps a CNN/ViT backbone; returns [B, N, C] spatial patch tokens."""

        def __init__(self, backbone, *, layout: str = "nchw"):
            super().__init__()
            self.backbone = backbone
            # layout: "nchw" → feature map [B,C,H,W]; "nhwc" → [B,H,W,C] (Swin)
            self.layout = layout

        def forward(self, x):
            out = self.backbone(x)
            if self.layout == "nchw":
                B, C, H, W = out.shape
                return out.permute(0, 2, 3, 1).reshape(B, H * W, C)
            else:
                B, H, W, C = out.shape
                return out.reshape(B, H * W, C)

    backbone = visual_backbone.lower()

    if backbone == "resnet101":
        try:
            base = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        except AttributeError:  # pragma: no cover - old torchvision
            base = models.resnet101(pretrained=True)
        # Remove avgpool and fc to keep the 7×7 spatial feature map (2048 ch)
        trunk = nn.Sequential(*list(base.children())[:-2])
        return _SpatialExtractor(trunk, layout="nchw"), nn.Linear(2048, output_dim)

    if backbone == "convnext_base":
        base = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        # base.features ends at 7×7 × 1024 before avgpool
        return _SpatialExtractor(base.features, layout="nchw"), nn.Linear(1024, output_dim)

    if backbone == "efficientnet_v2_s":
        base = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        # base.features ends at 7×7 × 1280 before avgpool
        return _SpatialExtractor(base.features, layout="nchw"), nn.Linear(1280, output_dim)

    if backbone == "swin_t":
        base = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
        # Swin outputs [B, 7, 7, 768] in NHWC after features + norm
        trunk = nn.Sequential(base.features, base.norm)
        return _SpatialExtractor(trunk, layout="nhwc"), nn.Linear(768, output_dim)

    if backbone == "densenet121":
        base = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        # features output needs ReLU before spatial extraction
        trunk = nn.Sequential(base.features, nn.ReLU(inplace=True))
        return _SpatialExtractor(trunk, layout="nchw"), nn.Linear(1024, output_dim)

    raise ValueError(
        "Unsupported visual_backbone. Choose one of: "
        "resnet101, convnext_base, efficientnet_v2_s, swin_t, densenet121."
    )


def _freeze_visual_backbone(visual_extractor, projection) -> None:
    for parameter in visual_extractor.parameters():
        parameter.requires_grad = False
    for parameter in projection.parameters():
        parameter.requires_grad = True


class R2GenT5Dataset:
    def __init__(
        self,
        examples: Sequence[RadiologyExample],
        tokenizer,
        *,
        max_target_length: int,
        include_labels: bool = True,
        target_prefix: str = "generate report: ",
        image_size: int = 224,
        evidence_by_study_id: dict[str, list[str]] | None = None,
        max_evidence_length: int = 192,
        evidence_prefix: str = "retrieved evidence: ",
    ) -> None:
        deps = require_r2gen_t5_dependencies()
        transforms = deps["transforms"]
        self.Image = deps["Image"]
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.include_labels = include_labels
        self.target_prefix = target_prefix
        self.image_size = image_size
        self.evidence_by_study_id = evidence_by_study_id
        self.max_evidence_length = max_evidence_length
        self.evidence_prefix = evidence_prefix
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # Pre-compute prefix token length so __getitem__ can mask it efficiently
        self._prefix_token_len = 0
        if target_prefix:
            self._prefix_token_len = len(
                tokenizer.encode(target_prefix.strip(), add_special_tokens=False)
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
            "indication": example.indication,
            "report": example.report,
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
            # Mask the decoder-prompt prefix so the loss only covers the actual report
            if self._prefix_token_len > 0:
                labels[: self._prefix_token_len] = -100
            result["labels"] = labels
        if self.evidence_by_study_id is not None:
            evidence_reports = self.evidence_by_study_id.get(example.study_id, [])
            evidence = f"{self.evidence_prefix}{' '.join(evidence_reports)}".strip()
            encoded_evidence = self.tokenizer(
                evidence,
                return_tensors="pt",
                max_length=self.max_evidence_length,
                truncation=True,
                padding="max_length",
            )
            result["evidence_input_ids"] = encoded_evidence["input_ids"].squeeze(0)
            result["evidence_attention_mask"] = encoded_evidence["attention_mask"].squeeze(0)
        return result


def collate_r2gen_t5_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    deps = require_r2gen_t5_dependencies()
    torch = deps["torch"]
    result = {
        "study_id": [item["study_id"] for item in batch],
        "image": torch.stack([item["image"] for item in batch]),
        "indication": [item["indication"] for item in batch],
        "report": [item["report"] for item in batch],
    }
    if "labels" in batch[0]:
        result["labels"] = torch.stack([item["labels"] for item in batch])
    if "evidence_input_ids" in batch[0]:
        result["evidence_input_ids"] = torch.stack(
            [item["evidence_input_ids"] for item in batch]
        )
        result["evidence_attention_mask"] = torch.stack(
            [item["evidence_attention_mask"] for item in batch]
        )
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
