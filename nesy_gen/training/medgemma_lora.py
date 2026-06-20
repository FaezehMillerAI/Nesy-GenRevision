from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import re
from typing import Mapping, Sequence

from nesy_gen.data.schema import RadiologyExample
from nesy_gen.models.chexagent import build_chexagent_conversation
from nesy_gen.models.medgemma import build_medgemma_messages


SECTION_RE = re.compile(r"\b(?:findings?|impression)\s*:\s*", re.IGNORECASE)


def normalize_findings_target(report: str) -> str:
    """Normalize a report to the Findings-style target used by generation."""

    text = " ".join(str(report).split())
    if not text:
        return ""
    matches = list(SECTION_RE.finditer(text))
    finding_match = next(
        (match for match in matches if match.group(0).lower().startswith("finding")),
        None,
    )
    if finding_match is not None:
        start = finding_match.end()
        impression_match = next(
            (
                match
                for match in matches
                if match.start() > start and match.group(0).lower().startswith("impression")
            ),
            None,
        )
        findings = text[start : impression_match.start() if impression_match else None].strip()
        if findings:
            return findings
    return text


def build_sft_rows(
    examples: Sequence[RadiologyExample],
    *,
    evidence_by_study_id: Mapping[str, Sequence[dict[str, str]]] | None = None,
    retrieval_probability: float = 0.5,
    few_shot_k: int = 3,
    seed: int = 13,
) -> list[dict[str, object]]:
    """Build deterministic multimodal SFT rows without reading another split's references."""

    if not 0.0 <= retrieval_probability <= 1.0:
        raise ValueError("retrieval_probability must be between 0 and 1")
    if few_shot_k < 0:
        raise ValueError("few_shot_k must be non-negative")
    rows: list[dict[str, object]] = []
    evidence_map = evidence_by_study_id or {}
    for example in examples:
        if not example.image_path:
            continue
        target = normalize_findings_target(example.report)
        if not target:
            continue
        candidates = list(evidence_map.get(example.study_id, ()))
        use_retrieval = bool(candidates) and _deterministic_probability(
            example.study_id, seed
        ) < retrieval_probability
        selected = candidates[:few_shot_k] if use_retrieval else []
        evidence_reports = [str(candidate.get("report", "")) for candidate in selected]
        evidence_reports = [report for report in evidence_reports if report.strip()]
        rows.append(
            {
                "study_id": example.study_id,
                "image_path": str(example.image_path),
                "indication": example.indication,
                "evidence_reports": evidence_reports,
                "messages": build_medgemma_messages(
                    None,
                    indication=example.indication,
                    evidence_reports=evidence_reports,
                    target=target,
                ),
                "target": target,
                "used_retrieval": bool(evidence_reports),
                "evidence_study_ids": [
                    str(candidate.get("study_id", "")) for candidate in selected
                ],
            }
        )
    return rows


class MedGemmaSFTCollator:
    """Official-style multimodal SFT collation with image/padding loss masking."""

    def __init__(self, processor, *, max_length: int = 2048) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: list[dict[str, object]]):
        images = []
        texts = []
        for example in examples:
            image = _open_rgb(str(example["image_path"]))
            messages = deepcopy(example["messages"])
            _inject_image(messages, image)
            images.append([image])
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=False,
                ).strip()
            )
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = batch["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        for token_id in _image_token_ids(self.processor):
            labels[labels == token_id] = -100
        batch["labels"] = labels
        return batch


class CheXagentSFTCollator:
    """Mask CheXagent image/prompt tokens so loss is applied only to Findings."""

    def __init__(self, tokenizer, *, max_length: int = 2048) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: list[dict[str, object]]):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional training dependency
            raise ImportError("CheXagent QLoRA requires PyTorch.") from exc

        rows: list[tuple[list[int], list[int]]] = []
        for example in examples:
            prompt = build_chexagent_conversation(
                str(example["image_path"]),
                tokenizer=self.tokenizer,
                indication=str(example.get("indication", "")),
                evidence_reports=list(example.get("evidence_reports", [])),
            )
            complete = build_chexagent_conversation(
                str(example["image_path"]),
                tokenizer=self.tokenizer,
                indication=str(example.get("indication", "")),
                evidence_reports=list(example.get("evidence_reports", [])),
                target=str(example["target"]),
            )
            prompt_ids = _as_token_ids(
                self.tokenizer.apply_chat_template(
                    prompt,
                    add_generation_prompt=True,
                    return_tensors=None,
                )
            )
            input_ids = _as_token_ids(
                self.tokenizer.apply_chat_template(
                    complete,
                    add_generation_prompt=False,
                    return_tensors=None,
                )
            )[: self.max_length]
            labels = input_ids.copy()
            labels[: min(len(prompt_ids), len(labels))] = [-100] * min(
                len(prompt_ids), len(labels)
            )
            rows.append((input_ids, labels))

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        max_length = max(len(input_ids) for input_ids, _ in rows)
        batch_input_ids = torch.full(
            (len(rows), max_length), int(pad_token_id), dtype=torch.long
        )
        attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
        batch_labels = torch.full((len(rows), max_length), -100, dtype=torch.long)
        for index, (input_ids, labels) in enumerate(rows):
            length = len(input_ids)
            batch_input_ids[index, :length] = torch.tensor(input_ids, dtype=torch.long)
            attention_mask[index, :length] = 1
            batch_labels[index, :length] = torch.tensor(labels, dtype=torch.long)
        return {
            "input_ids": batch_input_ids,
            "attention_mask": attention_mask,
            "labels": batch_labels,
        }


def _deterministic_probability(study_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{study_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _as_token_ids(value) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]


def _inject_image(messages: list[dict[str, object]], image: object) -> None:
    for message in messages:
        for content in message.get("content", []):
            if content.get("type") == "image":
                content["image"] = image
                return
    raise ValueError("SFT message is missing an image content block")


def _image_token_ids(processor) -> set[int]:
    tokenizer = processor.tokenizer
    ids = {262144}  # Gemma 3 image soft-token id used by the official recipe.
    boi_token = tokenizer.special_tokens_map.get("boi_token")
    if boi_token:
        ids.add(int(tokenizer.convert_tokens_to_ids(boi_token)))
    for token in ("<start_of_image>", "<image_soft_token>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            ids.add(token_id)
    return ids


def _open_rgb(path: str | Path):
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise ImportError("Install Pillow through `pip install -e .[finetune]`.") from exc
    with Image.open(path) as image:
        return image.convert("RGB")
