from __future__ import annotations

from pathlib import Path
from typing import Sequence


def build_medgemma_prompt(
    indication: str,
    evidence_reports: Sequence[str] = (),
) -> str:
    prompt = [
        "You are an expert chest radiologist.",
        "Write the Findings section for the current chest X-ray.",
        "Describe only findings supported by the current image.",
        "Preserve negation and laterality. Do not mention the examples or explain your reasoning.",
    ]
    if indication.strip():
        prompt.append(f"Clinical indication: {indication.strip()}")
    if evidence_reports:
        prompt.append(
            "The following reports come from visually similar training radiographs. "
            "Use them only as reporting-style and retrieval evidence; do not copy an unsupported finding:"
        )
        prompt.extend(
            f"Example {index}: {' '.join(str(report).split())}"
            for index, report in enumerate(evidence_reports, start=1)
        )
    return "\n".join(prompt)


def build_medgemma_messages(
    image: object,
    *,
    indication: str = "",
    evidence_reports: Sequence[str] = (),
    target: str | None = None,
) -> list[dict[str, object]]:
    """Build the shared inference/SFT chat contract for MedGemma."""

    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_medgemma_prompt(indication, evidence_reports)},
            ],
        },
    ]
    if target is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": " ".join(str(target).split())}],
            },
        )
    return messages


class MedGemmaDrafter:
    """Chest X-ray drafting from a base MedGemma checkpoint and optional PEFT adapter."""

    def __init__(
        self,
        model_name: str = "google/medgemma-4b-it",
        *,
        adapter_path: str | Path | None = None,
        use_bf16: bool = True,
    ) -> None:
        deps = _dependencies(include_peft=bool(adapter_path))
        torch = deps["torch"]
        dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_available() else torch.float32
        processor_source = (
            str(adapter_path)
            if adapter_path and (Path(adapter_path) / "preprocessor_config.json").exists()
            else model_name
        )
        self.processor = deps["AutoProcessor"].from_pretrained(processor_source)
        self.model = deps["AutoModelForImageTextToText"].from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if adapter_path:
            self.model = deps["PeftModel"].from_pretrained(self.model, str(adapter_path))
        self.model.eval()
        self.torch = torch
        self.dtype = dtype

    def draft(
        self,
        image_path: str | Path,
        *,
        indication: str = "",
        evidence_reports: Sequence[str] = (),
        max_new_tokens: int = 180,
    ) -> str:
        image = _open_rgb(image_path)
        messages = build_medgemma_messages(
            image,
            indication=indication,
            evidence_reports=evidence_reports,
        )
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)
        input_length = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.08,
                no_repeat_ngram_size=3,
            )[0][input_length:]
        return self.processor.decode(output, skip_special_tokens=True).strip()


def _open_rgb(path: str | Path):
    deps = _dependencies()
    with deps["Image"].open(path) as image:
        return image.convert("RGB")


def _dependencies(*, include_peft: bool = False):
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Install dependencies with `pip install -e .[torch] accelerate`.") from exc
    dependencies = {
        "torch": torch,
        "Image": Image,
        "AutoProcessor": AutoProcessor,
        "AutoModelForImageTextToText": AutoModelForImageTextToText,
    }
    if include_peft:
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Adapter inference requires `pip install peft`.") from exc
        dependencies["PeftModel"] = PeftModel
    return dependencies
