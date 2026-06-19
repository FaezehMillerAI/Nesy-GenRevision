from __future__ import annotations

from pathlib import Path
from typing import Sequence


def build_medgemma_prompt(
    indication: str,
    evidence_reports: Sequence[str] = (),
) -> str:
    prompt = [
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


class MedGemmaDrafter:
    """Training-free chest X-ray drafting through an official MedGemma checkpoint."""

    def __init__(
        self,
        model_name: str = "google/medgemma-4b-it",
        *,
        use_bf16: bool = True,
    ) -> None:
        deps = _dependencies()
        torch = deps["torch"]
        dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_available() else torch.float32
        self.processor = deps["AutoProcessor"].from_pretrained(model_name)
        self.model = deps["AutoModelForImageTextToText"].from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
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
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert chest radiologist."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_medgemma_prompt(indication, evidence_reports)},
                    {"type": "image", "image": image},
                ],
            },
        ]
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


def _dependencies():
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Install dependencies with `pip install -e .[torch] accelerate`.") from exc
    return {
        "torch": torch,
        "Image": Image,
        "AutoProcessor": AutoProcessor,
        "AutoModelForImageTextToText": AutoModelForImageTextToText,
    }
