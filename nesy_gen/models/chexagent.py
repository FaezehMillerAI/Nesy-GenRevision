from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Sequence


DEFAULT_CHEXAGENT_MODEL = "StanfordAIMI/CheXagent-2-3b-srrg-findings"
DEFAULT_CHEXAGENT_REVISION = "9f7225fc382ddd1297ade1aa796da660237940bc"


def build_chexagent_prompt(
    indication: str,
    evidence_reports: Sequence[str] = (),
) -> str:
    prompt = [
        "Generate only the narrative Findings section for the current chest X-ray.",
        "Use concise standard radiology language.",
        "Describe only observations supported by the current image.",
        "Preserve negation and laterality. Do not include headings, bullets, or an impression.",
    ]
    if indication.strip():
        prompt.append(f"Clinical indication: {indication.strip()}")
    if evidence_reports:
        prompt.append(
            "The following reports are from visually similar training radiographs. "
            "Use them only as non-authoritative style and retrieval evidence; never copy an "
            "unsupported finding:"
        )
        prompt.extend(
            f"Example {index}: {' '.join(str(report).split())}"
            for index, report in enumerate(evidence_reports, start=1)
        )
    return "\n".join(prompt)


def build_chexagent_conversation(
    image_path: str | Path,
    *,
    tokenizer,
    indication: str = "",
    evidence_reports: Sequence[str] = (),
    target: str | None = None,
) -> list[dict[str, str]]:
    query = tokenizer.from_list_format(
        [
            {"image": str(image_path)},
            {"text": build_chexagent_prompt(indication, evidence_reports)},
        ]
    )
    conversation = [
        {"from": "system", "value": "You are an expert chest radiologist."},
        {"from": "human", "value": query},
    ]
    if target is not None:
        conversation.append({"from": "gpt", "value": " ".join(str(target).split())})
    return conversation


class CheXagentDrafter:
    """Compact chest-X-ray-specific drafting with an optional PEFT adapter."""

    def __init__(
        self,
        model_name: str = DEFAULT_CHEXAGENT_MODEL,
        *,
        adapter_path: str | Path | None = None,
        use_bf16: bool = True,
    ) -> None:
        deps = _dependencies(include_peft=bool(adapter_path))
        torch = deps["torch"]
        dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_available() else torch.float32
        revision = (
            DEFAULT_CHEXAGENT_REVISION if model_name == DEFAULT_CHEXAGENT_MODEL else None
        )
        self.tokenizer = deps["AutoTokenizer"].from_pretrained(
            model_name,
            trust_remote_code=True,
            revision=revision,
        )
        self.model = deps["AutoModelForCausalLM"].from_pretrained(
            model_name,
            trust_remote_code=True,
            revision=revision,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        patch_chexagent_vision_forward(self.model)
        if adapter_path:
            self.model = deps["PeftModel"].from_pretrained(self.model, str(adapter_path))
        self.model.eval()
        self.torch = torch

    def draft(
        self,
        image_path: str | Path,
        *,
        indication: str = "",
        evidence_reports: Sequence[str] = (),
        max_new_tokens: int = 180,
    ) -> str:
        conversation = build_chexagent_conversation(
            image_path,
            tokenizer=self.tokenizer,
            indication=indication,
            evidence_reports=evidence_reports,
        )
        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0]
        return self.tokenizer.decode(
            output[input_ids.shape[-1] :],
            skip_special_tokens=True,
        ).strip()


def patch_chexagent_vision_forward(model) -> None:
    """Adapt pinned CheXagent vision features to the current SigLIP return API.

    CheXagent expects ``SiglipVisionTransformer(...).hidden_states[-1]`` from
    older Transformers. Current SigLIP drops that tuple from the public return
    object, while its encoder still exposes the same final pre-layernorm tensor
    as ``last_hidden_state``. Calling the encoder directly preserves the
    checkpoint's original feature semantics.
    """

    visual = model.model.visual
    if getattr(visual, "_nesy_gen_siglip_compat", False):
        return

    def compatible_forward(visual_self, pixel_values):
        embeddings = visual_self.model.embeddings(pixel_values)
        encoder_outputs = visual_self.model.encoder(inputs_embeds=embeddings)
        features = getattr(encoder_outputs, "last_hidden_state", None)
        if features is None:
            features = encoder_outputs[0]
        return visual_self.forward_resampler(features)

    visual.forward = MethodType(compatible_forward, visual)
    visual._nesy_gen_siglip_compat = True


def _dependencies(*, include_peft: bool = False):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "CheXagent requires `pip install -e .[finetune]`."
        ) from exc
    dependencies = {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }
    if include_peft:
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Adapter inference requires `pip install peft`.") from exc
        dependencies["PeftModel"] = PeftModel
    return dependencies
