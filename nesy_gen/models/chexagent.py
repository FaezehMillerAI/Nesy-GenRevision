from __future__ import annotations

from pathlib import Path
from typing import Sequence


DEFAULT_CHEXAGENT_MODEL = "StanfordAIMI/CheXagent-2-3b-srrg-findings"


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
        self.tokenizer = deps["AutoTokenizer"].from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = deps["AutoModelForCausalLM"].from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
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
