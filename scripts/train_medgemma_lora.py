from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.baselines.medsiglip_retrieval import MedSiglipRetriever  # noqa: E402
from nesy_gen.data.schema import load_jsonl  # noqa: E402
from nesy_gen.models.chexagent import (  # noqa: E402
    DEFAULT_CHEXAGENT_MODEL,
    DEFAULT_CHEXAGENT_REVISION,
)
from nesy_gen.training.medgemma_lora import (  # noqa: E402
    CheXagentSFTCollator,
    MedGemmaSFTCollator,
    build_sft_rows,
)


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    _validate_split_contract(args.train_split, args.eval_split)
    if not 0.0 <= args.retrieval_probability <= 1.0:
        parser.error("--retrieval-probability must be between 0 and 1")
    deps = _dependencies()
    torch = deps["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("Multimodal radiology QLoRA requires a CUDA GPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Select a GPU with bfloat16 support (A100 recommended).")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_jsonl(args.manifest)
    train = [
        example
        for example in examples
        if example.split == args.train_split and example.image_path and example.report.strip()
    ]
    evaluation = [
        example
        for example in examples
        if example.split == args.eval_split and example.image_path and example.report.strip()
    ]
    if args.max_train_examples:
        train = train[: args.max_train_examples]
    if args.max_eval_examples:
        evaluation = evaluation[: args.max_eval_examples]
    if not train or not evaluation:
        raise ValueError("Training and evaluation splits must contain image-report pairs.")

    evidence, retrieval_profile = _build_retrieval_evidence(args, train, evaluation)
    train_rows = build_sft_rows(
        train,
        evidence_by_study_id=evidence,
        retrieval_probability=args.retrieval_probability,
        few_shot_k=args.few_shot_k,
        seed=args.seed,
    )
    eval_rows = build_sft_rows(
        evaluation,
        evidence_by_study_id=evidence,
        retrieval_probability=args.retrieval_probability,
        few_shot_k=args.few_shot_k,
        seed=args.seed,
    )
    train_dataset = deps["Dataset"].from_list(train_rows)
    eval_dataset = deps["Dataset"].from_list(eval_rows)

    quantization = deps["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
        llm_int8_skip_modules=_vision_quantization_skip_modules(args.model_family),
    )
    if args.model_family == "chexagent":
        model_revision = (
            DEFAULT_CHEXAGENT_REVISION
            if args.model_name == DEFAULT_CHEXAGENT_MODEL
            else None
        )
        model = deps["AutoModelForCausalLM"].from_pretrained(
            args.model_name,
            trust_remote_code=True,
            revision=model_revision,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization,
        )
        processor = deps["AutoTokenizer"].from_pretrained(
            args.model_name,
            trust_remote_code=True,
            revision=model_revision,
        )
        processor.padding_side = "right"
    else:
        model_revision = None
        model = deps["AutoModelForImageTextToText"].from_pretrained(
            args.model_name,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization,
        )
        processor = deps["AutoProcessor"].from_pretrained(args.model_name)
        processor.tokenizer.padding_side = "right"
    model.config.use_cache = False
    model = deps["prepare_model_for_kbit_training"](
        model,
        use_gradient_checkpointing=True,
    )

    modules_to_save = []
    if args.train_embedding_layers and args.model_family == "medgemma":
        modules_to_save.extend(["lm_head", "embed_tokens"])
    if args.train_connector and args.model_family == "medgemma":
        modules_to_save.append("multi_modal_projector")
    target_modules = "all-linear"
    exclude_modules = None
    if args.model_family == "chexagent":
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        exclude_modules = r".*visual.*"
    peft_config = deps["LoraConfig"](
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_rank,
        bias="none",
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save or None,
    )
    model = deps["get_peft_model"](model, peft_config)
    _freeze_vision_encoder(model, train_connector=args.train_connector)
    model.print_trainable_parameters()

    training_args = deps["SFTConfig"](
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=args.learning_rate,
        max_grad_norm=0.3,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        tf32=True,
        report_to=[] if args.report_to == "none" else [args.report_to],
        loss_type="nll",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
        seed=args.seed,
        data_seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )
    collator = (
        CheXagentSFTCollator(processor, max_length=args.max_sequence_length)
        if args.model_family == "chexagent"
        else MedGemmaSFTCollator(processor, max_length=args.max_sequence_length)
    )
    trainer = deps["SFTTrainer"](
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(adapter_dir)
    (output_dir / "trainer_log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2), encoding="utf-8"
    )
    metadata = {
        "base_model": args.model_name,
        "model_family": args.model_family,
        "model_revision": model_revision or "",
        "adapter_dir": str(adapter_dir),
        "manifest": str(args.manifest),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "test_split_consumed": False,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "retrieval_conditioned_train_examples": sum(
            bool(row["used_retrieval"]) for row in train_rows
        ),
        "retrieval_probability": args.retrieval_probability,
        "retrieval_profile": retrieval_profile,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "vision_encoder_frozen": True,
        "connector_trained": args.train_connector and args.model_family == "medgemma",
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


def _build_retrieval_evidence(args, train, evaluation):
    if args.retrieval_probability <= 0 or args.few_shot_k == 0:
        return {}, {}
    retriever = MedSiglipRetriever(args.medsiglip_model)
    queries = [*train, *evaluation]
    retrieved = retriever.retrieve(
        train,
        queries,
        top_k=args.retrieval_top_k,
        batch_size=args.retrieval_batch_size,
        cache_path=args.retrieval_cache,
    )
    evidence = {
        example.study_id: [
            {"study_id": neighbour.retrieved_study_id, "report": neighbour.prediction}
            for neighbour in neighbours
        ]
        for example, neighbours in zip(queries, retrieved, strict=True)
    }
    profile = dict(retriever.last_profile)
    del retriever
    _empty_cuda_cache()
    return evidence, profile


def _freeze_vision_encoder(model, *, train_connector: bool) -> None:
    vision_markers = ("vision_tower", "vision_model", "vision_encoder", ".visual.")
    connector_markers = ("multi_modal_projector", "mm_projector")
    for name, parameter in model.named_parameters():
        is_vision = any(marker in name for marker in vision_markers)
        is_connector = any(marker in name for marker in connector_markers)
        if is_vision or (is_connector and not train_connector):
            parameter.requires_grad = False


def _vision_quantization_skip_modules(model_family: str) -> list[str] | None:
    # CheXagent's SigLIP pooler uses nn.MultiheadAttention.out_proj directly.
    # Replacing that internal linear layer with Linear4bit corrupts its expected
    # weight layout, so the frozen vision tower remains BF16 while the decoder
    # receives NF4 QLoRA weights.
    return ["model.visual"] if model_family == "chexagent" else None


def _validate_split_contract(train_split: str, eval_split: str) -> None:
    if "test" in {train_split.lower(), eval_split.lower()}:
        raise ValueError("The test split cannot be used by the QLoRA training script.")
    if train_split == eval_split:
        raise ValueError("Training and evaluation splits must be distinct.")


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-safe multimodal QLoRA fine-tuning for radiology CXR Findings."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="google/medgemma-4b-it")
    parser.add_argument(
        "--model-family", choices=["medgemma", "chexagent"], default="medgemma"
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--medsiglip-model", default="google/medsiglip-448")
    parser.add_argument("--retrieval-cache")
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--few-shot-k", type=int, default=3)
    parser.add_argument("--retrieval-probability", type=float, default=0.5)
    parser.add_argument("--retrieval-batch-size", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-connector", action="store_true")
    parser.add_argument(
        "--train-embedding-layers", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--report-to", choices=["none", "tensorboard"], default="tensorboard")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id")
    return parser


def _dependencies():
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover - optional GPU training dependencies
        raise ImportError("Install training dependencies with `pip install -e .[finetune]`.") from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForImageTextToText": AutoModelForImageTextToText,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoProcessor": AutoProcessor,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


if __name__ == "__main__":
    main()
