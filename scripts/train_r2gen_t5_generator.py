from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.schema import load_jsonl
from nesy_gen.baselines.visual_retrieval import visual_evidence_map
from nesy_gen.generation.constrained_decoding import PrimeKGDecodingConstraintBuilder
from nesy_gen.models.r2gen_t5 import (
    R2GenT5Config,
    R2GenT5Dataset,
    R2GenT5Model,
    collate_r2gen_t5_batch,
    require_r2gen_t5_dependencies,
)
from nesy_gen.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Vision-T5 image-to-report generator.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-model-name", default="t5-small")
    parser.add_argument("--tokenizer-name", default="t5-small")
    parser.add_argument(
        "--visual-backbone",
        choices=["resnet101", "convnext_base", "efficientnet_v2_s", "swin_t", "densenet121"],
        default="resnet101",
    )
    parser.add_argument("--freeze-visual-encoder", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Stop after this many non-improving validation epochs; 0 disables early stopping.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-target-length", type=int, default=160)
    parser.add_argument("--visual-seq-len", type=int, default=128)
    parser.add_argument("--dropout-prob", type=float, default=0.1)
    parser.add_argument("--target-prefix", default="generate report: ")
    parser.add_argument(
        "--retrieval-conditioning",
        action="store_true",
        help="Condition T5 on reports retrieved solely from frozen image features.",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=3)
    parser.add_argument("--max-evidence-length", type=int, default=192)
    parser.add_argument("--retrieval-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-val-examples", type=int)
    parser.add_argument(
        "--progress-style",
        choices=["plain", "tqdm", "none"],
        default="plain",
        help="plain is safest for Colab subprocess output; tqdm is best for terminals.",
    )
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument(
        "--graph-training-mode",
        choices=["none", "primekg_token"],
        default="none",
        help="Optional graph-aware training. Default keeps plain Vision-T5 intact.",
    )
    parser.add_argument("--graph-loss-nodes-csv")
    parser.add_argument("--graph-token-loss-weight", type=float, default=0.0)
    parser.add_argument("--unsupported-token-loss-weight", type=float, default=0.0)
    parser.add_argument("--graph-loss-max-terms", type=int, default=2500)
    parser.add_argument(
        "--graph-loss-source",
        choices=["reference", "indication_reference"],
        default="indication_reference",
    )
    parser.add_argument("--fp16", action="store_true", help="Mixed-precision with float16 + GradScaler.")
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Mixed-precision with bfloat16 (recommended on A100 — no GradScaler needed, more stable).",
    )
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=None,
        help=(
            "Separate LR for the visual backbone when --freeze-visual-encoder is NOT set. "
            "Typical value: 1/10 of --learning-rate (e.g. 5e-6 when LR=5e-5). "
            "When omitted the backbone uses the same LR as the rest of the model."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes for parallel image loading (0 = main thread only).",
    )
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Apply torch.compile to the T5 decoder (PyTorch >= 2.0, ~20%% faster on A100).",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    deps = require_r2gen_t5_dependencies()
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]
    get_linear_schedule_with_warmup = deps["get_linear_schedule_with_warmup"]

    examples = load_jsonl(args.manifest)
    train_examples = [example for example in examples if example.split == "train" and example.image_path]
    val_examples = [example for example in examples if example.split == "val" and example.image_path]
    if args.max_train_examples:
        train_examples = train_examples[: args.max_train_examples]
    if args.max_val_examples:
        val_examples = val_examples[: args.max_val_examples]
    if not train_examples or not val_examples:
        raise ValueError("Need non-empty train and val examples with image paths.")

    config = R2GenT5Config(
        text_model_name=args.text_model_name,
        tokenizer_name=args.tokenizer_name,
        visual_backbone=args.visual_backbone,
        freeze_visual_encoder=args.freeze_visual_encoder,
        image_size=args.image_size,
        max_target_length=args.max_target_length,
        visual_seq_len=args.visual_seq_len,
        dropout_prob=args.dropout_prob,
        target_prefix=args.target_prefix,
        use_retrieval_conditioning=args.retrieval_conditioning,
        max_evidence_length=args.max_evidence_length,
    )
    model = R2GenT5Model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_evidence = None
    val_evidence = None
    if args.retrieval_conditioning:
        if not args.freeze_visual_encoder:
            raise ValueError(
                "--retrieval-conditioning requires --freeze-visual-encoder so retrieval "
                "features remain stable and reproducible during training."
            )
        print("Building leave-one-study-out visual RAG evidence...", flush=True)
        train_evidence = visual_evidence_map(
            model,
            train_examples,
            train_examples,
            top_k=args.retrieval_top_k,
            batch_size=args.retrieval_batch_size,
            progress_desc="train visual RAG evidence",
        )
        val_evidence = visual_evidence_map(
            model,
            train_examples,
            val_examples,
            top_k=args.retrieval_top_k,
            batch_size=args.retrieval_batch_size,
            progress_desc="validation visual RAG evidence",
        )
        print(
            f"Visual RAG evidence ready: train={len(train_evidence)} val={len(val_evidence)}",
            flush=True,
        )

    # Mixed-precision setup: BF16 preferred on A100 (no scaler), FP16 elsewhere.
    use_bf16 = args.bf16 and device.type == "cuda"
    use_fp16 = args.fp16 and not use_bf16 and device.type == "cuda"
    amp_enabled = use_bf16 or use_fp16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(
        f"AMP: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'disabled'} "
        f"| device: {device}",
        flush=True,
    )

    if args.compile_model:
        if hasattr(torch, "compile"):
            model.text_model = torch.compile(model.text_model, mode="reduce-overhead")
            print("torch.compile applied to text model (mode=reduce-overhead).", flush=True)
        else:
            print("torch.compile not available (PyTorch < 2.0), skipping.", flush=True)

    pin = device.type == "cuda"
    num_workers = args.num_workers
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin,
    )
    if num_workers > 0:
        loader_kwargs.update(prefetch_factor=2, persistent_workers=True)
    train_loader = DataLoader(
        R2GenT5Dataset(
            train_examples,
            model.tokenizer,
            max_target_length=args.max_target_length,
            target_prefix=args.target_prefix,
            image_size=args.image_size,
            evidence_by_study_id=train_evidence,
            max_evidence_length=args.max_evidence_length,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_r2gen_t5_batch,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        R2GenT5Dataset(
            val_examples,
            model.tokenizer,
            max_target_length=args.max_target_length,
            target_prefix=args.target_prefix,
            image_size=args.image_size,
            evidence_by_study_id=val_evidence,
            max_evidence_length=args.max_evidence_length,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_r2gen_t5_batch,
        **loader_kwargs,
    )

    backbone_lr = args.backbone_learning_rate
    if backbone_lr is not None and not args.freeze_visual_encoder:
        # Differential LR: lower rate for pretrained backbone, full rate for T5 + projection
        visual_extractor_ids = {id(p) for p in model.visual_extractor.parameters()}
        param_groups = [
            {
                "params": [p for p in model.text_model.parameters() if p.requires_grad],
                "lr": args.learning_rate,
            },
            {
                "params": [p for p in model.visual_projection.parameters() if p.requires_grad],
                "lr": args.learning_rate,
            },
            {
                "params": [
                    p for p in model.visual_extractor.parameters()
                    if p.requires_grad and id(p) in visual_extractor_ids
                ],
                "lr": backbone_lr,
            },
        ]
        print(
            f"Differential LR: T5+projection={args.learning_rate:.2e}, backbone={backbone_lr:.2e}",
            flush=True,
        )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    total_steps = max(1, args.epochs * len(train_loader) // args.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.05 * total_steps)),
        num_training_steps=total_steps,
    )
    # GradScaler only for FP16 (BF16 has enough dynamic range — no overflow)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    graph_loss_helper = _build_graph_loss_helper(args, model)
    if graph_loss_helper is None:
        print("Graph-aware training: disabled. Training plain Vision-T5.", flush=True)
    else:
        print(
            (
                "Graph-aware training: enabled "
                f"mode={args.graph_training_mode} "
                f"graph_token_weight={args.graph_token_loss_weight} "
                f"unsupported_weight={args.unsupported_token_loss_weight}"
            ),
            flush=True,
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        seen_examples = 0
        optimizer.zero_grad(set_to_none=True)
        train_progress = _progress_iter(
            train_loader,
            desc=f"train {epoch + 1}/{args.epochs}",
            style=args.progress_style,
        )
        train_start = time.monotonic()
        for step, batch in enumerate(train_progress):
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                outputs = model.forward(
                    images,
                    labels=labels,
                    evidence_input_ids=batch.get("evidence_input_ids"),
                    evidence_attention_mask=batch.get("evidence_attention_mask"),
                )
                generation_loss = outputs.loss
                graph_loss, graph_metrics = _graph_training_loss(
                    graph_loss_helper,
                    outputs.logits,
                    labels,
                    batch,
                    torch,
                    device,
                )
                loss = generation_loss + graph_loss
                loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0 or step + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            total_loss += loss.item() * args.gradient_accumulation_steps
            seen_examples += len(batch["study_id"])
            _update_progress(
                train_progress,
                style=args.progress_style,
                desc=f"train {epoch + 1}/{args.epochs}",
                step=step + 1,
                total=len(train_loader),
                start_time=train_start,
                log_every=args.progress_every,
                metrics={
                    "loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                    "avg": f"{total_loss / (step + 1):.4f}",
                    "gen": f"{generation_loss.item():.4f}",
                    **graph_metrics,
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "seen": str(seen_examples),
                    **_gpu_progress(torch, device),
                },
            )

        model.eval()
        val_loss = 0.0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
            val_progress = _progress_iter(
                val_loader,
                desc=f"valid {epoch + 1}/{args.epochs}",
                style=args.progress_style,
            )
            val_start = time.monotonic()
            for step, batch in enumerate(val_progress):
                images = batch["image"].to(device)
                labels = batch["labels"].to(device)
                outputs = model.forward(
                    images,
                    labels=labels,
                    evidence_input_ids=batch.get("evidence_input_ids"),
                    evidence_attention_mask=batch.get("evidence_attention_mask"),
                )
                generation_loss = outputs.loss
                graph_loss, graph_metrics = _graph_training_loss(
                    graph_loss_helper,
                    outputs.logits,
                    labels,
                    batch,
                    torch,
                    device,
                )
                loss = generation_loss + graph_loss
                val_loss += loss.item()
                _update_progress(
                    val_progress,
                    style=args.progress_style,
                    desc=f"valid {epoch + 1}/{args.epochs}",
                    step=step + 1,
                    total=len(val_loader),
                    start_time=val_start,
                    log_every=args.progress_every,
                    metrics={
                        "loss": f"{loss.item():.4f}",
                        "avg": f"{val_loss / (step + 1):.4f}",
                        "gen": f"{generation_loss.item():.4f}",
                        **graph_metrics,
                        **_gpu_progress(torch, device),
                    },
                )

        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
        }
        history.append(row)
        print(row)
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            epochs_without_improvement = 0
            model.save_pretrained(out)
            (out / "best_checkpoint.json").write_text(
                json.dumps(
                    {
                        "epoch": row["epoch"],
                        "val_loss": row["val_loss"],
                        "selection_rule": "minimum validation loss",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Saved new best checkpoint: epoch={row['epoch']} val={best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print(
                    f"Early stopping after {epochs_without_improvement} non-improving epochs.",
                    flush=True,
                )
                break

    (out / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best Vision-T5 checkpoint saved to {out} (val_loss={best_val_loss:.4f})")


def _build_graph_loss_helper(args, model):
    if args.graph_training_mode == "none":
        return None
    if args.graph_training_mode != "primekg_token":
        raise ValueError(f"Unsupported graph training mode: {args.graph_training_mode}")
    if not args.graph_loss_nodes_csv:
        raise ValueError("--graph-loss-nodes-csv is required for primekg_token graph training.")
    if args.graph_token_loss_weight <= 0.0 and args.unsupported_token_loss_weight <= 0.0:
        raise ValueError("At least one graph loss weight must be above zero.")

    nodes = pd.read_csv(args.graph_loss_nodes_csv)
    builder = PrimeKGDecodingConstraintBuilder(
        nodes,
        model.tokenizer,
        max_penalty_terms=args.graph_loss_max_terms,
    )
    return {
        "builder": builder,
        "graph_token_loss_weight": float(args.graph_token_loss_weight),
        "unsupported_token_loss_weight": float(args.unsupported_token_loss_weight),
        "graph_loss_source": args.graph_loss_source,
    }


def _graph_training_loss(helper, logits, labels, batch, torch, device):
    if helper is None:
        return torch.zeros((), device=device), {}

    builder = helper["builder"]
    evidence_texts = _graph_loss_evidence_texts(batch, source=helper["graph_loss_source"])
    constraints = [builder.build(text) for text in evidence_texts]

    safe_labels = labels.clamp_min(0)
    valid_mask = labels.ne(-100)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    graph_token_loss = torch.zeros((), device=device)
    unsupported_loss = torch.zeros((), device=device)
    graph_token_rows = 0
    unsupported_rows = 0
    probs = None

    for row_idx, constraint in enumerate(constraints):
        row_valid = valid_mask[row_idx]
        if constraint.supported_token_ids and helper["graph_token_loss_weight"] > 0.0:
            supported = torch.tensor(
                sorted(constraint.supported_token_ids),
                dtype=torch.long,
                device=device,
            )
            support_mask = row_valid & torch.isin(safe_labels[row_idx], supported)
            if support_mask.any():
                graph_token_loss = graph_token_loss + token_nll[row_idx][support_mask].mean()
                graph_token_rows += 1

        if constraint.penalized_token_ids and helper["unsupported_token_loss_weight"] > 0.0:
            if probs is None:
                probs = torch.nn.functional.softmax(logits, dim=-1)
            penalized = torch.tensor(
                sorted(constraint.penalized_token_ids),
                dtype=torch.long,
                device=device,
            )
            if row_valid.any():
                row_mass = probs[row_idx, :, penalized].sum(dim=-1)
                unsupported_loss = unsupported_loss + row_mass[row_valid].mean()
                unsupported_rows += 1

    if graph_token_rows:
        graph_token_loss = graph_token_loss / graph_token_rows
    if unsupported_rows:
        unsupported_loss = unsupported_loss / unsupported_rows

    weighted = (
        helper["graph_token_loss_weight"] * graph_token_loss
        + helper["unsupported_token_loss_weight"] * unsupported_loss
    )
    return weighted, {
        "graph_raw": f"{graph_token_loss.item():.4f}",
        "graph_w": f"{weighted.item():.4f}",
        "unsup": f"{unsupported_loss.item():.4f}",
    }


def _graph_loss_evidence_texts(batch, *, source: str) -> list[str]:
    if source == "reference":
        return [str(report) for report in batch["report"]]
    return [
        " ".join(part for part in [str(indication), str(report)] if part)
        for indication, report in zip(batch["indication"], batch["report"], strict=True)
    ]


def _gpu_progress(torch, device) -> dict[str, str]:
    if getattr(device, "type", "") != "cuda" or not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    return {"gpu_gb": f"{allocated:.1f}/{reserved:.1f}"}


def _progress_iter(iterable, *, desc: str, style: str):
    if style == "tqdm":
        return tqdm(
            iterable,
            desc=desc,
            dynamic_ncols=True,
            leave=True,
            file=sys.stdout,
            disable=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )
    return iterable


def _update_progress(
    progress,
    *,
    style: str,
    desc: str,
    step: int,
    total: int,
    start_time: float,
    log_every: int,
    metrics: dict[str, str],
) -> None:
    if style == "none":
        return
    if style == "tqdm":
        progress.set_postfix(metrics)
        return
    log_every = max(1, log_every)
    if step % log_every != 0 and step != total:
        return
    elapsed = max(0.0, time.monotonic() - start_time)
    rate = step / elapsed if elapsed else 0.0
    remaining = (total - step) / rate if rate else 0.0
    metric_text = " ".join(f"{key}={value}" for key, value in metrics.items())
    print(
        (
            f"{desc} {_ascii_bar(step, total)} {step}/{total} "
            f"{100 * step / max(1, total):5.1f}% "
            f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(remaining)} "
            f"{metric_text}"
        ),
        flush=True,
    )


def _ascii_bar(step: int, total: int, *, width: int = 24) -> str:
    filled = int(width * step / max(1, total))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


if __name__ == "__main__":
    main()
