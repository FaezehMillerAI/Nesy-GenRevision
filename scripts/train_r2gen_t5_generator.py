from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.schema import load_jsonl
from nesy_gen.models.r2gen_t5 import (
    R2GenT5Config,
    R2GenT5Dataset,
    R2GenT5Model,
    collate_r2gen_t5_batch,
    require_r2gen_t5_dependencies,
)
from nesy_gen.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an R2Gen-style ResNet101 + T5 report generator.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-model-name", default="t5-small")
    parser.add_argument("--tokenizer-name", default="t5-small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-target-length", type=int, default=160)
    parser.add_argument("--visual-seq-len", type=int, default=128)
    parser.add_argument("--dropout-prob", type=float, default=0.1)
    parser.add_argument("--target-prefix", default="generate report: ")
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
    parser.add_argument("--fp16", action="store_true")
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
        max_target_length=args.max_target_length,
        visual_seq_len=args.visual_seq_len,
        dropout_prob=args.dropout_prob,
        target_prefix=args.target_prefix,
    )
    model = R2GenT5Model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        R2GenT5Dataset(
            train_examples,
            model.tokenizer,
            max_target_length=args.max_target_length,
            target_prefix=args.target_prefix,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_r2gen_t5_batch,
    )
    val_loader = DataLoader(
        R2GenT5Dataset(
            val_examples,
            model.tokenizer,
            max_target_length=args.max_target_length,
            target_prefix=args.target_prefix,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_r2gen_t5_batch,
    )

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
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    history = []
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
            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                loss = model.forward(images, labels=labels).loss
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
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "seen": str(seen_examples),
                    **_gpu_progress(torch, device),
                },
            )

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_progress = _progress_iter(
                val_loader,
                desc=f"valid {epoch + 1}/{args.epochs}",
                style=args.progress_style,
            )
            val_start = time.monotonic()
            for step, batch in enumerate(val_progress):
                images = batch["image"].to(device)
                labels = batch["labels"].to(device)
                loss = model.forward(images, labels=labels).loss
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

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    (out / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved R2Gen-T5 checkpoint to {out}")


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
