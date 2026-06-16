from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.schema import load_jsonl
from nesy_gen.models.report_generator import (
    GeneratorConfig,
    ReportGenerationDataset,
    build_processor_tokenizer_model,
    collate_generation_batch,
    require_torch_transformers,
    save_generator_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an image-to-report generator.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder-model", default="microsoft/swin-tiny-patch4-window7-224")
    parser.add_argument("--decoder-model", default="distilgpt2")
    parser.add_argument("--pretrained-vision-encoder-decoder-model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-target-length", type=int, default=160)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-val-examples", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    deps = require_torch_transformers()
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]
    get_linear_schedule_with_warmup = deps["get_linear_schedule_with_warmup"]
    torch.manual_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    examples = load_jsonl(args.manifest)
    train_examples = [example for example in examples if example.split == "train" and example.image_path]
    val_examples = [example for example in examples if example.split == "val" and example.image_path]
    if args.max_train_examples:
        train_examples = train_examples[: args.max_train_examples]
    if args.max_val_examples:
        val_examples = val_examples[: args.max_val_examples]
    if not train_examples:
        raise ValueError("No training examples with image_path found.")

    config = GeneratorConfig(
        encoder_model=args.encoder_model,
        decoder_model=args.decoder_model,
        pretrained_vision_encoder_decoder_model=args.pretrained_vision_encoder_decoder_model,
        max_target_length=args.max_target_length,
    )
    save_generator_config(out / "generator_config.json", config)
    image_processor, tokenizer, model = build_processor_tokenizer_model(config)
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    train_dataset = ReportGenerationDataset(
        train_examples,
        image_processor,
        tokenizer,
        max_target_length=args.max_target_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_generation_batch,
    )
    val_loader = None
    if val_examples:
        val_loader = DataLoader(
            ReportGenerationDataset(
                val_examples,
                image_processor,
                tokenizer,
                max_target_length=args.max_target_length,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_generation_batch,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(1, len(train_loader) * args.epochs // args.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.05 * total_steps)),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    history_path = out / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(tqdm(train_loader, desc=f"train epoch {epoch}"), start=1):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                decoder_attention_mask = batch["decoder_attention_mask"].to(device)
                with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                    output = model(
                        pixel_values=pixel_values,
                        labels=labels,
                        decoder_attention_mask=decoder_attention_mask,
                    )
                    loss = output.loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
                if step % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()) * args.gradient_accumulation_steps)
            train_loss = sum(losses) / len(losses)
            val_loss = evaluate_loss(model, val_loader, device, torch) if val_loader is not None else None
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss or ""})
            handle.flush()
            print({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}, flush=True)

    model.save_pretrained(out)
    image_processor.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "train_summary.json").write_text(
        json.dumps(
            {
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
                "device": str(device),
                "history": str(history_path),
                "checkpoint": str(out),
                "pretrained_vision_encoder_decoder_model": args.pretrained_vision_encoder_decoder_model,
                "freeze_encoder": args.freeze_encoder,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved checkpoint to {out}")


def evaluate_loss(model, val_loader, device, torch) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="validation"):
            output = model(
                pixel_values=batch["pixel_values"].to(device),
                labels=batch["labels"].to(device),
                decoder_attention_mask=batch["decoder_attention_mask"].to(device),
            )
            losses.append(float(output.loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses) if losses else 0.0


if __name__ == "__main__":
    main()
