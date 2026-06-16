from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nesy_gen.data.schema import load_jsonl
from nesy_gen.models.blip_report_generator import (
    BlipGeneratorConfig,
    BlipReportDataset,
    build_blip_processor_model,
    collate_blip_batch,
    require_blip_dependencies,
    save_blip_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BLIP for radiology report generation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-target-length", type=int, default=160)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-val-examples", type=int)
    parser.add_argument("--freeze-vision", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    deps = require_blip_dependencies()
    torch = deps["torch"]
    DataLoader = deps["DataLoader"]
    get_linear_schedule_with_warmup = deps["get_linear_schedule_with_warmup"]
    torch.manual_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = BlipGeneratorConfig(model_name=args.model_name, max_target_length=args.max_target_length)
    save_blip_config(out / "blip_generator_config.json", config)
    processor, model = build_blip_processor_model(config)
    if args.freeze_vision and hasattr(model, "vision_model"):
        for param in model.vision_model.parameters():
            param.requires_grad = False

    examples = load_jsonl(args.manifest)
    train_examples = [example for example in examples if example.split == "train" and example.image_path]
    val_examples = [example for example in examples if example.split == "val" and example.image_path]
    if args.max_train_examples:
        train_examples = train_examples[: args.max_train_examples]
    if args.max_val_examples:
        val_examples = val_examples[: args.max_val_examples]

    train_loader = DataLoader(
        BlipReportDataset(
            train_examples,
            processor,
            max_target_length=args.max_target_length,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_blip_batch,
    )
    val_loader = DataLoader(
        BlipReportDataset(
            val_examples,
            processor,
            max_target_length=args.max_target_length,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_blip_batch,
    ) if val_examples else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
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
            for step, batch in enumerate(tqdm(train_loader, desc=f"blip train epoch {epoch}"), start=1):
                with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                    output = model(
                        pixel_values=batch["pixel_values"].to(device),
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device),
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
            val_loss = _evaluate_loss(model, val_loader, device, torch) if val_loader else None
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss or ""})
            handle.flush()
            print({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}, flush=True)

    model.save_pretrained(out)
    processor.save_pretrained(out)
    (out / "train_summary.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "train_examples": len(train_examples),
                "val_examples": len(val_examples),
                "device": str(device),
                "history": str(history_path),
                "checkpoint": str(out),
                "freeze_vision": args.freeze_vision,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved BLIP checkpoint to {out}")


def _evaluate_loss(model, val_loader, device, torch) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="blip validation"):
            output = model(
                pixel_values=batch["pixel_values"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            losses.append(float(output.loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses) if losses else 0.0


if __name__ == "__main__":
    main()

